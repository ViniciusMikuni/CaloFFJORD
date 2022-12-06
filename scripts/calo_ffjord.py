import numpy as np
import os,re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import horovod.tensorflow.keras as hvd
import preprocessing
import argparse
import tensorflow.keras.backend as K
import pickle
# import tensorflow_addons as tfa

import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
import fjord_regularization
#tf.random.set_seed(1233)


class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size,is_1D=False, **kwargs):
        super().__init__(**kwargs)
        if is_1D:
            self.projection = layers.Conv1D(
                filters=embed_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="VALID",
            )
        else:
            self.projection = layers.Conv3D(
                filters=embed_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="VALID",
            )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

    
class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens


class BACKBONE_ODE(keras.Model):
    """ODE backbone network implementation"""
    def __init__(self, data_shape,num_cond,config,name='mlp_ode',use_1D=False,num_stack=1):
        super(BACKBONE_ODE, self).__init__()
        if config is None:
            raise ValueError("Config file not given")
        self._num_cond = num_cond
        self._data_shape = data_shape
        self._data_shape_flat = [np.prod(data_shape)]
        self._num_stack = num_stack
        
        self.config = config
        #config file with ML parameters to be used during training        
        self.activation = self.config['ACT']
        self.use_1D=use_1D

        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_cond = Input((self._num_cond))
        #conditional=inputs_time
        
        conditional = tf.concat([inputs_time,inputs_cond],-1)
        inputs,outputs = self.ViTModel(conditional)
        self._model = keras.Model(inputs=[inputs,inputs_time,inputs_cond],outputs=outputs)
                        

    def call(self, t, data,conditional):
        t_reshape = t*tf.ones_like(conditional,dtype=tf.float32)
        return self._model([data,t_reshape,conditional])


    def DenseModel(self,time_embed):
        inputs = Input((self._data_shape))
        nlayers =self.config['NLAYERS']
        dense_sizes = self.config['LAYER_SIZE']
        skip_layers = []
        
        #Encoder
        layer_encoded = self.time_dense(inputs,time_embed,dense_sizes[0])
        skip_layers.append(layer_encoded)
        for ilayer in range(1,nlayers):
            layer_encoded=self.time_dense(layer_encoded,time_embed,dense_sizes[ilayer])
            skip_layers.append(layer_encoded)
        skip_layers = skip_layers[::-1] #reverse the order


        #Decoder
        layer_decoded = skip_layers[0]
        for ilayer in range(len(skip_layers)-1):
            layer_decoded = self.time_dense(layer_decoded,time_embed,dense_sizes[len(skip_layers)-2-ilayer])
            #layer_decoded = (layer_decoded+ skip_layers[ilayer+1])
        
            
        outputs = layers.Dense(self._data_shape[0],activation=None,use_bias=True)(layer_decoded)
        return inputs, outputs

    def reshape_cond(self,cond,shape):
        time_layer = tf.reshape(cond,(-1,1,cond.shape[-1]))
        num_patches = shape[1]
        time_layer = tf.tile(time_layer,[1,num_patches,1])
        return time_layer

    def ViTModel(self,time_embed):
        inputs = Input((self._data_shape_flat))
        if self.use_1D:
            time_layer = tf.reshape(time_embed,(-1,1,time_embed.shape[-1]))
        else:
            time_layer = tf.reshape(time_embed,(-1,1,1,1,time_embed.shape[-1]))
            
        time_layer = tf.tile(time_layer,[1]+self._data_shape)
        transformer_layers = self.config['NLAYERS']
        num_heads=self.config['NHEADS']
        projection_dim = self.config['PROJECTION_SIZE']
        mlp_dim = self.config['MLP_SIZE']

        
        inputs_reshape = tf.concat([tf.reshape(inputs,[-1]+self._data_shape),time_layer],-1)
        #inputs_reshape = tf.reshape(inputs,[-1]+self._data_shape)+time_layer
        patches = TubeletEmbedding(embed_dim=projection_dim,is_1D=self.use_1D,
                                   patch_size=self.config['STRIDE'])(inputs_reshape)
        # Encode patches.
        encoded_patches = PositionalEncoder(embed_dim=projection_dim)(patches)
        reshaped_time = self.reshape_cond(time_embed,tf.shape(encoded_patches))

        encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)        
        for _ in range(transformer_layers):
            # Layer normalization 1.
            # x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            x1 =encoded_patches

            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim//num_heads, dropout=0.0
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            
            # Layer normalization 2.
            #x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            #x3=tf.concat([x2,reshaped_time],-1)
            x3=x2
            # MLP.
            #tf.nn.gelu
            x3 = layers.Dense(2*projection_dim,activation=self.activation)(x3)
            x3 = layers.Dense(projection_dim,activation=self.activation)(x3)
            #time_vit = layers.Dense(projection_dim,activation=self.activation)(reshaped_time)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])


        #representation = layers.Add()([encoded_patches, encoded_patches1])
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # representation = self.time_conv(representation,time_embed,
        #                                 projection_dim//2,
        #                                 kernel_size=3,padding='same',
        #                                 stride=self.config['STRIDE'],transpose=True)
        
        # layer = layers.Conv1D(1,kernel_size=1,padding="same",
        #                       strides=1,activation=None)(representation)
        # outputs = layers.Flatten()(layer)
        # return inputs, outputs
        
        pooling = layers.GlobalAvgPool1D()(representation)
        representation = layers.Flatten()(representation)
        # representation = layers.Dropout(0.1)(representation)
        layer = self.time_dense(tf.concat([representation,pooling],-1),time_embed,mlp_dim)
        # layer = self.time_dense(representation,time_embed,mlp_dim)
        layer = layers.Dropout(0.1)(layer)
        layer = self.time_dense(layer,time_embed,mlp_dim//2)
        outputs = layers.Dense(self._data_shape_flat[0],use_bias=True)(layer)
            
        return inputs, outputs        
    
    def ConvModel(self,time_embed):
        inputs = Input((self._data_shape_flat))
        inputs_reshape = tf.reshape(inputs,[-1]+self._data_shape)
        stride_size=self.config['STRIDE']
        kernel_size =self.config['KERNEL']
        nlayers =self.config['NLAYERS']
        conv_sizes = [self._num_stack * l for l in self.config['LAYER_SIZE']]
        skip_layers = []
        #Encoder

        skip_list=self.ConvEncoder(inputs_reshape,time_embed,stride_size=stride_size,
                              kernel_size=kernel_size,nlayers=nlayers,conv_sizes=conv_sizes)
        out=self.ConvDecoder(skip_list,time_embed,stride_size=stride_size,
                             kernel_size=kernel_size,nlayers=nlayers,conv_sizes=conv_sizes)
        

        # outputs = out
        outputs = layers.Flatten()(out) #Flatten the output 
        return inputs, outputs

    def ConvEncoder(self,inputs,time_embed,stride_size,kernel_size,nlayers,conv_sizes):
        skip_layers = []
        layer_encoded = inputs
        skip_layers.append(layer_encoded)
        for ilayer in range(nlayers):
            layer_encoded = self.time_conv(layer_encoded,time_embed,conv_sizes[ilayer],
                                           kernel_size=kernel_size,padding='same',
                                           stride=stride_size)
            # print(layer_encoded,nlayers,'encoded')
            skip_layers.append(layer_encoded)

        skip_layers = skip_layers[::-1] #reverse the order
        return skip_layers

    def ConvDecoder(self,skip_layers,time_embed,stride_size,kernel_size,nlayers,conv_sizes):

        layer_decoded = skip_layers[0]
        for ilayer in range(len(skip_layers)-1):
            layer_decoded = self.time_conv(layer_decoded,time_embed,
                                           conv_sizes[len(skip_layers)-2-ilayer],
                                           kernel_size=kernel_size,padding='same',
                                           stride=stride_size,transpose=True)

            # layer_decoded = (layer_decoded+ skip_layers[ilayer+1])/np.sqrt(2)
            layer_decoded = tf.concat([layer_decoded, skip_layers[ilayer+1]],-1)

            
        if len(self._data_shape) == 3:
            outputs = layers.Conv2D(1,kernel_size=kernel_size,padding="same",
                                    strides=1,activation=None,use_bias=True)(layer_decoded)
        else:
            layer_decoded = layers.Conv3D(conv_sizes[-1],kernel_size=kernel_size,padding="same",
                                    strides=1,activation=None,use_bias=True)(layer_decoded)
            layer_decoded = self.activate(layer_decoded)
            outputs = layers.Conv3D(1,kernel_size=1,padding="same",
                                    strides=1,activation=None,use_bias=True)(layer_decoded)
        return outputs
    
    def time_conv(self,input_layer,embed,hidden_size,stride=1,kernel_size=2,padding="same",activation=True,transpose=False):

        if len(self._data_shape) == 2:
            #Incorporate the time information to each layer used in the model
            time_layer = tf.reshape(embed,(-1,1,embed.shape[-1]))

            time_layer = tf.tile(time_layer,(1,input_layer.shape[1],1))
            layer = tf.concat([input_layer,time_layer],-1)
            if transpose:            
                layer = layers.Conv1DTranspose(hidden_size,kernel_size=kernel_size,padding=padding,
                                               strides=stride,activation=None)(layer)
                layer = self.activate(layer)
                layer = layers.Conv1D(hidden_size,kernel_size=kernel_size,padding=padding,
                                      strides=1,activation=None)(layer)
            else:
                layer = layers.Conv1D(hidden_size,kernel_size=kernel_size,padding=padding,
                                      strides=stride,activation=None)(layer)
                layer = self.activate(layer)
                layer = layers.Conv1D(hidden_size,kernel_size=kernel_size,padding=padding,
                                      strides=1,activation=None)(layer)

        elif len(self._data_shape) == 3:
            #Incorporate the time information to each layer used in the model
            time_layer = tf.reshape(embed,(-1,1,1,embed.shape[-1]))

            time_layer = tf.tile(time_layer,(1,input_layer.shape[1],input_layer.shape[2],1))
            layer = tf.concat([input_layer,time_layer],-1)
            if transpose:            
                layer = layers.Conv2DTranspose(hidden_size,kernel_size=kernel_size,padding=padding,
                                               strides=stride,activation=None)(layer)
                layer = self.activate(layer)
                layer = layers.Conv2D(hidden_size,kernel_size=kernel_size,padding=padding,
                                      strides=1,activation=None)(layer)
            else:
                layer = layers.Conv2D(hidden_size,kernel_size=kernel_size,padding=padding,
                                      strides=stride,activation=None)(layer)
                layer = self.activate(layer)
                layer = layers.Conv2D(hidden_size,kernel_size=kernel_size,padding=padding,
                                      strides=1,activation=None)(layer)


        elif len(self._data_shape) == 4:
            #Incorporate the time information to each layer used in the model
            time_layer = tf.reshape(embed,(-1,1,1,1,embed.shape[-1]))
            time_layer = tf.tile(time_layer,(1,input_layer.shape[1],input_layer.shape[2],
                                             input_layer.shape[3],1))
        
            layer = tf.concat([input_layer,time_layer],-1)
            if transpose:
                layer = layers.Conv3DTranspose(hidden_size,kernel_size=kernel_size,padding=padding,
                                               strides=stride,activation=None)(layer)
                layer = self.activate(layer)
                layer = layers.Conv3D(hidden_size,kernel_size=kernel_size,padding=padding,
                                      strides=1,activation=None)(layer)
                
            else:                
                layer = layers.Conv3D(hidden_size,kernel_size=kernel_size,padding=padding,
                                      strides=1,activation=None)(layer)
                layer = self.activate(layer)
                layer = layers.Conv3D(hidden_size,kernel_size=kernel_size,padding=padding,
                                      strides=stride,activation=None)(layer)

                
        # layer = layers.BatchNormalization()(layer)
        # layer = layers.Dropout(0.1)(layer)
        if activation:            
            return self.activate(layer)
        else:
            return layer
        

    def time_dense(self,input_layer,embed,hidden_size,activation=True):
        #Incorporate the time information to each layer used in the model
        layer = tf.concat([input_layer,embed],-1)
        layer = layers.Dense(hidden_size,activation=None)(layer)
                
        # layer = layers.BatchNormalization()(layer)
        # layer = layers.Dropout(0.1)(layer)
        if activation:            
            return self.activate(layer)
        else:
            return layer


    def activate(self,layer):
        if self.activation == 'leaky_relu':                
            return keras.layers.LeakyReLU(-0.01)(layer)
        elif self.activation == 'tanh':
            return keras.activations.tanh(layer)
        elif self.activation == 'softplus':
            return keras.activations.softplus(layer)
        elif self.activation == 'relu':
            return keras.activations.relu(layer)
        elif self.activation == 'swish':
            return keras.activations.swish(layer)
        else:
            raise ValueError("Activation function not supported!")   


    

def make_bijector_kwargs(bijector, name_to_kwargs):
    #Hack to pass the conditional information through all the bijector layers
    if hasattr(bijector, 'bijectors'):
        return {b.name: make_bijector_kwargs(b, name_to_kwargs) for b in bijector.bijectors}
    else:
        for name_regex, kwargs in name_to_kwargs.items():
            if re.match(name_regex, bijector.name):
                return kwargs
    return {}

def save_model(model,name="ffjord",checkpoint_dir = '../checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model.save_weights('{}/{}'.format(checkpoint_dir,name),save_format='tf')
    np.save('optimizer.npy', model.optimizer.get_weights())
    
def load_model(model,name="ffjord",checkpoint_dir = '../checkpoints'):
    model.load_weights('{}/{}'.format(checkpoint_dir,name)).expect_partial()
    
            
class FFJORD(keras.Model):
    def __init__(self, stacked_layers,num_output,config,trace_type='hutchinson',is_training=True,name='FFJORD'):
        super(FFJORD, self).__init__()
        self._num_output=num_output
        if config is None:
            raise ValueError("Config file not given")

        self.config = config
        
        ode_solve_fn = tfp.math.ode.DormandPrince(atol=1e-5).solve
        #Gaussian noise to trace solver
        if trace_type=='hutchinson':
            if is_training:
                trace_augmentation_fn = fjord_regularization.trace_jacobian_hutchinson
            else:
                trace_augmentation_fn = tfb.ffjord.trace_jacobian_hutchinson
        elif trace_type == 'exact':
            if is_training:
                raise Exception("Don't use exact trace for training!")
            trace_augmentation_fn = tfb.ffjord.trace_jacobian_exact #Regularization code only
        else:
            raise Exception("Invalid trace estimator")
        
        
        self.bijectors = []
        for ilayer,layer in enumerate(stacked_layers):
            if is_training:
                ffjord = fjord_regularization.FFJORD(
                    state_time_derivative_fn=layer, #Each of the models we build
                    ode_solve_fn=ode_solve_fn,
                    is_training=is_training,
                    trace_augmentation_fn=trace_augmentation_fn,
                    name='bijector{}'.format(ilayer), #Bijectors need to be named to receive conditional inputs
                    jacobian_factor = self.config['JAC'], #Regularization strength
                    kinetic_factor = self.config['REG'],
                    b = self.config['b'],
                )
            else:
                
                ffjord = tfb.FFJORD(
                    state_time_derivative_fn=layer, #Each of the  models we build
                    ode_solve_fn=ode_solve_fn,
                    trace_augmentation_fn=trace_augmentation_fn,
                    name='bijector{}'.format(ilayer),
            )
            
            self.bijectors.append(ffjord)

        #Reverse the bijector order
        self.chain = tfb.Chain(list(reversed(self.bijectors)))

        self.loss_tracker = keras.metrics.Mean(name="loss")
        #Determine the base distribution
        self.base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=np.zeros(self._num_output,dtype=np.float32),
            scale_diag=np.ones(self._num_output,dtype=np.float32)
        )
        
        self.flow=tfd.TransformedDistribution(distribution=self.base_distribution, bijector=self.chain)
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]
    

    def call(self, inputs, conditional=None):
        kwargs = make_bijector_kwargs(self.flow.bijector,{'bijector.': {'conditional':conditional }})
        return self.flow.bijector.forward(inputs,**kwargs)
        

    def log_loss(self,_x,_c):
        loss = -tf.reduce_mean(self.flow.log_prob(
            _x,
            bijector_kwargs=make_bijector_kwargs(
                self.flow.bijector, {'bijector.': {'conditional': _c}}) 
        ))
        return loss
    

    def conditional_prob(self,_x,_c):
        prob = self.flow.prob(
            _x,
            bijector_kwargs=make_bijector_kwargs(
                self.flow.bijector, {'bijector.': {'conditional': _c}})
        )
        
        return prob



    def discrepancy_slice_wasserstein(self,p1, p2):
        s = tf.shape(p1)
        if s[1] > 1:
            # For data more than one-dimensional, perform multiple random projection to 1-D
            proj = tf.random.normal([s[1], 128])
            proj *= tf.math.rsqrt(tf.reduce_sum(tf.math.square(proj), 0, keepdims=True))
            p1 = tf.matmul(p1, proj)
            p2 = tf.matmul(p2, proj)


        def sort_rows(matrix, num_rows):
            matrix_T = tf.transpose(matrix, [1, 0])
            sorted_matrix_T,_ = tf.math.top_k(matrix_T, num_rows)
            return tf.transpose(sorted_matrix_T, [1, 0])

        p1 = sort_rows(p1, s[0])
        p2 = sort_rows(p2, s[0])
        wdist = tf.reduce_mean(tf.square(p1 - p2))
        return tf.reduce_mean(wdist)


    @tf.function()
    def train_step(self, inputs):

        data,cond = inputs
        with tf.GradientTape() as tape:
            loss = self.log_loss(data,cond)
            sample = self.flow.sample(
                tf.shape(data)[0],
                bijector_kwargs=make_bijector_kwargs(
                    self.flow.bijector, {'bijector.': {'conditional': cond}})
        )

            gen_energies = sample[:,504:]
            data_energies = data[:,504:]

            loss_energy=self.discrepancy_slice_wasserstein(gen_energies,data_energies)

            loss = loss + 100*loss_energy
            #+10*loss_energy

        g = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(g, self.trainable_variables))

        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result(),'swd':loss_energy}
    
    @tf.function
    def test_step(self, inputs):
        data,cond = inputs        
        loss = self.log_loss(data,cond)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    

        
    

if __name__ == '__main__':
    #Start horovod and pin each GPU to a different process
    hvd.init()    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


    parser = argparse.ArgumentParser()
    

    parser.add_argument('--model', default='calogan', help='Name of the model to train. Options are: mnist, challenge, calogan')
    parser.add_argument('--nevts', default=-1, help='Name of the model to train. Options are: mnist, moon, calorimeter')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
    flags = parser.parse_args()
        
    model_name = flags.model #Let's try the parallel training using different models and different network architectures. Possible options are [mnist,moon,calorimeter]
        
    if model_name == 'mnist':
        from tensorflow.keras.datasets import mnist, fashion_mnist
        dataset_config = preprocessing.LoadJson('config_mnist.json')
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        ntrain,samples_train =preprocessing.MNIST_prep(X_train, y_train)
        ntest,samples_test =preprocessing.MNIST_prep(X_test, y_test)
        
        use_1D = False #Use convolutional networks for the model backbone
    elif model_name == 'calogan':        
        dataset_config = preprocessing.LoadJson('config_calogan_vit.json')
        file_path=dataset_config['FILE']
        ntrain,ntest,samples_train,samples_test = preprocessing.CaloGAN_prep(file_path,int(flags.nevts),use_logit=True)
        use_1D = True
    elif model_name == 'calochallenge':
        dataset_config = preprocessing.LoadJson('config_challenge_vit.json')
        file_path=dataset_config['FILE']
        ntrain,ntest,samples_train,samples_test = preprocessing.CaloChallenge_prep(file_path,int(flags.nevts))
        use_1D = False
    else:
        raise ValueError("Model not implemented!")
        
        
    LR = float(dataset_config['LR'])
    NUM_EPOCHS = dataset_config['MAXEPOCH']
    STACKED_FFJORDS = dataset_config['NSTACKED'] #Number of stacked transformations
    
    NUM_LAYERS = dataset_config['NLAYERS'] #Hiddden layers per bijector
    BATCH_SIZE = dataset_config['BATCH']

    
    #Stack of bijectors 
    stacked_convs = []
    for istack in range(STACKED_FFJORDS):
        conv_model = BACKBONE_ODE(dataset_config['SHAPE'], 1, config=dataset_config,use_1D=use_1D,num_stack=1)
        stacked_convs.append(conv_model)

    #Create the model
    model = FFJORD(stacked_convs,np.prod(list(dataset_config['SHAPE'])), config=dataset_config)

    opt = tf.optimizers.Adam(learning_rate=LR)

    
    if flags.load:
        load_model(model,checkpoint_dir='../checkpoint_{}'.format(dataset_config['MODEL']))
        # old_weights = np.load('optimizer.npy', allow_pickle=True)
        
    opt = hvd.DistributedOptimizer(opt)
    # Horovod: add Horovod DistributedOptimizer.
    model.compile(optimizer=opt,experimental_run_tf_function=False)
    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        ReduceLROnPlateau(patience=10, min_lr=1e-7,verbose=hvd.rank() == 0),
        EarlyStopping(patience=dataset_config['EARLYSTOP'],restore_best_weights=True),
    ]

    if hvd.rank()==0:
        checkpoint = ModelCheckpoint('../checkpoint_{}/ffjord'.format(dataset_config['MODEL']),save_best_only=True,mode='auto',
                                                               period=1,save_weights_only=True)
        callbacks.append(checkpoint)
        model(np.zeros((1,dataset_config['SHAPE'][0])),np.zeros((1,1)))
        print(model.summary())
    
    history = model.fit(
        samples_train.batch(BATCH_SIZE),
        steps_per_epoch=int(ntrain/BATCH_SIZE),
        validation_data=samples_test.batch(BATCH_SIZE),
        validation_steps=max(1,int(ntest/BATCH_SIZE)),
        epochs=NUM_EPOCHS,
        verbose=hvd.rank() == 0,
        callbacks=callbacks,
    )

    if hvd.rank() == 0:            
        save_model(model,checkpoint_dir='../checkpoint_{}'.format(dataset_config['MODEL']))
