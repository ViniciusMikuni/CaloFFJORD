import numpy as np
import os,re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import horovod.tensorflow.keras as hvd
import horovod.tensorflow as hvd_tf
import preprocessing
import argparse
import tensorflow.keras.backend as K
import pickle

import tensorflow_addons as tfa

import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
import fjord_regularization
tf.random.set_seed(1233)


class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size,is_1D=False, **kwargs):
        super().__init__(**kwargs)
        if is_1D:
            self.projection = layers.Conv1D(
                filters=embed_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="VALID",
                use_bias=False,
            )
        else:
            self.projection = layers.Conv3D(
                filters=embed_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="VALID",
                use_bias=False,
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
    def __init__(self, data_shape,num_cond,config,name='mlp_ode',
                 use_1D=False,use_dense=False):
        super(BACKBONE_ODE, self).__init__()
        if config is None:
            raise ValueError("Config file not given")
        self._num_cond = num_cond
        self._data_shape = data_shape
        self._data_shape_flat = [np.prod(data_shape)]

        
        self.config = config
        #config file with ML parameters to be used during training        
        self.activation = self.config['ACT']
        self.use_1D=use_1D


        #transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_cond = Input((self._num_cond))

        if self.config['USE_MLP_COND']:
            conditional = layers.Dense(16,activation=self.activation)(inputs_cond)
            conditional = tf.concat([inputs_time,conditional],-1)
        else:        
            conditional = tf.concat([inputs_time,inputs_cond],-1)
        

        if use_dense:
            inputs,outputs = self.DenseModel(conditional)
        else:
            inputs,outputs = self.ViTModel(conditional)

        self._model = keras.Model(inputs=[inputs,inputs_time,inputs_cond],outputs=outputs)
            
                        

    def call(self, t, data,conditional):
        t_reshape = t*tf.ones_like(conditional[:,:1],dtype=tf.float32)
        return self._model([data,t_reshape,conditional])

    
    def DenseModel(self,time_embed):
        inputs = Input((self._data_shape))
        inputs_reshape = tf.expand_dims(inputs,-1)
        nlayers =self.config['NDENSE']
        ndim=self._data_shape[0]
        nhidden = self.config['HIDDEN_MULTIPLIER']*ndim
        

        layer_encoded = self.time_dense(inputs,time_embed,nhidden)
        for ilayer in range(1,nlayers-1):
            layer_encoded=self.time_dense(layer_encoded,time_embed,nhidden)
            
        #outputs=self.time_dense(layer_encoded,time_embed,ndim,activation=False)
        outputs = layers.Dense(ndim,activation=None)(layer_encoded)
        #outputs = layers.Flatten()(layer_encoded)
        return inputs, outputs

    def ViTModel(self,time_embed):
        ''' Visual transformer model as the network backbone for the FFJORD implementation'''
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
        patches = TubeletEmbedding(embed_dim=projection_dim,is_1D=self.use_1D,
                                   patch_size=self.config['STRIDE'])(inputs_reshape)
        
        # Encode patches.
        encoded_patches = PositionalEncoder(embed_dim=projection_dim)(patches)


        
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim)(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            #x3=x2
            # MLP.
            #tf.nn.gelu

            # x3 = self.time_dense(x3,time_patch,2*projection_dim)
            # x3 = self.time_dense(x3,time_patch,projection_dim)

            x3 = layers.Dense(2*projection_dim,activation=self.activation)(x3)
            x3 = layers.Dense(projection_dim,activation=self.activation)(x3)

            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        pooling = layers.GlobalAvgPool1D()(representation)
        representation = layers.Flatten()(representation)
        representation = self.time_dense(tf.concat([representation,pooling],-1),time_embed,mlp_dim)
        #representation = layers.Dropout(0.1)(representation)
        representation = self.time_dense(representation,time_embed,mlp_dim//2)
        outputs = layers.Dense(self._data_shape_flat[0])(representation)
        return inputs, outputs





    def time_dense(self,input_layer,embed,hidden_size,activation=True):
        #Incorporate the time information to each layer used in the model
        layer = tf.concat([input_layer,embed],-1)
        layer = layers.Dense(hidden_size,activation=None)(layer)

        #layer = layers.LayerNormalization(epsilon=1e-6)(layer)
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
    def __init__(self, stacked_layers,stacked_energies,
                 num_output,num_energy,
                 config,trace_type='hutchinson',
                 is_training=True,mask=None,name='FFJORD'):
        
        super(FFJORD, self).__init__()

        if config is None:
            raise ValueError("Config file not given")

        self.config = config
        
        ode_solve_fn = tfp.math.ode.DormandPrince(
            atol=1e-3,
            # rtol=1e-3,
            # first_step_size=0.2
        ).solve

        ode_solve_fn_energy = tfp.math.ode.DormandPrince(
            atol=1e-4,
            rtol=1e-4,
            #rtol=1e-5,
            #first_step_size=0.2
        ).solve
        
        #Gaussian noise to trace solver
        if trace_type=='hutchinson':
            if is_training:
                trace_augmentation_fn = fjord_regularization.trace_jacobian_hutchinson
            else:
                trace_augmentation_fn = tfb.ffjord.trace_jacobian_hutchinson
        elif trace_type == 'exact':
            trace_augmentation_fn = tfb.ffjord.trace_jacobian_exact #Regularization code only
        else:
            raise Exception("Invalid trace estimator")
        
        
        self.bijectors = []
        self.bijectors_energy = []
        
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
                    trace_augmentation_fn=tfb.ffjord.trace_jacobian_exact,
                    #trace_augmentation_fn=trace_augmentation_fn,
                    name='bijector{}'.format(ilayer),
            )
                
            self.bijectors.append(ffjord)


        for ilayer,layer in enumerate(stacked_energies):
            if is_training:
                ffjord = fjord_regularization.FFJORD(
                    state_time_derivative_fn=layer, #Each of the models we build
                    ode_solve_fn=ode_solve_fn_energy,
                    is_training=is_training,
                    trace_augmentation_fn=trace_augmentation_fn,
                    name='energy{}'.format(ilayer), #Bijectors need to be named to receive conditional inputs
                    jacobian_factor = self.config['JAC'], #Regularization strength
                    kinetic_factor = self.config['REG'],
                    b = self.config['b'],
                )
            else:
                
                ffjord = tfb.FFJORD(
                    state_time_derivative_fn=layer, #Each of the  models we build
                    ode_solve_fn=ode_solve_fn_energy,
                    trace_augmentation_fn=trace_augmentation_fn,
                    #trace_augmentation_fn=tfb.ffjord.trace_jacobian_exact,
                    name='energy{}'.format(ilayer),
                )
                
            self.bijectors_energy.append(ffjord)

        #Reverse the bijector order
        self.chain = tfb.Chain(list(reversed(self.bijectors)))
        self.chain_energy = tfb.Chain(list(reversed(self.bijectors_energy)))

        self.loss_tracker = keras.metrics.Mean(name="loss")
        
        #Determine the base distribution
        self.base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=np.zeros(num_output,dtype=np.float32),           
            scale_diag=np.ones(num_output,dtype=np.float32)
        )


        self.base_distribution_energy = tfp.distributions.MultivariateNormalDiag(
            loc=np.zeros(num_energy,dtype=np.float32),
            scale_diag=np.ones(num_energy,dtype=np.float32)
        )
        
        self.flow=tfd.TransformedDistribution(distribution=self.base_distribution, bijector=self.chain)
        self.flow_energy=tfd.TransformedDistribution(distribution=self.base_distribution_energy,
                                                     bijector=self.chain_energy)
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]
    
    def compile(self,e_optimizer, v_optimizer):
        super(FFJORD, self).compile(experimental_run_tf_function=False)
        self.e_optimizer = e_optimizer
        self.v_optimizer = v_optimizer

    # def call(self, inputs, conditional=None):
    #     kwargs = make_bijector_kwargs(self.flow.bijector,{'bijector.': {'conditional':conditional }})
    #     return self.flow.bijector.forward(inputs,**kwargs)
        
    def generate(self,conditional,nsplit=10):
        voxel = []
        energy = []
        
        for cond_split in np.array_split(np.array(conditional), nsplit):
            bijector_kwargs_energy = make_bijector_kwargs(
                self.flow_energy.bijector, {'energy.': {'conditional': cond_split}})
            energies = self.flow_energy.sample(cond_split.shape[0],bijector_kwargs= bijector_kwargs_energy)
            energy.append(energies.numpy())
            bijector_kwargs = make_bijector_kwargs(
                self.flow.bijector, {'bijector.': {'conditional': tf.concat([energies,cond_split],-1)}})
            voxels = self.flow.sample(cond_split.shape[0],bijector_kwargs= bijector_kwargs)
            voxel.append(voxels.numpy())
        voxel = np.concatenate(voxel,0)
        energy = np.concatenate(energy,0)
        return voxel,energy
    
    def log_loss(self,voxel,energies,conditional):
        bijector_kwargs = make_bijector_kwargs(
            self.flow.bijector, {'bijector.': {'conditional': tf.concat([energies,conditional],-1)}})
        
        #loss = loss_energy
        loss = -tf.reduce_mean(self.flow.log_prob(voxel,bijector_kwargs= bijector_kwargs)) 
        
        return loss

    def log_loss_energy(self,energies,conditional):
        bijector_kwargs_energy = make_bijector_kwargs(
            self.flow_energy.bijector, {'energy.': {'conditional': conditional}})
        
        loss_energy = -tf.reduce_mean(self.flow_energy.log_prob(energies,
                                                                bijector_kwargs= bijector_kwargs_energy))
        
        return loss_energy


    @tf.function()
    def train_step(self, inputs):
        data,energy,cond = inputs
        with tf.GradientTape() as tape:
            loss_voxel = self.log_loss(data,energy,cond)

        v = tape.gradient(loss_voxel, self.flow.trainable_variables)
        v = [tf.clip_by_norm(grad, 1) for grad in v]
        self.v_optimizer.apply_gradients(zip(v, self.flow.trainable_variables))
        self.loss_tracker.update_state(loss_voxel)

        with tf.GradientTape() as tape:
            loss_energy = self.log_loss_energy(energy,cond)
        
        e = tape.gradient(loss_energy, self.flow_energy.trainable_variables)
        self.e_optimizer.apply_gradients(zip(e, self.flow_energy.trainable_variables))
        
        return {"loss": self.loss_tracker.result(),'energy':loss_energy}
    
    @tf.function
    def test_step(self, inputs):
        data,energy,cond = inputs        
        loss_voxel=self.log_loss(data,energy,cond)
        loss_energy = self.log_loss_energy(energy,cond)
        loss = loss_voxel+loss_energy
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
        ntest,samples_test =preprocessing.MNIST_prep(X_test, y_use)
        
        test_1D = False #Use convolutional networks for the model backbone
    elif model_name == 'calogan':        
        dataset_config = preprocessing.LoadJson('config_calogan_vit.json')
        file_path=dataset_config['FILE']
        ntrain,ntest,samples_train,samples_test,mask = preprocessing.CaloGAN_prep(file_path,int(flags.nevts),use_logit=True)
        use_1D = True
    elif model_name == 'calochallenge':
        dataset_config = preprocessing.LoadJson('config_challenge_vit.json')
        file_path=dataset_config['FILE']
        ntrain,ntest,samples_train,samples_test,mask = preprocessing.CaloChallenge_prep(file_path,int(flags.nevts),use_logit=True)
        use_1D = False
    else:
        raise ValueError("Model not implemented!")
        
        
    LR = float(dataset_config['LR'])
    NUM_EPOCHS = dataset_config['MAXEPOCH']
    STACKED_FFJORDS = dataset_config['NSTACKED'] #Number of stacked transformations
    
    NUM_LAYERS = dataset_config['NLAYERS'] #Hiddden layers per bijector
    BATCH_SIZE = dataset_config['BATCH']
    NLAYER = dataset_config['NENERGY']

    
    #Stack of bijectors 
    stacked_vit = []
    stacked_dense = [] 
    for istack in range(1):
        vit_model = BACKBONE_ODE(dataset_config['SHAPE'], NLAYER+1,
                                 config=dataset_config,use_1D=use_1D)
        stacked_vit.append(vit_model)

    for istack in range(STACKED_FFJORDS):
        dense_model = BACKBONE_ODE([NLAYER], 1, config=dataset_config,use_1D=True,
                                   name='mlp_energy',use_dense=True)
        stacked_dense.append(dense_model)

    #Create the model
    model = FFJORD(stacked_vit,stacked_dense,
                   np.prod(list(dataset_config['SHAPE'])),NLAYER,
                   mask=mask,
                   config=dataset_config)

    
    opt_v = tfa.optimizers.AdamW(learning_rate=LR,weight_decay=1e-2*LR)
    # opt_v = tf.optimizers.Adam(learning_rate=LR)
    opt_v = hvd.DistributedOptimizer(opt_v)
    
    opt_e = tfa.optimizers.AdamW(learning_rate=0.1*LR,weight_decay=1e-2*LR)
    # opt_e = tf.optimizers.SGD(learning_rate=LR)
    # opt_e = tf.optimizers.Adam(learning_rate=LR)
    opt_e = hvd.DistributedOptimizer(opt_e)

    
    # Horovod: add Horovod DistributedOptimizer.
    model.compile(v_optimizer=opt_v,e_optimizer=opt_e)

    if flags.load:
        load_model(model,checkpoint_dir='../checkpoint_{}'.format(dataset_config['MODEL']))

    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        ReduceLROnPlateau(patience=30, min_lr=1e-7,verbose=hvd.rank() == 0),
        EarlyStopping(patience=dataset_config['EARLYSTOP'],restore_best_weights=True),
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=10, initial_lr=LR, verbose=1),
    ]

    if hvd.rank()==0:
        checkpoint = ModelCheckpoint('../checkpoint_{}/ffjord'.format(dataset_config['MODEL']),save_best_only=True,mode='auto',
                                                               period=1,save_weights_only=True)
        callbacks.append(checkpoint)
        # model(np.zeros((1,dataset_config['SHAPE'][0])),np.zeros((1,1)))
        # print(model.summary())
    
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
