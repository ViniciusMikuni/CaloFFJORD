import json, yaml
import os
import h5py as h5
import numpy as np
import tensorflow as tf


layer_norm_challenge = {
    'mean': [-0.667027473449707, -1.8747042417526245, -0.25153979659080505, 0.37705254554748535, 0.6952704191207886, 0.6853489875793457, -0.025409294292330742, -0.9231266975402832, -1.3478056192398071],
    'std': [0.2340407818555832, 1.0933985710144043, 0.8074700832366943, 0.8865748047828674, 2.1300981044769287, 4.074546813964844, 5.794466495513916, 7.009632110595703, 8.237323760986328]    

}



def split_data(data,nevts,frac=0.8):
    data = data.shuffle(nevts)
    train_data = data.take(int(frac*nevts)).repeat()
    test_data = data.skip(int(frac*nevts)).repeat()
    return train_data,test_data

def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

def SaveJson(save_file,data):
    with open(save_file,'w') as f:
        json.dump(data, f)

def ApplyPreprocessing(shower,fname='preprocessing_challenge.json',mask=None):
    params = LoadJson(fname)
    transform = (np.ma.divide((shower-params['mean']),params['std']).filled(0)).astype(np.float32)
    if mask is not None:
        return  transform*mask
    else:
        return  transform

def ReversePreprocessing(shower,fname='preprocessing_challenge.json',mask=None):
    params = LoadJson(fname)
    transform = (params['std']*shower+params['mean']).astype(np.float32)
    if mask is not None:
        return  transform*mask
    else:
        return  transform

def CalcPreprocessing(shower,fname):
    mask = shower!=0
    # print(np.sum(mask,0))
    # input()
    mean = np.average(shower,axis=0)
    std = np.std(shower,axis=0)
    data_dict = {
        'mean':mean.tolist(),
        #'std':np.sqrt(np.average((shower - mean)**2,axis=0,weights=mask)).tolist()
        'std':np.std(shower,0).tolist()
    }
    for key in data_dict:
        print(key,data_dict[key])
    
    # SaveJson(fname,data_dict)
    print("done!")

def ReversePrepLayerChallenge(e_layer):
    norm = layer_norm_challenge
    return  (norm['std']*e_layer+norm['mean']).astype(np.float32)


def ApplyPrepLayerChallenge(e_layer):
    norm = layer_norm_challenge
    return  ((e_layer-norm['mean'])/norm['std']).astype(np.float32)


def ReverseNormChallenge(e,e_layer,voxels,max_deposit=2.3,use_logit=True,
                         preprocessing_file='preprocessing_challenge.json'):
    
    '''Revert the transformations applied to the training set'''
    #shape=voxels.shape
    alpha = 1e-6
    energy = 10**(3*e)
    #return energy,voxels

    layer_norm = e_layer
    if use_logit:
        e_layer =ReversePrepLayerChallenge(e_layer)
        exp = np.exp(e_layer)    
        x = exp/(1+exp)
        e_layer = (x-alpha)/(1 - 2*alpha)


    layer_norm= np.zeros(e_layer.shape,dtype=np.float32)
    
    for i in range(e_layer.shape[1]):
        layer_norm[:,i] = np.squeeze(max_deposit*energy)*e_layer[:,i]

    
    layer_norm[:,0] = np.squeeze(max_deposit*energy)*e_layer[:,0]*e_layer[:,1]
    for i in range(1,e_layer.shape[1]-1):
        layer_norm[:,i] = e_layer[:,i+1]*(np.squeeze(max_deposit*energy)*e_layer[:,0] - np.sum(layer_norm[:,:i],-1))
    layer_norm[:,-1] = np.squeeze(max_deposit*energy)*e_layer[:,0] - np.sum(layer_norm[:,:-1],-1)

    
    if use_logit:
        voxels = ReversePreprocessing(voxels,preprocessing_file)    
        exp = np.exp(voxels)    
        x = exp/(1+exp)
        voxels = (x-alpha)/(1 - 2*alpha)

    #Normalize each calorimeter layer before multiplying by the estimated energy
    voxels = voxels.reshape(voxels.shape[0],layer_norm.shape[1],-1)
    voxels = voxels/np.sum(voxels,-1,keepdims=True)
    voxels = voxels*np.expand_dims(layer_norm,-1)
    voxels = voxels.reshape(voxels.shape[0],-1)
    return energy,voxels


def ReverseNormCaloGAN(e,e_layer,e_voxel,use_logit=True,
                       preprocessing_file='preprocessing_calogan.json'):
    '''Revert the transformations applied to the training set'''    
    alpha = 1e-6
    gen_energy = 10**(e+1)
    

    if use_logit:
        exp = np.exp(e_layer)    
        x = exp/(1+exp)
        e_layer = (x-alpha)/(1 - 2*alpha)
    
    layer_norm= np.zeros(e_layer.shape,dtype=np.float32)
        
    layer_norm[:,0] = np.squeeze(gen_energy)*e_layer[:,0]*e_layer[:,1]
    layer_norm[:,1] = np.squeeze(gen_energy)*e_layer[:,0]*e_layer[:,2]*(1-e_layer[:,1])
    layer_norm[:,2] = np.squeeze(gen_energy)*e_layer[:,0]*(1-e_layer[:,1])*(1-e_layer[:,2])
    
    
    voxel = ReversePreprocessing(e_voxel,preprocessing_file)
    if use_logit:        
        exp = np.exp(voxel)    
        x = exp/(1+exp)    
        voxel = (x-alpha)/(1 - 2*alpha)

    voxel[:,:288] = voxel[:,:288] * np.ma.divide(np.expand_dims(layer_norm[:,0],-1),np.sum(voxel[:,:288],-1,keepdims=True)).filled(0)
    voxel[:,288:432] = voxel[:,288:432] * np.ma.divide(np.expand_dims(layer_norm[:,1],-1),np.sum(voxel[:,288:432],-1,keepdims=True)).filled(0)
    voxel[:,432:] = voxel[:,432:] * np.ma.divide(np.expand_dims(layer_norm[:,2],-1),np.sum(voxel[:,432:],-1,keepdims=True)).filled(0)
    return gen_energy,voxel*100

def DataLoaderCaloGAN(file_name,nevts=-1,use_logit=True,use_noise=True,
                      preprocessing_file='preprocessing_calogan.json'):
    '''
    Inputs:
    - name of the file to load
    - number of events to use
    Outputs:
    - Generated particle energy (value to condition the flow) (nevts,1)
    - Energy deposition in each layer (nevts,3)
    - Normalized energy deposition per voxel (nevts,504)
    '''
    import horovod.tensorflow.keras as hvd
    hvd.init()
    
    with h5.File(file_name,"r") as h5f:
        if nevts <0:
            nevts = len(h5f['energy'])
        e = h5f['energy'][hvd.rank():nevts:hvd.size()].astype(np.float32)
        layer0= h5f['layer_0'][hvd.rank():nevts:hvd.size()].astype(np.float32)/100000.0
        layer1= h5f['layer_1'][hvd.rank():nevts:hvd.size()].astype(np.float32)/100000.0
        layer2= h5f['layer_2'][hvd.rank():nevts:hvd.size()].astype(np.float32)/100000.0

    def preprocessing(data):
        ''' 
        Inputs: Energy depositions in a layer
        Outputs: Total energy of the layer and normalized energy deposition
        '''
        x = data.shape[1]
        y = data.shape[2]
        data_flat = np.reshape(data,[-1,x*y])
        #add noise like caloflow does
        if use_noise:data_flat +=np.random.uniform(0,1e-7,size=data_flat.shape)        

        energy_layer = np.sum(data_flat,-1).reshape(-1,1)

        #Some particle showers have no energy deposition at the last layer
        mask = np.sum(data_flat,0)!=0
        data_flat = np.ma.divide(data_flat,energy_layer).filled(0)        

        if use_logit:
            #Log transform from caloflow paper
            alpha = 1e-6
            x = alpha + (1 - 2*alpha)*data_flat
            data_flat = np.ma.log(x/(1-x)).filled(0)
            
        return energy_layer,data_flat, mask


    flat_energy , flat_shower,mask = preprocessing(np.nan_to_num(layer0))    
    for il, layer in enumerate([layer1,layer2]):
        energy ,shower,mask_layer = preprocessing(np.nan_to_num(layer))
        flat_energy = np.concatenate((flat_energy,energy),-1)
        flat_shower = np.concatenate((flat_shower,shower),-1)
        mask = np.concatenate((mask,mask_layer),-1)

    # CalcPreprocessing(flat_shower,preprocessing_file)
    # input()
    

    def convert_energies(e,layer_energies):
        converted = np.zeros(layer_energies.shape,dtype=np.float32)
        #CaloFlow FLow 1
        converted[:,0] = np.sum(layer_energies,-1)/np.squeeze(e)
        converted[:,1] = layer_energies[:,0]/np.sum(layer_energies,-1)
        converted[:,2] = layer_energies[:,1]/np.sum(layer_energies[:,1:],-1)

        # converted = layer_energies.copy()/e
        
        alpha = 1e-6
        # print(np.min(converted,axis=0),np.max(converted,axis=0))
        # input()
        if use_logit:
            x = alpha + (1 - 2*alpha)*converted
            converted = np.ma.log(x/(1-x)).filled(0)
        return converted
    
    
    flat_energy = convert_energies(e,flat_energy)
    # CalcPreprocessing(flat_energy,preprocessing_file)
    # input()

    flat_shower = ApplyPreprocessing(flat_shower,preprocessing_file)
    #flat_energy = ApplyPrepLayerCalogan(flat_energy)
    # CalcPreprocessing(flat_energy,preprocessing_file)
    # input()
    return np.log10(e/10.),flat_energy,flat_shower, mask


def CaloGAN_prep(file_path,nevts=-1,use_logit=True):    
    energy, energy_layer, energy_voxel,mask = DataLoaderCaloGAN(file_path,nevts,use_logit=use_logit)
    
    tf_data = tf.data.Dataset.from_tensor_slices(energy_voxel)
    tf_energy = tf.data.Dataset.from_tensor_slices(energy_layer)
    tf_cond = tf.data.Dataset.from_tensor_slices(energy)
    samples =tf.data.Dataset.zip((tf_data, tf_energy,tf_cond))
    nevts=energy.shape[0]
    frac = 0.8 #Fraction of events used for training
    train,test = split_data(samples,nevts,frac)
    return int(frac*nevts), int((1-frac)*nevts), train,test,mask



def DataLoaderChallenge(file_name,nevts,max_deposit=2.3,use_logit=True,
                        use_noise=True,
                        preprocessing_file='preprocessing_challenge.json'):
    
    import horovod.tensorflow.keras as hvd
    hvd.init()
    
    rank = hvd.rank()
    size = hvd.size()

    with h5.File(file_name,"r") as h5f:
        e = h5f['incident_energies'][rank:int(nevts):size].astype(np.float32)/1000.0
        shower = h5f['showers'][rank:int(nevts):size].astype(np.float32)/1000.0

    alpha = 1e-6

    mask = np.sum(shower.reshape([shower.shape[0],-1]),0)!=0

    energy_layer = np.sum(shower,(2,3))

    if use_noise:energy_layer +=np.random.uniform(0,1e-8,size=energy_layer.shape)
    shower = np.ma.divide(shower,np.expand_dims(energy_layer,(2,3))).filled(0)
    if use_noise:shower += np.random.uniform(0,1e-7,size=shower.shape)        

    # print(np.min(shower[shower>0]))
    # input()
    shower = shower.reshape(shower.shape[0],-1)
    
    # shower = shower/(max_deposit*e) #Ensure sum of energies is below generated

    if use_logit:
        x = alpha + (1 - 2*alpha)*shower
        shower = np.ma.log(x/(1-x)).filled(0)
        # CalcPreprocessing(shower,preprocessing_file)
        # input()
    
        shower = ApplyPreprocessing(shower,preprocessing_file)


    def convert_energies(e,layer_energies):
        converted = np.zeros(layer_energies.shape,dtype=np.float32)
        #CaloFlow FLow 1
        # for i in range(layer_energies.shape[1]):
        #     converted[:,i] = np.ma.divide(layer_energies[:,i-1],np.squeeze(max_deposit*e)).filled(0)
        
        converted[:,0] = np.ma.divide(np.sum(layer_energies,-1),np.squeeze(max_deposit*e)).filled(0)
        for i in range(1,layer_energies.shape[1]):
            converted[:,i] = np.ma.divide(layer_energies[:,i-1],np.sum(layer_energies[:,i-1:],-1)).filled(0)
            
        # converted = layer_energies/(max_deposit*e)
        alpha = 1e-6
        if use_logit:
            x = alpha + (1 - 2*alpha)*converted
            converted = np.ma.log(x/(1-x)).filled(0)
            converted =ApplyPrepLayerChallenge(converted)
        return converted
    
    
    energy_layer = convert_energies(e,energy_layer)

    # CalcPreprocessing(energy_layer,preprocessing_file)
    # input()

    
    return np.log10(e)/3., energy_layer, shower.reshape([shower.shape[0],-1]),mask
    


def CaloChallenge_prep(file_path,nevts=-1,use_logit=True,max_deposit=2.3):    
    energy, energy_layer, energy_voxel,mask_voxel = DataLoaderChallenge(file_path,nevts,max_deposit=max_deposit,use_logit=use_logit)
    tf_cond = tf.data.Dataset.from_tensor_slices(energy)
    tf_energy = tf.data.Dataset.from_tensor_slices(energy_layer)
    tf_data = tf.data.Dataset.from_tensor_slices(energy_voxel)
    
    samples =tf.data.Dataset.zip((tf_data,tf_energy, tf_cond))
    nevts=energy.shape[0]
    frac = 0.8 #Fraction of events used for training
    train,test = split_data(samples,nevts,frac)
    return int(frac*nevts), int((1-frac)*nevts), train,test, mask_voxel



def MNIST_prep(X,y,nevts=-1):
    '''
    Splits the MNIST dataset into equally sized chunks based on 
    the number of simultaneous processes to run
        
    Inputs: Data and label used for conditioning
    Outputs: Merged dataset used for training and total number of 
    training events in the split sample
        '''
    import horovod.tensorflow.keras as hvd
    hvd.init()
    if nevts<0:
        nevts = y.shape[0]

    y = y.reshape((-1,1))[hvd.rank():nevts:hvd.size()] #Split dataset equally between GPUs
    y = y.astype('float32')/10.0
    
    X = X.reshape(-1,28,28,1)[hvd.rank():nevts:hvd.size()]
    X = X.astype('float32')
    X = X.reshape(-1,784)
    X+=np.random.uniform(0,1,X.shape)
    X /= 256.0

    alpha = 1e-6
    x = alpha + (1 - 2*alpha)*X
    X = np.ma.log(x/(1-x)).filled(0)
    
    # X= 2*X-1
    
    tf_data = tf.data.Dataset.from_tensor_slices(X)
    tf_cond = tf.data.Dataset.from_tensor_slices(y)
    samples =tf.data.Dataset.zip((tf_data, tf_cond)).shuffle(X.shape[0]).repeat()
    return y.shape[0],samples




if __name__ == "__main__":
    file_path = '/pscratch/sd/v/vmikuni/SGM/gamma.hdf5'
    energy, energy_layer, energy_voxel = DataLoader(file_path,1000)
    print(energy.shape, energy_layer.shape, energy_voxel.shape)




    
