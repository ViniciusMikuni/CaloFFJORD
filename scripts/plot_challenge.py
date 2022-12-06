import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os,time
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import preprocessing
import utils
from sklearn.metrics import roc_curve, auc
from calo_ffjord_split import FFJORD, BACKBONE_ODE, make_bijector_kwargs, load_model
from wgan import WGAN

hvd.init()
utils.SetStyle()
parser = argparse.ArgumentParser()

parser.add_argument('--plot_folder', default='../plots', help='Path to store plot files')
parser.add_argument('--model', default='ffjord', help='Model to load')
parser.add_argument('--nevts', type=float,default=30000, help='Number of events to load')

flags = parser.parse_args()

        
if flags.model == 'ffjord':
    dataset_config = preprocessing.LoadJson('config_challenge_vit.json')
    use_logit=True
elif flags.model == 'wgan':
    dataset_config = preprocessing.LoadJson('config_challenge_wgan.json')
    use_logit=False
else:
    raise ValueError("Model not implemented!")
    
file_path=dataset_config['FILE']
print(file_path)

sim_energy, sim_energy_layer, sim_energy_voxel,mask = preprocessing.DataLoaderChallenge(file_path,-1,use_logit=use_logit,use_noise=False)

if flags.model == 'ffjord':
    STACKED_FFJORDS = dataset_config['NSTACKED'] #Number of stacked transformations    
    NUM_LAYERS = dataset_config['NLAYERS'] #Hiddden layers per bijector
    NLAYER = dataset_config['NENERGY']
    
    stacked_vit = []
    stacked_dense = [] 
    for istack in range(1):
        vit_model = BACKBONE_ODE(dataset_config['SHAPE'], NLAYER+1,
                                 config=dataset_config,use_1D=False)
        stacked_vit.append(vit_model)

    for istack in range(STACKED_FFJORDS):
        dense_model = BACKBONE_ODE([NLAYER], 1, config=dataset_config,use_1D=True,
                                   name='mlp_energy',use_dense=True)
        stacked_dense.append(dense_model)

    #Create the model
    model = FFJORD(stacked_vit,stacked_dense,
                   np.prod(list(dataset_config['SHAPE'])),NLAYER,
                   is_training=False,
                   config=dataset_config)
    load_model(model,checkpoint_dir='../checkpoint_{}'.format(dataset_config['MODEL']))
    start = time.time()
    flow_energy_voxel,flow_energy_layer = model.generate(sim_energy)
    end = time.time()
elif flags.model == 'wgan':
    model = WGAN(dataset_config['SHAPE'], 1, num_noise=dataset_config['NOISE_DIM'],config=dataset_config,use_1D=False)
    load_model(model,checkpoint_dir='../checkpoint_{}'.format(dataset_config['MODEL']))
    
    sample = np.squeeze(model.generate(sim_energy))

        
true_energy, sim_energy_voxel = preprocessing.ReverseNormChallenge(sim_energy,sim_energy_layer,sim_energy_voxel,use_logit=use_logit) #convert it back
#print(np.max(sim_energy_voxel))
sim_energy_voxel[sim_energy_voxel<1e-4]=0

_, flow_energy_voxel = preprocessing.ReverseNormChallenge(sim_energy,flow_energy_layer,flow_energy_voxel,use_logit=use_logit)
flow_energy_voxel[flow_energy_voxel<1e-4]=0
flow_energy_voxel*=mask

def AverageELayer(data_dict):
    
    def _preprocess(data):
        preprocessed = np.reshape(data,(data.shape[0],-1))
        #preprocessed = np.sum(preprocessed,-1,keepdims=True)
        #preprocessed = np.mean(preprocessed,0)
        return preprocessed
        
    feed_dict = {}
    for key in data_dict:
        feed_dict[key] = _preprocess(data_dict[key])

    fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='Layer number', ylabel= 'Mean deposited energy [GeV]')
    
    #ax0.set_yscale("log")
    fig.savefig('{}/EnergyZ_{}.pdf'.format(flags.plot_folder,dataset_config['MODEL']))
    return feed_dict



def HistEtot(data_dict):
    def _preprocess(data):
        preprocessed = np.reshape(data,(data.shape[0],-1))
        return np.sum(preprocessed,-1)

    feed_dict = {}
    for key in data_dict:
        feed_dict[key] = _preprocess(data_dict[key])

            
    binning = np.geomspace(np.quantile(feed_dict['Geant4'],0.01),np.quantile(feed_dict['Geant4'],1.0),10)
    
    fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Deposited energy [GeV]', ylabel= 'Normalized entries',logy=True,binning=binning)
    ax0.set_xscale("log")
    fig.savefig('{}/Challenge_TotalE_{}.pdf'.format(flags.plot_folder,dataset_config['MODEL']))
    return feed_dict


def HistELayer(data_dict):
    def _preprocess(data):
        preprocessed = np.reshape(data,(data.shape[0],9,-1))
        return np.sum(preprocessed,-1)

    

    for i in range(9):
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])[:,i]
            
        binning = np.geomspace(max(np.quantile(feed_dict['Geant4'],0.01),1e-4),np.quantile(feed_dict['Geant4'],1.0),10)
        #binning = np.linspace(-10,10,10)
        #binning = np.geomspace(1e-4,1e2,10)
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Deposited energy [GeV] Layer {}'.format(i), ylabel= 'Normalized entries',logy=True,binning=binning)
        ax0.set_xscale("log")
        fig.savefig('{}/Challenge_TotalE_Layer{}_{}.pdf'.format(flags.plot_folder,i,dataset_config['MODEL']))
    return feed_dict


def Classifier(data_dict):
    from tensorflow import keras
    train = np.concatenate([data_dict['Geant4'],data_dict['FFJORD']],0)
    labels = np.concatenate([np.zeros((data_dict['Geant4'].shape[0],1)),
                             np.ones((data_dict['FFJORD'].shape[0],1))],0)
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1,activation='sigmoid')
    ])
    opt = tf.optimizers.Adam(learning_rate=2e-4)
    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    model.fit(train, labels,batch_size=1000, epochs=30)
    pred = model.predict(train)
    fpr, tpr, _ = roc_curve(labels,pred, pos_label=1)    
    print("AUC: {}".format(auc(fpr, tpr)))


data_dict = {
    'Geant4':sim_energy_voxel,
    'FFJORD':flow_energy_voxel,
}

AverageELayer(data_dict)
HistELayer(data_dict)
HistEtot(data_dict)

Classifier(data_dict)
print(end - start)
