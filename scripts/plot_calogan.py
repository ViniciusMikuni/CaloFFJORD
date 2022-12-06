import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import horovod.tensorflow.keras as hvd
import preprocessing
import utils
from calo_ffjord_split import FFJORD, BACKBONE_ODE, make_bijector_kwargs, load_model
from wgan import WGAN
import time

hvd.init()
utils.SetStyle()
parser = argparse.ArgumentParser()

parser.add_argument('--plot_folder', default='../plots', help='Path to store plot files')
parser.add_argument('--model', default='ffjord', help='Model to load')
parser.add_argument('--nevts', type=float,default=30000, help='Number of events to load')
flags = parser.parse_args()


if flags.model == 'ffjord':
    dataset_config = preprocessing.LoadJson('config_calogan_vit.json')
    use_logit=True
elif flags.model == 'wgan':
    dataset_config = preprocessing.LoadJson('config_calogan_wgan.json')
    use_logit=True
    
file_path=dataset_config['FILE']

energy, gen_energy_layer, gen_energy_voxel,mask = preprocessing.DataLoaderCaloGAN(file_path,-1,use_logit=use_logit,use_noise=False)

if flags.model == 'ffjord':
    STACKED_FFJORDS = dataset_config['NSTACKED'] #Number of stacked transformations    
    NUM_LAYERS = dataset_config['NLAYERS'] #Hiddden layers per bijector
    NLAYER = dataset_config['NENERGY']
    
    stacked_vit = []
    stacked_dense = [] 
    for istack in range(1):
        vit_model = BACKBONE_ODE(dataset_config['SHAPE'], NLAYER+1,
                                 config=dataset_config,use_1D=True)
        stacked_vit.append(vit_model)

    for istack in range(STACKED_FFJORDS):
        dense_model = BACKBONE_ODE([NLAYER], 1, config=dataset_config,
                                   use_1D=True,
                                   name='mlp_energy',use_dense=True)
        stacked_dense.append(dense_model)

        #Create the model
    model = FFJORD(stacked_vit,stacked_dense,
                   np.prod(list(dataset_config['SHAPE'])),NLAYER,
                   is_training=False,
                   config=dataset_config)
    load_model(model,checkpoint_dir='../checkpoint_{}'.format(dataset_config['MODEL']))
    start = time.time()
    flow_energy_voxel,flow_energy_layer = model.generate(energy)
    end = time.time()
    
elif flags.model == 'wgan':
    model = WGAN(dataset_config['SHAPE'], 1, num_noise=dataset_config['NOISE_DIM'],config=dataset_config,use_1D=True)
    load_model(model,checkpoint_dir='../checkpoint_{}'.format(dataset_config['MODEL']))
    
    sample = np.squeeze(model.generate(energy))
    flow_energy_voxel = sample[:,:504]
    flow_energy_layer = sample[:,504:]

    
        
gen_energy, gen_energy_voxel = preprocessing.ReverseNormCaloGAN(energy, gen_energy_layer, gen_energy_voxel,use_logit=use_logit)
gen_energy_voxel[gen_energy_voxel<1e-4]=0

_, flow_energy_voxel = preprocessing.ReverseNormCaloGAN(energy, flow_energy_layer, flow_energy_voxel,use_logit=use_logit)
flow_energy_voxel[flow_energy_voxel<1e-4]=0
flow_energy_voxel=flow_energy_voxel



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

            
    binning = np.geomspace(np.quantile(feed_dict['Geant4'],0.01),1.3*np.quantile(feed_dict['Geant4'],1.0),30)
    fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Deposited energy [GeV]', ylabel= 'Normalized entries',logy=True,binning=binning)
    ax0.set_xscale("log")
    fig.savefig('{}/CaloGAN_TotalE_{}.pdf'.format(flags.plot_folder,dataset_config['MODEL']))
    return feed_dict


def HistElayer(data_dict):
    def _preprocess(data):
        preprocessed = np.reshape(data,(data.shape[0],-1))
        layer1=np.sum(preprocessed[:,:288],-1)
        layer2=np.sum(preprocessed[:,288:432],-1)
        layer3=np.sum(preprocessed[:,432:],-1)
        return layer1,layer2,layer3

    feed_dict = {}
    for ilayer in range(3):
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])[ilayer]


        binning = np.geomspace(1e-2+np.min(feed_dict['Geant4']),1.2*np.max(feed_dict['Geant4']),30)
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Deposited energy [GeV]', ylabel= 'Normalized entries',logy=True,binning=binning)
        ax0.set_xscale("log")
        fig.savefig('{}/CaloGAN_Layer{}E_{}.pdf'.format(flags.plot_folder,ilayer+1,dataset_config['MODEL']))
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
    'Geant4':gen_energy_voxel,
    'FFJORD':flow_energy_voxel,
}

AverageELayer(data_dict)
HistEtot(data_dict)
HistElayer(data_dict)
Classifier(data_dict)
print(end - start)
