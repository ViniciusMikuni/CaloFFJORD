import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import preprocessing
import utils
from calo_ffjord import FFJORD, BACKBONE_ODE, make_bijector_kwargs, load_model

hvd.init()
utils.SetStyle()
parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/SCRATCH/FCC/dataset_2_2_small.hdf5', help='Path to calorimeter dataset used during training')
parser.add_argument('--plot_folder', default='../plots', help='Path to store plot files')
parser.add_argument('--nevts', type=float,default=3000, help='Number of events to load')
parser.add_argument('--config', default='config_dataset2.json', help='Training parameters')
flags = parser.parse_args()

        
dataset_config = preprocessing.LoadJson(flags.config)
file_path=flags.data_folder



sim_energy, sim_energy_voxel = preprocessing.DataLoaderDataset2(file_path,int(flags.nevts))

STACKED_FFJORDS = dataset_config['NSTACKED'] #Number of stacked transformations    
NUM_LAYERS = dataset_config['NLAYERS'] #Hiddden layers per bijector
use_conv = True

stacked_convs = []
for istack in range(STACKED_FFJORDS):
    conv_model = BACKBONE_ODE(dataset_config['SHAPE'], 1, config=dataset_config,use_conv=use_conv,num_stack=istack+1)
    stacked_convs.append(conv_model)

#Create the model
model = FFJORD(stacked_convs,int(np.prod(dataset_config['SHAPE'])),config=dataset_config,is_training=False)
load_model(model,checkpoint_dir='../checkpoint_{}'.format(dataset_config['MODEL']))

sample_flow = model.flow.sample(
    int(flags.nevts),
    bijector_kwargs=make_bijector_kwargs(
        model.flow.bijector, {'bijector.': {'conditional': sim_energy[:int(flags.nevts)]}})
).numpy()

true_energy, sim_energy_voxel = preprocessing.ReverseNormChallenge(sim_energy,sim_energy_voxel) #convert it back
sim_energy_voxel[sim_energy_voxel<1e-4]=0

_, gen_energy_voxel = preprocessing.ReverseNormChallenge(sim_energy, sample_flow)
gen_energy_voxel[gen_energy_voxel<1e-4]=0


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


def Classifier(data_dict):
    from tensorflow import keras
    train = np.concatenate([data_dict['Geant4'],data_dict['FFJORD']],0)
    labels = np.concatenate([np.zeros((data_dict['Geant4'].shape[0],1)),
                             np.ones((data_dict['FFJORD'].shape[0],1))],0)
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1,activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    model.fit(train, labels, epochs=10)


data_dict = {
    'Geant4':sim_energy_voxel,
    'FFJORD':gen_energy_voxel,
}

HistEtot(data_dict)
Classifier(data_dict)
