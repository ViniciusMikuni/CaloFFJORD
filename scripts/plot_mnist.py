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

parser.add_argument('--plot_folder', default='../plots', help='Path to store plot files')
parser.add_argument('--nevts', type=float,default=10, help='Number of events to load')
parser.add_argument('--config', default='config_mnist.json', help='Training parameters')
flags = parser.parse_args()

        
dataset_config = preprocessing.LoadJson(flags.config)

STACKED_FFJORDS = dataset_config['NSTACKED'] #Number of stacked transformations    
NUM_LAYERS = dataset_config['NLAYERS'] #Hiddden layers per bijector
use_conv = True

stacked_convs = []
for _ in range(STACKED_FFJORDS):
    conv_model = BACKBONE_ODE(dataset_config['SHAPE'], 1, config=dataset_config,use_conv=use_conv)
    stacked_convs.append(conv_model)

model = FFJORD(stacked_convs,np.prod(dataset_config['SHAPE']),is_training=False,
               config=dataset_config)
load_model(model,checkpoint_dir='../checkpoint_{}'.format(dataset_config['MODEL']))

sample_flow = model.flow.sample(
    int(flags.nevts),
    bijector_kwargs=make_bijector_kwargs(
        model.flow.bijector, {'bijector.': {'conditional': 0.4*np.ones((flags.nevts,1),dtype=np.float32)}})).numpy()

# sample_flow=(sample_flow+1)/2.
transformed=sample_flow.reshape(-1,28,28,1)
#Plotting


alpha = 1e-6
exp = np.exp(transformed)    
x = exp/(1+exp)
transformed = (x-alpha)/(1 - 2*alpha)*255
# transformed=transformed/np.max(transformed.reshape(transformed.shape[0],-1),-1)*255

#transformed=transformed>1

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(transformed[i], cmap='gray', interpolation='none')    
plt.savefig('{}/conditional_mnist.pdf'.format(flags.plot_folder))
