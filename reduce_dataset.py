import numpy as np
import pandas as pd
import h5py
import setGPU

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Activation, Concatenate, Dropout, Layer
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras import backend as K
import math
import pickle
from datetime import datetime
from tensorboard import program
import os
import tensorflow_model_optimization as tfmot
from qkeras import QDense, QActivation, QBatchNormalization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib

EVENT_IDS = {
    'met': 6,
    'e': 5,
    'g': 4,
    'm': 3,
    'j': 2,
    'b': 1,
    '0': 0,
}

def reduce_dataset(input_file, output_file, size=None):

    with h5py.File(input_file, 'r') as h5f:
        dataset = np.array(h5f['Particles'])
        if size is not None:
            dataset = dataset[:size]

    print(f'Input shape: {dataset.shape}')
    # three types of particles for each the maximum pT and number in the event
    # PID MET N_leptons pt_leptons eta_leptons phi_leptons N_jets pt_jets eta_jets phi_jets
    reduced_dataset = np.zeros((dataset.shape[0], 9))
    # MET pt
    reduced_dataset[:,0] = dataset[:,0,0]

    n_electrons = np.count_nonzero(dataset==2,axis=1)[:,-1]
    n_muons = np.count_nonzero(dataset==3,axis=1)[:,-1]
    reduced_dataset[:,1] = np.sum((n_electrons, n_muons), axis=0)

    pt_electrons = np.amax(np.where(dataset[:,:,-1]==2, dataset[:,:,0],0), axis=1)
    pt_muons = np.amax(np.where(dataset[:,:,-1]==3, dataset[:,:,0],0), axis=1)
    reduced_dataset[:,2] = np.amax((pt_electrons, pt_muons), axis=0)

    # select leptons and order with highest particle pT first
    d_leptons = np.where(np.logical_or(dataset[:,:,[-1]]==2,dataset[:,:,[-1]]==3), dataset, 0)
    ind = np.argsort(-1 * d_leptons[:,:,0])
    ind = np.stack(d_leptons.shape[2]*[ind], axis=1)
    d_leptons = np.take_along_axis(d_leptons.transpose(0, 2, 1), ind, axis=2).transpose(0, 2, 1)
    # eta_leptons
    reduced_dataset[:,3] = d_leptons[:,0,1]
    # phi_leptons
    reduced_dataset[:,4] = d_leptons[:,0,2]

    n_jets = np.count_nonzero(dataset==4,axis=1)[:,-1]
    reduced_dataset[:,5] = n_jets

    pt_jets = np.amax(np.where(dataset[:,:,-1]==4, dataset[:,:,0], 0), axis=1)
    reduced_dataset[:,6] = pt_jets

    # select jets and order with highest particle pT first
    d_jets = np.where(np.logical_or(dataset[:,:,[-1]]==4,dataset[:,:,[-1]]==4), dataset, 0)
    ind = np.argsort(-1 * d_jets[:,:,0])
    ind = np.stack(d_jets.shape[2]*[ind], axis=1)
    d_jets = np.take_along_axis(d_jets.transpose(0, 2, 1), ind, axis=2).transpose(0, 2, 1)
    # eta_leptons
    reduced_dataset[:,7] = d_jets[:,0,1]
    # phi_leptons
    reduced_dataset[:,8] = d_jets[:,0,2]

    # Change the path below for a specific name for the hf5 file
    out_file = h5py.File(output_file, 'w')
    out_file.create_dataset('dataset', data=reduced_dataset, compression='gzip')
    out_file.close()


if __name__ == '__main__':
    reduce_dataset('/eos/project/d/dshep/TOPCLASS/L1jetLepData/cocktail_0.66invfb_fixed.h5', 'datasets/background_reduced.h5', size=5000000)
    reduce_dataset('/eos/project/d/dshep/TOPCLASS/L1jetLepData/leptoquark_LOWMASS_lepFilter_13TeV/leptoquark_LOWMASS_lepFilter_13TeV.h5', 'datasets/leptoquark_reduced.h5')
    reduce_dataset('/eos/project/d/dshep/TOPCLASS/L1jetLepData/Ato4l_lepFilter_13TeV/Ato4l_lepFilter_13TeV.h5', 'datasets/ato4l_reduced.h5')
    reduce_dataset('/eos/project/d/dshep/L1anomaly_DELPHES/hChToTauNu_13TeV_PU20.h5', 'datasets/hChToTauNu_reduced.h5')
    reduce_dataset('/eos/project/d/dshep/L1anomaly_DELPHES/hToTauTau_13TeV_PU20.h5', 'datasets/hToTauTau_reduced.h5')

