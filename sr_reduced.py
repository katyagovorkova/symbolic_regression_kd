import os
import h5py
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import exp, sin, cos

from sklearn.decomposition import PCA
from pysr import PySRRegressor
from matplotlib import pyplot as plt
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import QConv2D, QDense, QActivation
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
    )
from sklearn.model_selection import train_test_split

from plotting import plot_rocs
from losses import reco_loss, mse_loss
from models import conv_ae_reduced


import setGPU

with h5py.File('datasets/background_reduced.h5', 'r') as h5f:
    data = np.array(h5f['dataset'][:500000])
# fit scaler to the full data
data[:,2] = data[:,2]/100
data[:,6] = data[:,6]/100
# define training, test and validation datasets
x_train, x_test = \
    train_test_split(data, test_size=0.2, shuffle=True)
del data

x_train = x_train.reshape((-1,9,1,1))

# latent dim; quant size; pruning
teacher_model = conv_ae_reduced(x_train.shape, 3, 0, False)
# define callbacks
callbacks=[
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
    ]
# train
history = teacher_model.fit(x=x_train, y=x_train,
    epochs=100,
    batch_size=1024,
    verbose=2,
    validation_split=0.2,
    callbacks=callbacks)
teacher_loss = mse_loss(x_train, teacher_model.predict(x_train))
# prepare SR input
y_train = np.log(teacher_loss+1)
x_train = x_train.reshape((-1,9))

y_test = np.log(mse_loss(x_test, teacher_model.predict(x_test.reshape((-1,9,1,1))))+1)


model = PySRRegressor(
    julia_project="/afs/cern.ch/user/e/egovorko/cernbox/SymbolicRegression.jl",
    model_selection="best",  # Result is mix of simplicity+accuracy
    maxsize=50,
    niterations=10,
    binary_operators=["+", "*", "/", "-"],
    unary_operators=[
        "square",
        "cos",
        "exp"
    # ^ Custom operator (julia syntax)
    ],
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
    equation_file='results/model.csv'
)

model.fit(x_train, y_train)

best_model = str(model.sympy(-1))

eval_dict = {
    'cos': np.cos,
    'exp': np.exp,
    'square': np.square
    }
eval_dict_x_test = eval_dict.copy()
for i in range(x_test.shape[-1]):
    eval_dict_x_test[f'x{i}'] = x_test[:,i]

background_loss = eval(
    best_model,
    eval_dict_x_test
    )
test_range = (y_test.min(), y_test.max())
plt.hist(y_test,
    label='Teacher loss',
    bins=100,
    range=test_range,
    density=True,
    alpha=0.5
    )
plt.hist(background_loss,
    label=best_model[0:len(best_model)//3]+'\n'+best_model[len(best_model)//3:len(best_model)//3*2]+'\n'+best_model[len(best_model)//3*2:],
    bins=100,
    range=test_range,
    density=True,
    alpha=0.5
    )
plt.legend(loc='best',fontsize=6)
plt.savefig('background_loss.pdf')
plt.clf()



colors = ['#016c59', '#7a5195', '#ef5675', '#ffa600', '#67a9cf']
BSM_SAMPLES = ['Leptoquark', 'A to 4 leptons', 'hChToTauNu', 'hToTauTau']
input_signal_files = ['datasets/leptoquark_reduced.h5',
    'datasets/ato4l_reduced.h5',
    'datasets/hChToTauNu_reduced.h5',
    'datasets/hToTauTau_reduced.h5']


bsm_losses = []
for bsm_data_name, input_signal_files in zip(BSM_SAMPLES, input_signal_files):
    i = BSM_SAMPLES.index(bsm_data_name)
    # load testing BSM samples
    with h5py.File(input_signal_files, 'r') as f:
        # test model on BSM data
        bsm_data = np.array(f['dataset']).reshape((-1,9))
        bsm_data[:,2] = bsm_data[:,2]/100
        bsm_data[:,6] = bsm_data[:,6]/100
        teacher_loss = np.log(mse_loss(bsm_data, teacher_model.predict(bsm_data.reshape((-1,9,1,1))))+1)
        #
        eval_dict_bsm = eval_dict.copy()
        for i_feat in range(bsm_data.shape[-1]):
            eval_dict_bsm[f'x{i_feat}'] = bsm_data[:,i_feat]
        bsm_loss = eval(
            best_model,
            eval_dict_bsm
            )
        bsm_losses.append([bsm_data_name, bsm_loss])
        plot_rocs(background_loss, bsm_loss, bsm_data_name, colors[i])
        plot_rocs(y_test, teacher_loss, f'Teacher {bsm_data_name}', colors[i], linestyle='-.', alpha=0.5)

plt.legend(loc='best')
plt.savefig('rocs.pdf')
plt.clf()


plt.hist(background_loss,
    label='background',
    density=True,
    alpha=0.5,
    bins=100,
    range=test_range
    )
for bsm_obj in bsm_losses:
    bsm = bsm_obj[0]
    bsm_loss = bsm_obj[1]
    plt.hist(bsm_loss,
        label=bsm,
        density=True,
        alpha=0.5,
        bins=100,
        range=test_range
        )
plt.legend(loc='best')
plt.savefig('bsm_loss.pdf')
plt.clf()

print(f'Best model: \n {best_model}')

