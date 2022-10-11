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

from plotting import plot_rocs
from losses import reco_loss, mse_loss
from models import conv_ae

TRAIN = True
RETRAIN_TEACHER = True
DO_PCA = True


input_train_file = 'l1_ae_train_loss.h5'
input_test_file = 'l1_ae_test_loss.h5'
input_signal_file = 'l1_ae_signal_loss.h5'
data_name = 'data'
n_features = 3
teacher_loss_name = 'teacher_loss'
n_events = 10000


if RETRAIN_TEACHER:
    # load pt scaler:
    with open('/eos/cms/store/cmst3/user/egovorko/kd_output/data_-1.pickle', 'rb') as f:
        _, _, _, _, _, _, _, pt_scaler, \
        _, _, _, _ = pickle.load(f)

if TRAIN==True:

    # load teacher's loss for training
    with h5py.File(input_train_file, 'r') as f:
            x_train = np.array(f[data_name][:n_events,:,:n_features]).reshape((-1,57))
            y_train = np.array(f[teacher_loss_name][:n_events])

    if DO_PCA:
        pca = PCA(n_components=6)
        x_train_original = x_train.reshape((-1,19,3))
        x_train = pca.fit_transform(x_train)

    if RETRAIN_TEACHER:

        x_train = x_train.reshape((-1,6,1,1))

        # latent dim; quant size; pruning
        teacher_model = conv_ae(x_train.shape, 3, 0, False)
        # define callbacks
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
            ]
        x_train_original[:,:,0] = pt_scaler.transform(x_train_original[:,:,0])
        y_train_scaled = (pca.transform(x_train_original.reshape((-1,57)))).reshape((-1,6,1,1))
        print(f'Input shape: {y_train_scaled.shape}')
        # train
        history = teacher_model.fit(x=x_train, y=y_train_scaled,
            epochs=100,
            batch_size=1024,
            verbose=2,
            validation_split=0.2,
            callbacks=callbacks)
        y_teacher_train = mse_loss(y_train_scaled, teacher_model.predict(x_train))
        # prepare SR input
        y_train = np.log(y_teacher_train+1)
        x_train = x_train.reshape((-1,6))

    model = PySRRegressor(
        julia_project="/afs/cern.ch/user/e/egovorko/cernbox/SymbolicRegression.jl",
        model_selection="best",  # Result is mix of simplicity+accuracy
        maxsize=50,
        niterations=100,
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

    # load teacher's loss for training
    with h5py.File(input_test_file, 'r') as f:
            x_test = np.array(f[data_name][:,:,:n_features]).reshape((-1,57))
            y_test = np.array(f[teacher_loss_name])
    if DO_PCA:
        x_test_original = x_test.reshape((-1,19,3))
        x_test = pca.transform(x_test)
    if RETRAIN_TEACHER:
        x_test_original[:,:,0] = pt_scaler.transform(x_test_original[:,:,0])
        y_test_scaled = (pca.transform(x_test_original.reshape((-1,57)))).reshape((-1,6,1,1))
        y_test = np.log(mse_loss(y_test_scaled, teacher_model.predict(x_test.reshape((-1,6,1,1))))+1)

    with h5py.File('/afs/cern.ch/work/e/egovorko/public/test_dataset_pca.h5', 'w') as f:
        f.create_dataset('x_test_pca', data=x_test)
        f.create_dataset('y_test', data=y_test)

    best_model = str(model.sympy(-1))
    # save the model separately
    with open('results/model.json', 'w') as f:
        json.dump(best_model, f)

else:
    # read PCA'ed x_test
    with h5py.File('/afs/cern.ch/work/e/egovorko/public/test_dataset_pca.h5', 'r') as f:
        x_test = np.array(f['x_test_pca'])
        y_test = np.array(f['y_test'])
    # read save the model
    with open('results/model.json') as f:
        best_model = json.load(f)

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
test_range = (y_test.min()-2, y_test.max()-5)
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


BSM_SAMPLES = ['Leptoquark', 'A to 4 leptons', 'hChToTauNu', 'hToTauTau']
colors = ['#016c59', '#7a5195', '#ef5675', '#ffa600', '#67a9cf']


# load testing BSM samples
with h5py.File(input_signal_file, 'r') as f:
    # test model on BSM data
    bsm_losses = []
    for i, bsm_data_name in enumerate(BSM_SAMPLES):
        bsm_data = np.array(f[f'bsm_data_{bsm_data_name}'][:,:,:n_features].reshape((-1,57)))
        teacher_loss = np.array(f[f'{teacher_loss_name}_{bsm_data_name}'])
        #
        if DO_PCA:
            bsm_data_original = bsm_data.reshape((-1,19,3))
            bsm_data = pca.transform(bsm_data)

        if RETRAIN_TEACHER:
            bsm_data_original[:,:,0] = pt_scaler.transform(bsm_data_original[:,:,0])
            bsm_data_scaled = (pca.transform(bsm_data_original.reshape((-1,57)))).reshape((-1,6,1,1))
            teacher_loss = np.log(mse_loss(bsm_data_scaled, teacher_model.predict(bsm_data.reshape((-1,6,1,1))))+1)
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

