import math
import numpy as np
import tensorflow as tf


idx_met_0,idx_met_1=0,1
idx_eg_0,idx_eg_1=1,5
idx_mu_0,idx_mu_1=5,9
idx_jet_0,idx_jet_1=9,19


def tf_mse_loss(inputs, outputs):
    return tf.math.reduce_mean(tf.math.square(outputs-inputs), axis=-1)

def make_mse(inputs, outputs):

    loss = tf_mse_loss(tf.reshape(inputs, [-1, 9]), tf.reshape(outputs, [-1, 9]))
    loss = tf.math.reduce_mean(loss, axis=0) # average over batch
    return loss

def mse_loss(inputs, outputs):
    inputs = inputs.reshape(inputs.shape[0],-1)
    outputs = outputs.reshape(outputs.shape[0],-1)
    return np.mean(np.square(inputs-outputs), axis=-1)

def reco_loss(inputs, outputs, dense=False):

    if dense:
        outputs = outputs.reshape(outputs.shape[0],19,3,1)
        inputs = inputs.reshape(inputs.shape[0],19,3,1)

    # trick on phi
    outputs_phi = math.pi*np.tanh(outputs)
    # trick on eta
    outputs_eta_egamma = 3.0*np.tanh(outputs)
    outputs_eta_muons = 2.1*np.tanh(outputs)
    outputs_eta_jets = 4.0*np.tanh(outputs)
    outputs_eta = np.concatenate([outputs[:,idx_met_0:idx_met_1,:,:], outputs_eta_egamma[:,idx_eg_0:idx_eg_1,:,:], outputs_eta_muons[:,idx_mu_0:idx_mu_1,:,:], outputs_eta_jets[:,idx_jet_0:idx_jet_1,:,:]], axis=1)
    outputs = np.concatenate([outputs[:,:,0,:], outputs_eta[:,:,1,:], outputs_phi[:,:,2,:]], axis=2)
    # change input shape
    inputs = np.squeeze(inputs, -1)
    # # calculate and apply mask
    mask = np.not_equal(inputs, 0)
    outputs = np.multiply(outputs, mask)

    reco_loss = mse_loss(inputs.reshape(inputs.shape[0],57), outputs.reshape(outputs.shape[0],57))
    return reco_loss
