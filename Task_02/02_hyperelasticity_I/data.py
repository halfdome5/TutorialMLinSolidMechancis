#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:57:42 2022

@author: jasper
"""

# %%
'''
import modules

'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# user-defined models
import models as lm

# %%
'''
pre-processing

'''

def reshape(x):
    batch_size = x.shape[0]
    return tf.reshape(x, [batch_size, 9])

def reshape_C(Cs):
    batch_size = Cs.shape[0]
    # reshape to voigt notation
    cs_tmp = tf.reshape(Cs, [batch_size, 9])
    # drop dependent values
    cs = tf.concat([cs_tmp[:,:3], cs_tmp[:,4:6], cs_tmp[:,8:]], axis=1)
    return cs

def get_sample_weights(Ps, batch_sizes):
    norms = tf.norm(Ps, axis=[1,2])
    w = np.zeros(norms.shape)
    for i in range(batch_sizes.shape[0]):
        il = tf.math.reduce_sum(batch_sizes[:i])
        iu = tf.math.reduce_sum(batch_sizes[:i+1])
        
        w[il:iu] = tf.math.reduce_sum(norms[il:iu]) / batch_sizes[i]
        
    return w ** (-1)



# %%    
'''
data import

''' 

def load_stress_strain_data(paths):
    '''
    Calls read_txt function. The objectivity condition is fulfilled by using using six
    indepented components of C instead of nine components of F.

    '''
    # initialize lists
    n = len(paths)
    Fs = []
    Ps = []
    Ws = []
    batch_sizes = np.empty(n, dtype='int32')
    # iterate over files
    for i, path in enumerate(paths):
        F, P, W = read_txt(path)
        
        batch_sizes[i] = tf.shape(W)[0]
        Fs.append(F)
        Ps.append(P)
        Ws.append(W)
        
    # concatenate and convert to tensorflow tensor
    Fs = tf.concat(Fs, axis=0)
    Ps = tf.concat(Ps, axis=0)
    Ws = tf.concat(Ws, axis=0)
        
    return Fs, Ws, Ps, batch_sizes



def read_txt(path):
    '''
    Reads deformation gradient F, Piola-Kirchhoff stress tensor and the strain
    energy density from a text file in path.

    '''
    arr = np.loadtxt(path)
    
    nrows = np.size(arr, axis=0)

    F = arr[:, :9].reshape(nrows, 3, 3)
    P = arr[:, 9:18].reshape(nrows, 3, 3)
    W = arr[:, 18:]
    
    return F, P, W


# %%
'''
custom non-trainable layer

'''

class P(layers.Layer):
    '''
    computed gradient of potential
    '''
    def __init__(self, W):
        super().__init__()
        self.W = W
        
    def call(self, F):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(F)
            w = self.W(F)
        p = tape.gradient(w, F)
        return p

# additional functions called in layer
def W(F):
    I = compute_invariants(F)
    return compute_potential(I[:,0], I[:,1], I[:,3], I[:,4])

def compute_potential(I1, J, I4, I5):
    W = 8 * I1 + 10 * J**2 - 56 * tf.math.log(J) \
        + 0.2 * (I4 ** 2 + I5 ** 2) - 44
    return W


def compute_invariants(F):
    C = lm.RightCauchyGreenLayer()(F)
    I = lm.InvariantLayer()(F, C)
    return I
