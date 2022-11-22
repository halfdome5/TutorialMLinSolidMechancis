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


# %%
'''
pre-processing

'''

def reshape_input(Cs, Ps):
    batch_size = Cs.shape[0]
    # reshape to voigt notation
    cs_tmp = tf.reshape(Cs, [batch_size, 9])
    ps = tf.reshape(Ps, [batch_size, 9])
    # drop dependent values
    cs = tf.concat([cs_tmp[:,:3], cs_tmp[:,4:6], cs_tmp[:,8:]], axis=1)
    return cs, ps

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
    Cs = []
    Ps = []
    Ws = []
    batch_sizes = np.empty(n, dtype='int32')
    # iterate over files
    for i, path in enumerate(paths):
        # read data from path
        F, P, W = read_txt(path)
        # get batch size of that data set
        batch_sizes[i] = tf.shape(W)[0]
        # use right Cauchy-Green tensor for fulfillment of objectivity condition
        C = tf.einsum('ikj,ikl->ijl', F, F)
        # reshape matrices to Voigt notation
        Cs.append(C)
        Ps.append(P)
        Ws.append(W)
        
    # concatenate and convert to tensorflow tensor
    Cs = tf.concat(Cs, axis=0)
    Ps = tf.concat(Ps, axis=0)
    Ws = tf.concat(Ws, axis=0)
        
    return Cs, Ps, Ws, batch_sizes



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

#F, P, W = read_txt('data/test/mixed_test.txt')

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
    I1, J, I4, I5 = compute_invariants(F)
    return compute_potential(I1, J, I4, I5)

def compute_potential(I1, J, I4, I5):
    W = 8 * I1 + 10 * J**2 - 56 * tf.math.log(J) \
        + 0.25 * (I4 ** 2 + I5 ** 2) - 44
    return W

def compute_invariants(F):
    # transversely isotropic structural tensor
    G_ti = np.array([[4, 0, 0],
                  [0, 0.5, 0],
                  [0, 0, 0.5]])
    # transpose F and compute right Cauchy-Green tensor
    C = tf.einsum('ikj,ikl->ijl',F,F)
    # compute invariants
    I1 = tf.linalg.trace(C)
    J = tf.linalg.det(F)
    I4 = tf.linalg.trace(C @ G_ti)
    
    C_inv = tf.linalg.inv(C)
    I3 = tf.linalg.det(C)
    Cof_C = tf.constant(np.array([I3i * C_inv[i,:,:] for i,I3i in enumerate(I3)]))
    I5 = tf.linalg.trace(Cof_C @ G_ti)
    
    return I1, J, I4, I5
