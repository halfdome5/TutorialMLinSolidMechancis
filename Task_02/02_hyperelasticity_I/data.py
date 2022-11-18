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
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# user-defined modules
import models as lm
    

# %%
'''
import data: deformation gradient, stress tensor and strain energy density

'''

def read_txt(path):
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
