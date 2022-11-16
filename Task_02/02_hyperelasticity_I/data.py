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

F, P, W = read_txt('data/test/mixed_test.txt')

#%%
'''
compute invariants

'''

def compute_invariants(F, G):
    #G = tf.convert_to_tensor(G)
    # transpose F and compute right Cauchy-Green tensor
    C = tf.einsum('ikj,ikl->ijl',F,F)
    # compute invariants
    I1 = tf.linalg.trace(C)
    I3 = tf.linalg.det(C)
    J = tf.linalg.det(F)
    I4 = tf.linalg.trace(C @ G)
    C_inv = tf.linalg.inv(C)
    Cof_C = np.array([i3 * C_inv[i,:,:] for i, i3 in enumerate(I3)])
    I5 = tf.linalg.trace(Cof_C @ G)
    
    return I1, J, I4, I5

G = np.array([[4, 0, 0],
              [0, 0.5, 0],
              [0, 0, 0.5]])
I, J, I4, I5 = compute_invariants(F, G)
    

#%%
'''
compare invariants

'''

def compare_invariants(x1, x2, str):
    mse = tf.keras.metrics.mean_squared_error(x1, x2)
    mae = tf.keras.metrics.mean_absolute_error(x1, x2)
    print('''{} check complete:
          MSE = {}, MAE = {}\n'''.format(str, mse, mae))    
    #print(x1 - x2)
    
# %%
'''
compute analytical potential from invariants

'''

def compute_analytical_potential(I1, J, I4, I5):
    W = 8 * I1 + 10 * J**2 - 56 * tf.math.log(J) \
        + 0.25 * (I4 ** 2 + I5 ** 2) - 44
    return W


# %%
'''
compute Piola-Kirchhoff stress tensor

'''


# %% 
'''
plot over load steps

'''

def plot():
    x = np.arange(len(W)) + 1
    
    # deformation gradient
    fig = plt.figure(1, dpi=600)
    plt.plot(x, F[:, 0, 0], label=r'$F_{11}$')
    plt.plot(x, F[:, 0, 1], label=r'$F_{12}$')
    plt.plot(x, F[:, 0, 2], label=r'$F_{13}$')
    plt.plot(x, F[:, 1, 0], label=r'$F_{21}$')
    plt.plot(x, F[:, 1, 1], label=r'$F_{22}$')
    plt.plot(x, F[:, 1, 2], label=r'$F_{23}$')
    plt.plot(x, F[:, 2, 0], label=r'$F_{31}$')
    plt.plot(x, F[:, 2, 1], label=r'$F_{32}$')
    plt.plot(x, F[:, 2, 2], label=r'$F_{33}$')
    plt.title('Mixed Test')
    plt.xlabel('load step')
    plt.ylabel('deformation gradient')
    plt.grid()
    plt.legend()
    plt.show()
    
    # stress tensor
    fig = plt.figure(2, dpi=600)
    plt.plot(x, P[:, 0, 0], label=r'$P_{11}$')
    plt.plot(x, P[:, 0, 1], label=r'$P_{12}$')
    plt.plot(x, P[:, 0, 2], label=r'$P_{13}$')
    plt.plot(x, P[:, 1, 0], label=r'$P_{21}$')
    plt.plot(x, P[:, 1, 1], label=r'$P_{22}$')
    plt.plot(x, P[:, 1, 2], label=r'$P_{23}$')
    plt.plot(x, P[:, 2, 0], label=r'$P_{31}$')
    plt.plot(x, P[:, 2, 1], label=r'$P_{32}$')
    plt.plot(x, P[:, 2, 2], label=r'$P_{33}$')
    plt.title('Mixed Test')
    plt.xlabel('load step')
    plt.ylabel('stress tensor')
    plt.grid()
    plt.legend()
    plt.show()
    
    # strain energy density
    fig = plt.figure(3, dpi=600)
    plt.plot(x, W, label=r'$W$')
    plt.title('Mixed Test')
    plt.xlabel('load step')
    plt.ylabel('stress tensor')
    plt.grid()
    plt.legend()
    plt.show()
    
plot()