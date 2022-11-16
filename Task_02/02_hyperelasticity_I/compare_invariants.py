#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:36:43 2022

@author: jasper
"""

#%%
'''
import modules

'''
import numpy as np
import tensorflow as tf

# user defined modules
import data as ld


# %%
'''
invariants

'''

# load invariants from data
path = 'data/invariants/I_biaxial.txt'

arr = np.loadtxt(path)

I1_c = arr[:, :1]
J_c = arr[:, 1:2]
I4_c = arr[:, 2:3]
I5_c = arr[:, 3:]

#print(J_c)

# caluculate invariants
# from deformation gradient G_ti
path = 'data/calibration/biaxial.txt'

F_c, P_c, W_c = ld.read_txt(path)
G_ti = np.array([[4, 0, 0],
              [0, 0.5, 0],
              [0, 0, 0.5]])
I1, J, I4, I5 = ld.compute_invariants(F_c, G_ti)

# evaluate conculated invariants    
ld.compare_invariants(I1_c.T, I1, 'I1')
ld.compare_invariants(J_c.T, J, 'J')
ld.compare_invariants(I4_c.T, I4, 'I4')
ld.compare_invariants(I5_c.T, I5, 'I5')


# %%
'''
pontential

'''

# calculate analytical potantial
W = ld.compute_analytical_potential(I1, J, I4, I5)

# evaluate analytical potential
ld.compare_invariants(W_c.T, W, 'W')

