#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:24:45 2022

@author: jasper
"""

#%%
'''
import modules

'''
import numpy as np
import tensorflow as tf

# user defined modules
import models as lm
import data as ld
from plots import plot_imported_data, plot_invariants, plot_potential
from plots import plot_stress_tensor

# %%
'''
metrics

'''

def compute_metrics(x1, x2):
    mse = tf.keras.metrics.mean_squared_error(x1, x2)
    mae = tf.keras.metrics.mean_absolute_error(x1, x2)
    return mse, mae

# %%
'''
evaluate imported F, P and W data

'''

# load deformation, stress and energy data
F, P, W = ld.read_txt('data/calibration/biaxial.txt')

plot_imported_data(F, P, W)

# %%
'''
evaluate invariants

'''

# load invariants from data
arr = np.loadtxt('data/invariants/I_biaxial.txt')

I1 = arr[:, :1]
J = arr[:, 1:2]
I4 = arr[:, 2:3]
I5 = arr[:, 3:]
invariants = [I1, J, I4, I5]

# compute analytical invariants
analytical_invariants = ld.compute_invariants(F)

# evaluate invariants
plot_invariants(invariants, analytical_invariants)
mse, mae = compute_metrics(I1.T, analytical_invariants[0])
print('''I1:\tMSE = {}, MAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(I1.T, analytical_invariants[1])
print('''J:\tMSE = {}, MAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(I1.T, analytical_invariants[2])
print('''I4:\tMSE = {}, MAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(I1.T, analytical_invariants[3])
print('''I5:\tMSE = {}, MAE = {}\n'''.format(mse, mae))


# %%
'''
evaluate strain energy density

'''

# compute analytical potential
W_analytical = ld.W(F)

# evalutation
plot_potential(W, W_analytical)
mse, mae = compute_metrics(W.T, W_analytical)
print('''W:\tMSE = {}, MAE = {}\n'''.format(mse, mae))

# %%
'''
evaluta Piola-Kirchhoff stress tensor

'''

# compute analytical stress tensor
P_analytical = ld.P(ld.W)(F)

# evaluation
plot_stress_tensor(P, P_analytical)
mse, mae = compute_metrics(P[:, 0, 0].T, P_analytical[:, 0, 0])
print('''P_11:\tMSE = {}, MAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 0, 1].T, P_analytical[:, 0, 1])
print('''P_12:\tMSE = {}, MAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 0, 2].T, P_analytical[:, 0, 2])
print('''P_13:\tMSE = {}, MAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 1, 0].T, P_analytical[:, 1, 0])
print('''P_21:\tMSE = {}, MAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 1, 1].T, P_analytical[:, 1, 1])
print('''P_22:\tMSE = {}, MAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 1, 2].T, P_analytical[:, 1, 2])
print('''P_23:\tMSE = {}, MAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 2, 0].T, P_analytical[:, 2, 0])
print('''P_31:\tMSE = {}, MAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 2, 1].T, P_analytical[:, 2, 1])
print('''P_32:\tMSE = {}, MAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 2, 2].T, P_analytical[:, 2, 2])
print('''P_33:\tMSE = {}, MAE = {}\n'''.format(mse, mae))

