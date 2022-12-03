#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:32:50 2022

@author: jasper
"""

# %%
'''
import modules

'''
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# %%
"""
Plots for reviewing imported data

"""

def plot_imported_data(F, P, W):
    ''' Plots imported deformation gradient, Piola-Kirchhoff stress tensor
    and strain energy density '''
    x = np.arange(W.size) + 1

    # deformation gradient
    fig = plt.figure(dpi=600)
    plt.plot(x, F[:, 0, 0], color='firebrick', marker='s', markevery=10, label=r'$F_{11}$')
    plt.plot(x, F[:, 1, 0], color='green', marker='^', markevery=15, label=r'$F_{21}$')
    plt.plot(x, F[:, 2, 0], color='lightgrey', marker='^', markevery=18, label=r'$F_{31}$')
    plt.plot(x, F[:, 0, 1], color='cornflowerblue', marker='^', markevery=13, label=r'$F_{12}$')
    plt.plot(x, F[:, 1, 1], color='navy', marker='s', markevery=11, label=r'$F_{22}$')
    plt.plot(x, F[:, 2, 1], color='lightgrey', marker='^', markevery=18, label=r'$F_{32}$')
    plt.plot(x, F[:, 0, 2], color='lightgrey', marker='^', markevery=14, label=r'$F_{13}$')
    plt.plot(x, F[:, 1, 2], color='lightgrey', marker='^', markevery=16, label=r'$F_{23}$')
    plt.plot(x, F[:, 2, 2], color='darkorange', marker='s', markevery=12, label=r'$F_{33}$')
    plt.xlabel('load step')
    plt.ylabel(r'$F_{ij}$')
    plt.xlim(np.min(x), np.max(x))
    plt.grid()
    plt.legend(handlelength=0, loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)
    fig.savefig('images/F.png', dpi=fig.dpi, bbox_inches='tight')
    
    # stress tensor
    fig = plt.figure(dpi=600)
    plt.plot(x, P[:, 0, 0], color='firebrick', marker='s', markevery=10, label=r'$P_{11}$')
    plt.plot(x, P[:, 1, 0], color='green', marker='^', markevery=15, label=r'$P_{21}$')
    plt.plot(x, P[:, 2, 0], color='lightgrey', marker='^', markevery=17, label=r'$P_{31}$')
    plt.plot(x, P[:, 0, 1], color='cornflowerblue',marker='^', markevery=13, label=r'$P_{12}$')
    plt.plot(x, P[:, 1, 1], color='navy', marker='s', markevery=11, label=r'$P_{22}$')
    plt.plot(x, P[:, 2, 1], color='lightgrey', marker='^', markevery=18, label=r'$P_{32}$')
    plt.plot(x, P[:, 0, 2], color='lightgrey', marker='^', markevery=14, label=r'$P_{13}$')
    plt.plot(x, P[:, 1, 2], color='lightgrey', marker='^', markevery=16, label=r'$P_{23}$')
    plt.plot(x, P[:, 2, 2], color='darkorange', marker='s', markevery=12, label=r'$P_{33}$')
    plt.xlabel('load step')
    plt.ylabel(r'$P_{ij}$')
    plt.xlim(np.min(x), np.max(x))
    plt.grid()
    plt.legend(handlelength=0, loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)
    fig.savefig('images/P.png', dpi=fig.dpi, bbox_inches='tight')
    
    # strain energy density
    fig = plt.figure(dpi=600)
    plt.plot(x, W, label=r'$W$')
    plt.xlabel('load step')
    plt.ylabel(r'W')
    plt.xlim(np.min(x), np.max(x))
    plt.grid()
    plt.legend()
    fig.savefig('images/W.png', dpi=fig.dpi, bbox_inches='tight')
    plt.show()

# %%
"""
Plot for evaluation of analytical model

"""

def plot_invariants(invariants, analytical_invariants):
    ''' Plots analytically computed and reference invariants '''
    I1 = invariants[0]
    J = invariants[1]
    I4 = invariants[2]
    I5 = invariants[3]
    I1_ana = analytical_invariants[0]
    J_ana = analytical_invariants[1]
    I4_ana = analytical_invariants[2]
    I5_ana = analytical_invariants[3]
    
    x = np.arange(invariants[0].size) + 1
    
    fig = plt.figure(dpi=600)
    plt.plot(x, I1, color='firebrick', linestyle='None', marker='s', markevery=10, label=r'$I_{1}$')
    plt.plot(x, J, color='navy', linestyle='None', marker='s', markevery=10, label=r'$J$')
    plt.plot(x, I4, color='darkorange', linestyle='None', marker='s', markevery=10, label=r'$I_4$')
    plt.plot(x, I5, color='green', linestyle='None', marker='s', markevery=10, label=r'$I_5$')
    plt.plot(x, I1_ana, color='firebrick')
    plt.plot(x, J_ana, color='navy')
    plt.plot(x, I4_ana, color='darkorange')
    plt.plot(x, I5_ana, color='green')
    plt.title('\u25a0 data, \u2015 analytical')
    plt.xlabel('load step')
    plt.ylabel(r'$I$')
    plt.xlim(np.min(x), np.max(x))
    plt.grid()
    plt.legend(handlelength=0, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig('images/I_analytical.png', dpi=fig.dpi, bbox_inches='tight')
    plt.show()
    
def plot_potential(W, W_ana):
    ''' Plots analytically computed and reference potential '''
    x = np.arange(W.size) + 1
    # deformation gradient
    fig = plt.figure(dpi=600)
    plt.plot(x, W, color='firebrick', linestyle='None', marker='s', markevery=10, label='data')
    plt.plot(x, W_ana, color='firebrick', label='analytical')
    plt.xlabel('load step')
    plt.ylabel(r'$W$')
    plt.xlim(np.min(x), np.max(x))
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig('images/W_analytical.png', dpi=fig.dpi, bbox_inches='tight')
    plt.show()
    
def plot_stress_tensor_analytical(P, P_ana):
    ''' Plots analytically computed stress tensor and reference stress tensor '''
    x = np.arange(np.size(P,axis=0)) + 1

    # difference
    fig = plt.figure(dpi=600)
    plt.plot(x, P[:, 0, 0] - P_ana[:, 0, 0], color='firebrick', label=r'$P_{11}$')
    plt.plot(x, P[:, 1, 0] - P_ana[:, 1, 0], color='green', label=r'$P_{21}$')
    plt.plot(x, P[:, 2, 0] - P_ana[:, 2, 0], color='lightgrey', label=r'$P_{31}$')
    plt.plot(x, P[:, 0, 1] - P_ana[:, 0, 1], color='cornflowerblue', label=r'$P_{12}$')
    plt.plot(x, P[:, 1, 1] - P_ana[:, 1, 1], color='navy', label=r'$P_{22}$')
    plt.plot(x, P[:, 2, 1] - P_ana[:, 2, 1], color='lightgrey', label=r'$P_{32}$')
    plt.plot(x, P[:, 0, 2] - P_ana[:, 0, 2], color='lightgrey', label=r'$P_{13}$')
    plt.plot(x, P[:, 1, 2] - P_ana[:, 1, 2], color='lightgrey', label=r'$P_{23}$')
    plt.plot(x, P[:, 2, 2] - P_ana[:, 2, 2], color='darkorange', label=r'$P_{33}$')
    plt.xlabel('load step')
    plt.ylabel(r'$P_{ij}-P_{ij,analyical}$')
    plt.xlim(np.min(x), np.max(x))
    plt.grid()
    leg = plt.legend(handlelength=0.75, loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    fig.savefig('images/P_difference.png', dpi=fig.dpi, bbox_inches='tight')
    
    fig = plt.figure(dpi=600)
    # imported data
    plt.plot(x, P[:, 0, 0], color='firebrick', linestyle='None', marker='s', markevery=10, label=r'$P_{11}$')
    plt.plot(x, P[:, 1, 0], color='green', linestyle='None', marker='^', markevery=15, label=r'$P_{21}$')
    plt.plot(x, P[:, 2, 0], color='lightgrey', linestyle='None', marker='^', markevery=17, label=r'$P_{31}$')
    plt.plot(x, P[:, 0, 1], color='cornflowerblue', linestyle='None', marker='^', markevery=13, label=r'$P_{12}$')
    plt.plot(x, P[:, 1, 1], color='navy', linestyle='None', marker='s', markevery=11, label=r'$P_{22}$')
    plt.plot(x, P[:, 2, 1], color='lightgrey', linestyle='None', marker='^', markevery=18, label=r'$P_{32}$')
    plt.plot(x, P[:, 0, 2], color='lightgrey', linestyle='None', marker='^', markevery=14, label=r'$P_{13}$')
    plt.plot(x, P[:, 1, 2], color='lightgrey', linestyle='None', marker='^', markevery=16, label=r'$P_{23}$')
    plt.plot(x, P[:, 2, 2], color='darkorange', linestyle='None', marker='s', markevery=12, label=r'$P_{33}$')
    
    # analytical data
    plt.plot(x, P_ana[:, 0, 0], color='firebrick')
    plt.plot(x, P_ana[:, 1, 0], color='green')
    plt.plot(x, P_ana[:, 2, 0], color='lightgrey')
    plt.plot(x, P_ana[:, 0, 1], color='cornflowerblue')
    plt.plot(x, P_ana[:, 1, 1], color='navy')
    plt.plot(x, P_ana[:, 2, 1], color='lightgrey')
    plt.plot(x, P_ana[:, 0, 2], color='lightgrey')
    plt.plot(x, P_ana[:, 1, 2], color='lightgrey')
    plt.plot(x, P_ana[:, 2, 2], color='darkorange')
    plt.title('\u25a0 data, \u2015 analytical')
    plt.xlabel('load step')
    plt.ylabel(r'$P_{ij}$')
    plt.xlim(np.min(x), np.max(x))
    plt.grid()
    plt.legend(handlelength=0, loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)
    fig.savefig('images/P_analytical.png', dpi=fig.dpi, bbox_inches='tight')
    plt.show()

# %%
"""
Plot for evaluation of trained model

"""

    
def plot_stress_tensor_prediction(P, P_pred, title, fname=None):
    ''' Plots predicted stress tensor and reference stress tensor '''
    x = np.arange(np.size(P,axis=0)) + 1
    P = tf.cast(P, dtype='float32')
    P_pred = tf.cast(P_pred, dtype='float32')

    # difference
    fig = plt.figure(dpi=600)
    plt.plot(x, P[:, 0, 0] - P_pred[:, 0, 0], color='firebrick', label=r'$P_{11}$')
    plt.plot(x, P[:, 1, 0] - P_pred[:, 1, 0], color='green', label=r'$P_{21}$')
    plt.plot(x, P[:, 2, 0] - P_pred[:, 2, 0], color='lightgrey', label=r'$P_{31}$')
    plt.plot(x, P[:, 0, 1] - P_pred[:, 0, 1], color='cornflowerblue', label=r'$P_{12}$')
    plt.plot(x, P[:, 1, 1] - P_pred[:, 1, 1], color='navy', label=r'$P_{22}$')
    plt.plot(x, P[:, 2, 1] - P_pred[:, 2, 1], color='lightgrey', label=r'$P_{32}$')
    plt.plot(x, P[:, 0, 2] - P_pred[:, 0, 2], color='lightgrey', label=r'$P_{13}$')
    plt.plot(x, P[:, 1, 2] - P_pred[:, 1, 2], color='lightgrey', label=r'$P_{23}$')
    plt.plot(x, P[:, 2, 2] - P_pred[:, 2, 2], color='darkorange', label=r'$P_{33}$')
    plt.title(title)
    plt.xlabel('load step')
    plt.ylabel(r'$P_{ij}-P_{ij,prediction}$')
    plt.xlim(np.min(x), np.max(x))
    plt.grid()
    leg = plt.legend(handlelength=0.75, loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    if fname is not None:
        fig.savefig(f'images/P_pred_diff_{fname}.png', dpi=fig.dpi, bbox_inches='tight')
    
    fig = plt.figure(dpi=600)
    # imported data
    plt.plot(x, P[:, 0, 0], linestyle='None', color='firebrick',marker='s', markevery=10, label=r'$P_{11}$')
    plt.plot(x, P[:, 1, 0], linestyle='None', color='green', marker='^', markevery=15, label=r'$P_{21}$')
    plt.plot(x, P[:, 2, 0], linestyle='None', color='lightgrey', marker='^', markevery=17, label=r'$P_{31}$')
    plt.plot(x, P[:, 0, 1], linestyle='None', color='cornflowerblue', marker='^', markevery=13, label=r'$P_{12}$')
    plt.plot(x, P[:, 1, 1], linestyle='None', color='navy', marker='s', markevery=11, label=r'$P_{22}$')
    plt.plot(x, P[:, 2, 1], linestyle='None', color='lightgrey', marker='^', markevery=18, label=r'$P_{32}$')
    plt.plot(x, P[:, 0, 2], linestyle='None', color='lightgrey', marker='^', markevery=14, label=r'$P_{13}$')
    plt.plot(x, P[:, 1, 2], linestyle='None', color='lightgrey', marker='^', markevery=16, label=r'$P_{23}$')
    plt.plot(x, P[:, 2, 2], linestyle='None', color='darkorange', marker='s', markevery=12, label=r'$P_{33}$')
    
    # predicted data
    plt.plot(x, P_pred[:, 0, 0], color='firebrick')
    plt.plot(x, P_pred[:, 1, 0], color='green')
    plt.plot(x, P_pred[:, 2, 0], color='lightgrey')
    plt.plot(x, P_pred[:, 0, 1], color='cornflowerblue')
    plt.plot(x, P_pred[:, 1, 1], color='navy')
    plt.plot(x, P_pred[:, 2, 1], color='lightgrey')
    plt.plot(x, P_pred[:, 0, 2], color='lightgrey')
    plt.plot(x, P_pred[:, 1, 2], color='lightgrey')
    plt.plot(x, P_pred[:, 2, 2], color='darkorange')
    plt.title(f'{title}\n\u25a0 data, \u2015 prediction')
    plt.xlabel('load step')
    plt.ylabel(r'$P_{ij}$')
    plt.xlim(np.min(x), np.max(x))
    plt.grid()
    plt.legend(handlelength=0, loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)
    if fname is not None:
        fig.savefig(f'images/P_pred_{fname}.png', dpi=fig.dpi, bbox_inches='tight')
    plt.show()
    
def plot_right_cauchy_green_tensor(C, title, fname=None):
    '''
    Plots the components of the Cauchy-Green tensor which is provided in Voigt
    notation with six independent compontents.
    
    C = [C11, C12, C13, C22, C23, C33]
    '''
    x = np.arange(np.size(C,axis=0)) + 1

    # difference
    fig = plt.figure(dpi=600)
    plt.plot(x, C[:, 0], color='firebrick', marker='s', markevery=10, label=r'$C_{11}$')
    plt.plot(x, C[:, 1], color='cornflowerblue', marker='^', markevery=13, label=r'$C_{12}$')
    plt.plot(x, C[:, 2], color='lightgrey', marker='^', markevery=14, label=r'$C_{13}$')
    plt.plot(x, C[:, 3], color='navy', marker='s', markevery=11, label=r'$C_{22}$')
    plt.plot(x, C[:, 4], color='lightgrey', marker='^', markevery=16, label=r'$C_{23}$')
    plt.plot(x, C[:, 5], color='darkorange', marker='s', markevery=12, label=r'$C_{33}$')
    plt.title(title)
    plt.xlabel('load step')
    plt.ylabel(r'$C_{ij}$')
    plt.xlim(np.min(x), np.max(x))
    plt.grid()
    plt.legend(handlelength=0, loc='center left', bbox_to_anchor=(1, 0.5))
    if fname is not None:
        fig.savefig(f'images/C_{fname}.png', dpi=fig.dpi, bbox_inches='tight')
  

def plot_potential_prediction(W, W_pred, title, fname=None):
    ''' plots predicted and reference potential '''
    x = np.arange(W.shape[0]) + 1
    # deformation gradient
    fig = plt.figure(dpi=600)
    plt.plot(x, W, color='firebrick', linestyle='None', marker='s', markevery=10, label='data')
    plt.plot(x, W_pred, color='firebrick', label='prediction')
    plt.title(title)
    plt.xlabel('load step')
    plt.ylabel(r'$W$')
    plt.xlim(np.min(x), np.max(x))
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if fname is not None:
        fig.savefig(f'images/W_prediction_{fname}.png', dpi=fig.dpi, bbox_inches='tight')
    plt.show()


# %%
"""
Model calibration

"""

def plot_calibration_loss(h):
    ''' Plots loss for every calibration epoch '''
    fig = plt.figure(1, dpi=600)
    plt.semilogy(h.history['loss'], label='training loss')
    plt.xlabel('calibration epoch')
    plt.ylabel('log$_{10}$ MSE')
    plt.grid(which='both')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig('images/loss.png', dpi=fig.dpi, bbox_inches='tight')