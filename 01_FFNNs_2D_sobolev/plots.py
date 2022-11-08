#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:30:16 2022

@author: jasper
"""

# %%   
"""
Import modules

"""

from matplotlib import pyplot as plt
from matplotlib import cm
import tensorflow as tf


# %%
"""

Class definition

"""

class Plot:
    def __init__(self, x, y, z, z_cal, reshape_dim, label_dict):
        self.x = x
        self.y = y
        self.z = z
        self.z_cal = z_cal
        self.reshape_dim = reshape_dim
        self.label_dict = label_dict
        
        self.fig = plt.figure(dpi=600)
        self.ax = plt.axes(projection='3d')
        
    def scatter(self):
        self.ax.scatter(self.x, self.y, self.z_cal, c='green', 
                        label='calibration data')
        plt.legend()
        
    def surf(self):
        X = tf.reshape(self.x, self.reshape_dim)
        Y = tf.reshape(self.y, self.reshape_dim)
        Z = tf.reshape(self.z, self.reshape_dim)
        
        self.surf = self.ax.plot_surface(X, Y, Z, cmap=cm.inferno)
        self.fig.colorbar(self.surf, orientation='vertical', pad=0.15)
        
    def draw(self, kw='all'):
        if kw == 'scatter':
            self.scatter()
        elif kw == 'surf':
            self.surf()
        else:
            self.scatter()
            self.surf()
        
        self.ax.set_xlabel(self.label_dict['x'])
        self.ax.set_ylabel(self.label_dict['y'])
        self.ax.set_zlabel(self.label_dict['z'])
        #self.ax.set_title('Training on functional value and gradient')
        plt.show()