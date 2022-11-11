"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Jasper Schommartz, Toprak Kis
         
11/2022
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

# class: Plot
#
# decription: class for creating different types of 3D plots
# call strategy:
#   1. initialize by calling Plot()
#   2. add plots using add_... methods
#   3. draw all plots in the same figure by calling draw()

class Plot:
    def __init__(self, x, y, reshape_dim, label_dict):
        self.x = x
        self.y = y
        self.reshape_dim = reshape_dim
        self.label_dict = label_dict
        
        self.fig = plt.figure(dpi=600)
        self.ax = plt.axes(projection='3d')
        
    def add_scatter(self, z, label):
        self.ax.scatter(self.x, self.y, z, 
                        c='green', 
                        label='calibration data', 
                        s=2)
        plt.legend()
        
    def add_surf(self, z):
        X = tf.reshape(self.x, self.reshape_dim)
        Y = tf.reshape(self.y, self.reshape_dim)
        Z = tf.reshape(z, self.reshape_dim)
        
        self.surf = self.ax.plot_surface(X, Y, Z, 
                                         cmap=cm.inferno,
                                         alpha=0.9)
        self.fig.colorbar(self.surf, orientation='vertical', pad=0.15)
        
    def add_wireframe(self, z):
        X = tf.reshape(self.x, self.reshape_dim)
        Y = tf.reshape(self.y, self.reshape_dim)
        Z = tf.reshape(z, self.reshape_dim)
        
        self.surf = self.ax.plot_wireframe(X, Y, Z, 
                                           rstride=1, cstride=1)
        
    def draw(self):
        # draw plots based on key word
        self.ax.set_xlabel(self.label_dict['x'])
        self.ax.set_ylabel(self.label_dict['y'])
        self.ax.set_zlabel(self.label_dict['z'])
        if 'title' in self.label_dict:
            self.ax.set_title(self.label_dict['title'])
        plt.show()
