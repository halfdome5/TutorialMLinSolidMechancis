"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein
         
08/2022
"""

# %%   
"""
Import modules

"""

import tensorflow as tf
import numpy as np
import models as lm
from matplotlib import pyplot as plt

# %%
"""

Load data model

"""
def load(r_type, verbose=0):
    model = lm.main(r_type='f1')
    if verbose == 1:
        model.summary()
        print(model.get_weights())
    return model


# %%
"""

Generate data

"""

def f(r_type, show_plot=False, **kwargs):
    
    n = 20
    m = 20
    
    x1s = np.linspace(-4,4,n)
    x2s = np.linspace(-4,4,m)
    x1s,x2s = np.meshgrid(x1s,x2s)
    
    xs = tf.stack((x1s.reshape(-1), x2s.reshape(-1)), axis=1)
    
    model = load(r_type, **kwargs)
    ys = model.predict(xs)
    
    if show_plot:
        plot_data(xs, ys)

    return xs, ys, n, m

# %%
"""

Plot data (optional)

"""

def plot_data(xs,ys):

    plt.figure(dpi=600)
    ax = plt.axes(projection='3d')
    ax.grid()
    
    ax.scatter(xs[:,0],xs[:,1], ys, c='green', label='calibration data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.ylabel('y')
    plt.legend()
    plt.show()