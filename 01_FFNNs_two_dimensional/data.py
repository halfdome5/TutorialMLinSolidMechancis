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

ws = np.array([[1,-1]])
bs = np.array([0])

model = lm.main(model_type='nontrainable', weights=ws.reshape((2,1)), bias=bs)
model.summary()
print(model.get_weights())


# %%
"""

Generate data

"""

def f():
    
    x1s = np.linspace(-4,4,20)**2
    x2s = np.linspace(-4,4,20)**2
    x1s,x2s = np.meshgrid(x1s,x2s)
    
    xs = tf.stack((x1s.reshape(-1), x2s.reshape(-1)), axis=1)
    
    ys = model.predict(xs)

    return xs, ys

# %%   
"""
Generate data for a bathtub function

"""

def bathtub():

    xs = np.linspace(1,10,450)
    ys = np.concatenate([np.square(xs[0:150]-4)+1, \
                          1+0.1*np.sin(np.linspace(0,3.14,90)), np.ones(60), \
                          np.square(xs[300:450]-7)+1])
    
        
    xs = xs / 10.0
    ys = ys / 10.0

    xs_c = np.concatenate([xs[0:240], xs[330:420]])
    ys_c = np.concatenate([ys[0:240], ys[330:420]])

    xs = tf.expand_dims(xs, axis = 1)
    ys = tf.expand_dims(ys, axis = 1)

    xs_c = tf.expand_dims(xs_c, axis = 1)
    ys_c = tf.expand_dims(ys_c, axis = 1)
    
    return xs, ys, xs_c, ys_c

# %%
"""

Plot data

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
    
xs, ys = f()
plot_data(np.sqrt(xs), ys)