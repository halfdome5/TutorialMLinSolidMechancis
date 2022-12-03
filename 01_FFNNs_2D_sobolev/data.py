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

import tensorflow as tf
import numpy as np
import models as lm

# %%
# own modules
import plots as pl


# %%
"""

Load data model

"""
def makeLayer(r_type, **kwargs):
    cf = {
        'f1': lm.f1,
        'f2': lm.f2,
          }
    class_obj = cf.get(r_type, None)
    if class_obj:
        return class_obj(**kwargs)
    raise ValueError


# %%
"""

Generate data

"""

def f(r_type, show_plot=False, **kwargs):
    
    n = 20
    m = 20
    
    x1s = np.linspace(-4,4,n)
    x2s = np.linspace(-4,4,m)
    x1s, x2s = np.meshgrid(x1s, x2s)
    
    xs = tf.stack((x1s.reshape(-1), x2s.reshape(-1)), axis=1)
    
    model = makeLayer(r_type, **kwargs)
    ys, dys = model(xs)
    
    if show_plot:
        plot_data(xs, ys, dys, [n, m])

    return xs, ys, dys, [n, m]

# %%
"""

Plot data (optional)

"""

def plot_data(xs, ys, dys, reshape_dim):
    
    label_dict = {'x': r'$x_1$',
                  'y': r'$x_2$',
                  'z': r'$f$'}
    p = pl.Plot(xs[:,0], xs[:,1], reshape_dim, label_dict)
    p.add_scatter(ys, label='calibration data')
    p.draw()

    label_dict = {'x': r'$x_1$',
                  'y': r'$x_2$',
                  'z': r'$\frac{\partial f}{\partial x_1}$'}
    p = pl.Plot(xs[:,0], xs[:,1], reshape_dim, label_dict)
    p.add_scatter(dys[:,0], label='calibration data')
    p.draw()

    label_dict = {'x': r'$x_1$',
                  'y': r'$x_2$',
                  'z': r'$\frac{\partial f}{\partial x_2}$'}
    p = pl.Plot(xs[:,0], xs[:,1], reshape_dim, label_dict)
    p.add_scatter(dys[:,1], label='calibration data')
    #p.add_wireframe(dys[:,1])
    p.draw()
    
   
#f(r_type='f2', show_plot=True)