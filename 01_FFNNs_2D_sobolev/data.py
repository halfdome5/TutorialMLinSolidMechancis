"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Jasper Schommartz, Toprak Kis
         
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
# own modules
import plots as p


# %%
"""

Load data model

"""
def makeLayer(r_type, **kwargs):
    cf = {
        'f1': lm.f1,
        'f2': lm.f2,
        'f2Sobolev': lm.f2Sobolev
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
    
    label_dict = {'x': 'x1',
                  'y': 'x2',
                  'z': 'f'}
    p_func_value = p.Plot(xs[:,0], xs[:,1], [], ys, reshape_dim, label_dict)
    p_func_value.draw('scatter')

    label_dict = {'x': 'x1',
                  'y': 'x2',
                  'z': 'dfdx1'}
    p_grad_x1 = p.Plot(xs[:,0], xs[:,1], [], dys[:,0], reshape_dim, label_dict)
    p_grad_x1.draw('scatter')

    label_dict = {'x': 'x1',
                  'y': 'x2',
                  'z': 'dfdx2'}
    p_grad_x2 = p.Plot(xs[:,0], xs[:,1], [], dys[:,1], reshape_dim, label_dict)
    p_grad_x2.draw('scatter')
    
   
f(r_type='f2Sobolev', show_plot=True)