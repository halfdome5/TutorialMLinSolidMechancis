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
import tensorflow as tf
import datetime
now = datetime.datetime.now

# %% Own modules
import data as ld
import models as lm
import plots as pl



# %%   
"""
Load model

"""
lw = [1, 1]     # output_1 = function value, output_2 = gradient
model = lm.main(r_type='InputConvex', loss_weights=lw)
model.summary()

# %%   
"""
Load data

"""

xs, ys, dys, reshape_dim = ld.f(r_type='f2', show_plot=True)

# %%   
"""
Model calibration

"""

t1 = now()
print(t1)

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
h = model.fit([xs], [ys, dys], epochs=5000, verbose=2)

t2 = now()
print('it took', t2 - t1, '(sec) to calibrate the model')

# plot some results
plt.figure(1, dpi=600)
plt.semilogy(h.history['loss'], label='training loss')
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.legend()


# %%   
"""
Evaluation

"""

ys_pred, dys_pred = model.predict(xs)
    

# function value
label_dict = {'x': r'$x_1$',
              'y': r'$x_2$',
              'z': r'$f$'}
p = pl.Plot(xs[:,0], xs[:,1], reshape_dim, label_dict)
p.add_scatter(ys, label='calibration data')
p.add_surf(ys_pred)
p.draw()

# difference of predicted and actual function value
label_dict = {'x': r'$x_1$',
              'y': r'$x_2$',
              'z': r'$f-f_{pred}$'}
p = pl.Plot(xs[:,0], xs[:,1], reshape_dim, label_dict)
p.add_surf(ys_pred.T - ys)
p.draw()
    

# gradient in x1 direction
label_dict = {'x': r'$x_1$',
              'y': r'$x_2$',
              'z': r'$\frac{\partial f}{\partial x_1}$'}
p = pl.Plot(xs[:,0], xs[:,1], reshape_dim, label_dict)
p.add_scatter(dys[:,0], label='calibration data')
p.add_surf(dys_pred[:,0])
p.draw()

# gradient in x2 direction
label_dict = {'x': r'$x_1$',
              'y': r'$x_2$',
              'z': r'$\frac{\partial f}{\partial x_2}$'}
p = pl.Plot(xs[:,0], xs[:,1], reshape_dim, label_dict)
p.add_scatter(dys[:,1], label='calibration data')
p.add_surf(dys_pred[:,1])
p.draw()

# difference of predicted and actual gradient in x1 direction
label_dict = {'x': r'$x_1$',
              'y': r'$x_2$',
              'z': r'$\frac{\partial f}{\partial x_2}-\frac{\partial f_{pred}}{\partial x_2}$'}
p = pl.Plot(xs[:,0], xs[:,1], reshape_dim, label_dict)
p.add_surf(dys_pred[:,0] - dys[:,0])
p.draw()

# difference of predicted and actual gradient in x2 direction
label_dict = {'x': r'$x_1$',
              'y': r'$x_2$',
              'z': r'$\frac{\partial f}{\partial x_2}-\frac{\partial f_{pred}}{\partial x_2}$'}
p = pl.Plot(xs[:,0], xs[:,1], reshape_dim, label_dict)
p.add_surf(dys_pred[:,1] - dys[:,1])
p.draw()


# %% 
"""
Model parameters

"""

def print_model_parameters():
    model.summary()
    for idx, layer in enumerate(model.layers):
        print(layer.name, layer)
        #print(layer.weights, "\n")
        print(layer.get_weights())
        
#print_model_parameters()