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
import plots as p



# %%   
"""
Load model

"""
training_method = 'gradient' # 'function', 'gradient', 'sobolev'
model = lm.main(r_type='InputConvex', training_method=training_method)
model.summary()

# %%   
"""
Load data

"""

xs, ys, dys, reshape_dim = ld.f(r_type='f2Sobolev', show_plot=True)

# %%   
"""
Model calibration

"""

t1 = now()
print(t1)

epochs = 2500

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
if training_method == 'function':
    h = model.fit([xs], [ys], epochs = epochs,  verbose = 2)
elif training_method == 'sobolev':
    h = model.fit([xs], [ys, dys], epochs = epochs,  verbose = 2)
elif training_method == 'gradient':
    h = model.fit([xs], [dys], epochs = epochs,  verbose = 2)

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
if training_method == 'function':
    ys_pred = model.predict(xs)
elif training_method == 'sobolev':
    ys_pred, dys_pred = model.predict(xs)
elif training_method == 'gradient':
    dys_pred = model.predict(xs)
    

# plot results
if training_method == 'function' or training_method == 'sobolev':
    # function value
    label_dict = {'x': r'$x_1$',
                  'y': r'$x_2$',
                  'z': r'$f$'}
    p_function_value = p.Plot(xs[:,0], xs[:,1], ys_pred, ys, 
                                reshape_dim, label_dict)
    p_function_value.draw()
    
    # difference of function value
    label_dict = {'x': r'$x_1$',
                  'y': r'$x_2$',
                  'z': r'$f-f_{pred}$'}
    p_function_value = p.Plot(xs[:,0], xs[:,1], ys_pred.T - ys, [], 
                                reshape_dim, label_dict)
    p_function_value.draw('surf')
    
    
if training_method == 'sobolev' or training_method == 'gradient':
    # gradient in x1 direction
    label_dict = {'x': r'$x_1$',
                  'y': r'$x_2$',
                  'z': r'$\frac{\partial f}{\partial x_1}$'}
    p_grad_x1 = p.Plot(xs[:,0], xs[:,1], dys_pred[:,0], dys[:,0], 
                         reshape_dim, label_dict)
    p_grad_x1.draw()
    
    # gradient in x2 direction
    label_dict = {'x': r'$x_1$',
                  'y': r'$x_2$',
                  'z': r'$\frac{\partial f}{\partial x_2}$'}
    p_grad_x2 = p.Plot(xs[:,0], xs[:,1], dys_pred[:,1], dys[:,1], 
                         reshape_dim, label_dict)
    p_grad_x2.draw()
    
    # difference of gradient in x1 direction
    label_dict = {'x': r'$x_1$',
                  'y': r'$x_2$',
                  'z': r'$\frac{\partial f}{\partial x_2}-\frac{\partial f_{pred}}{\partial x_2}$'}
    p_function_value = p.Plot(xs[:,0], xs[:,1], dys_pred[:,0] - dys[:,0], [], 
                                reshape_dim, label_dict)
    p_function_value.draw('surf')
    
    # difference of gradient in x2 direction
    label_dict = {'x': r'$x_1$',
                  'y': r'$x_2$',
                  'z': r'$\frac{\partial f}{\partial x_2}-\frac{\partial f_{pred}}{\partial x_2}$'}
    p_function_value = p.Plot(xs[:,0], xs[:,1], dys_pred[:,1] - dys[:,1], [], 
                                reshape_dim, label_dict)
    p_function_value.draw('surf')


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