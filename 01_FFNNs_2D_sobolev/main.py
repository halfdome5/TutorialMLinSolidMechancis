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

model = lm.main(r_type='InputConvex', sobolev_training=True)
model.summary()

# %%   
"""
Load data

"""

#xs, ys, dys, n, m = ld.f(r_type='f2Sobolev', show_plot=True)
xs, ys, dys, reshape_dim = ld.f(r_type='f2Sobolev', show_plot=True)

# %%   
"""
Model calibration

"""

t1 = now()
print(t1)

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
h = model.fit([xs], [ys, dys], epochs = 1500,  verbose = 2)

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

label_dict = {'x': 'x1',
              'y': 'x2',
              'z': 'f'}
p_function_value = p.Plot(xs[:,0], xs[:,1], ys_pred, ys, 
                            reshape_dim, label_dict)
p_function_value.draw()

label_dict = {'x': 'x1',
              'y': 'x2',
              'z': 'dfdx1'}
p_grad_x1 = p.Plot(xs[:,0], xs[:,1], dys_pred[:,0], dys[:,0], 
                     reshape_dim, label_dict)
p_grad_x1.draw()

label_dict = {'x': 'x1',
              'y': 'x2',
              'z': 'dfdx2'}
p_grad_x2 = p.Plot(xs[:,0], xs[:,1], dys_pred[:,1], dys[:,1], 
                     reshape_dim, label_dict)
p_grad_x2.draw()


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
