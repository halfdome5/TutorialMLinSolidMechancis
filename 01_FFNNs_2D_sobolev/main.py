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
from matplotlib import pyplot as plt
from matplotlib import cm
import tensorflow as tf
import datetime
now = datetime.datetime.now

# %% Own modules
import data as ld
import models as lm



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

xs, ys, dys, n, m = ld.f(r_type='f2Sobolev', show_plot=True)

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

# plot function values
fig = plt.figure(2, dpi=600)
ax = plt.axes(projection='3d')
ax.grid()

ax.scatter(xs[:,0], xs[:,1], ys, c='green', label='calibration data')
surf = ax.plot_surface(tf.reshape(xs[:,0], [n,m]), tf.reshape(xs[:,1], [n,m]), 
                tf.reshape(ys_pred, [n,m]), cmap=cm.inferno)
fig.colorbar(surf, orientation='vertical', pad=0.1)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f')
plt.legend()
plt.show()

# plot gradients in x1
fig = plt.figure(3, dpi=600)
ax = plt.axes(projection='3d')
ax.grid()

ax.scatter(xs[:,0], xs[:,1], dys[:,0], c='green', label='calibration data')
surf = ax.plot_surface(tf.reshape(xs[:,0], [n,m]), tf.reshape(xs[:,1], [n,m]), 
                tf.reshape(dys_pred[:,0], [n,m]), cmap=cm.inferno)
fig.colorbar(surf, orientation='vertical', pad=0.1)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('dfdx1')
plt.legend()
plt.show()

# plot gradients in x1
fig = plt.figure(4, dpi=600)
ax = plt.axes(projection='3d')
ax.grid()

ax.scatter(xs[:,0], xs[:,1], dys[:,1], c='green', label='calibration data')
surf = ax.plot_surface(tf.reshape(xs[:,0], [n,m]), tf.reshape(xs[:,1], [n,m]), 
                tf.reshape(dys_pred[:,1], [n,m]), cmap=cm.inferno)
fig.colorbar(surf, orientation='vertical', pad=0.15)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('dfdx2')
plt.legend()
plt.show()


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
