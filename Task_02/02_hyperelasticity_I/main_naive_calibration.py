"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity 1

==================

Authors: Jasper Schommartz, Toprak Kis
         
11/2022
"""


# %%   
"""
Import modules

"""
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import datetime
now = datetime.datetime.now

# %% Own modules
import data as ld
import models as lm
import plots as pl
from metrics import compute_metrics



# %%   
"""
Load model

"""
lw = [1, 0]     # output_1 = function value, output_2 = gradient
model = lm.main(r_type='FeedForward', loss_weights=lw)
model.summary()

# %%   
"""
Load data

"""

# load deformation, stress and energy data
F, P, W = ld.read_txt('data/calibration/uniaxial.txt')
batch_size = tf.shape(W)[0]
# compute right Chauchy-Green tensor
C = tf.einsum('ikj,ikl->ijl',F,F)
# use six intependent components as input
xs = tf.reshape(C, [batch_size, 9])[:,:6]
# reshape output
ys = tf.reshape(P, [batch_size, 9])

# %%   
"""
Model calibration

"""

t1 = now()
print(t1)

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
h = model.fit([xs], [ys, np.zeros([batch_size,6])], epochs=5000, verbose=2)

t2 = now()
print('it took', t2 - t1, '(sec) to calibrate the model')

# plot some results
fig = plt.figure(1, dpi=600)
plt.semilogy(h.history['loss'], label='training loss')
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.legend()
fig.savefig('images/loss.png', dpi=fig.dpi, bbox_inches='tight')


# %%   
"""
Evaluation

"""


ys_pred, dys_pred = model.predict(xs)

# reshape Voigt vector to matrix
P_pred = tf.reshape(ys_pred, [batch_size, 3, 3])

# plot and evaluate stress tensor
pl.plot_stress_tensor_prediction(P, P_pred)
mse, mae = compute_metrics(P[:, 0, 0].T, P_pred[:, 0, 0])
print('''P_11:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 0, 1].T, P_pred[:, 0, 1])
print('''P_12:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 0, 2].T, P_pred[:, 0, 2])
print('''P_13:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 1, 0].T, P_pred[:, 1, 0])
print('''P_21:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 1, 1].T, P_pred[:, 1, 1])
print('''P_22:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 1, 2].T, P_pred[:, 1, 2])
print('''P_23:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 2, 0].T, P_pred[:, 2, 0])
print('''P_31:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 2, 1].T, P_pred[:, 2, 1])
print('''P_32:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
mse, mae = compute_metrics(P[:, 2, 2].T, P_pred[:, 2, 0])
print('''P_33:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))    

# plot right Chauchy-Green tensor
pl.plot_right_cauchy_green_tensor(C)

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