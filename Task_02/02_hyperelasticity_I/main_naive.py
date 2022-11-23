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
Load calibration data

"""

# select load cases for calibration
paths = [
    'data/calibration/biaxial.txt',
    'data/calibration/pure_shear.txt',
    'data/calibration/uniaxial.txt'
    ]
#xs, ys, _, batch_sizes = ld.load_stress_strain_data(paths)
Cs, Ps, _, batch_sizes = ld.load_stress_strain_data(paths)

# %%
'''
Preprocessing

'''

# apply load weighting strategy
#sw = ld.get_sample_weights(Ps, batch_sizes)
sw = np.ones(np.sum(batch_sizes))
  
# reshape inputs
xs, ys = ld.reshape_input(Cs, Ps)


# %%   
"""
Model calibration

"""

t1 = now()
print(t1)

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
h = model.fit([xs], [ys, np.zeros([batch_sizes.sum(),6])], 
              epochs=5000,
              verbose=2,
              sample_weight=sw)

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

# selecte load cases for testing
paths = [
    'data/calibration/biaxial.txt',
    'data/calibration/pure_shear.txt',
    'data/calibration/uniaxial.txt',
    'data/test/biax_test.txt',
    'data/test/mixed_test.txt'
    ]

titles = [
    r'Biaxial calibration',
    r'Pure shear calibration',
    r'Uniaxial calibration',
    r'Biaxial test',
    r'Mixed test'
    ]

fnames = [
    'biaxial',
    'pure_shear',
    'uniaxial',
    'biax_test',
    'mixed_test'
    ]



# evaluate each data set separately
for i, path in enumerate(paths):
    # reference data
    #xs, ys, _, [batch_size] = ld.load_stress_strain_data([path])
    Cs, Ps, _, [batch_size] = ld.load_stress_strain_data([path])
    
    # reshape
    xs, ys = ld.reshape_input(Cs, Ps)
    
    # predict using the trained model
    ys_pred, dys_pred = model.predict(xs)

    # reshape results from Voigt vector to matrix
    P = tf.reshape(ys, [batch_size, 3, 3])
    P_pred = tf.reshape(ys_pred, [batch_size, 3, 3]) 
    
    # plot right Chauchy-Green tensor
    pl.plot_right_cauchy_green_tensor(xs, titles[i], fnames[i])
    
    # plot stress tensor
    pl.plot_stress_tensor_prediction(P, P_pred, titles[i], fnames[i])
    
    # compute and print errors
    print('''------------------------------------
--- {} ---
------------------------------------'''.format(path))
    mse, mae = compute_metrics(P[:, 0, 0], P_pred[:, 0, 0])
    print('''P_11:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
    mse, mae = compute_metrics(P[:, 0, 1], P_pred[:, 0, 1])
    print('''P_12:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
    mse, mae = compute_metrics(P[:, 0, 2], P_pred[:, 0, 2])
    print('''P_13:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
    mse, mae = compute_metrics(P[:, 1, 0], P_pred[:, 1, 0])
    print('''P_21:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
    mse, mae = compute_metrics(P[:, 1, 1], P_pred[:, 1, 1])
    print('''P_22:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
    mse, mae = compute_metrics(P[:, 1, 2], P_pred[:, 1, 2])
    print('''P_23:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
    mse, mae = compute_metrics(P[:, 2, 0], P_pred[:, 2, 0])
    print('''P_31:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
    mse, mae = compute_metrics(P[:, 2, 1], P_pred[:, 2, 1])
    print('''P_32:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))
    mse, mae = compute_metrics(P[:, 2, 2], P_pred[:, 2, 2])
    print('''P_33:\tMSE = {}, \tMAE = {}\n'''.format(mse, mae))   

# check normalization criterion
xs = np.array([[1, 0, 0, 1, 0, 1],
              [1, 0, 0, 1, 0, 1]])
ys_pred, dys_pred = model.predict(xs)
print(ys_pred)
print(dys_pred)

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