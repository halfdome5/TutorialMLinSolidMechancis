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
model = lm.main(r_type='Naive', loss_weights=lw)
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

# xs = F, ys = P
xs, _, ys, batch_sizes = ld.load_stress_strain_data(paths)
dys = np.zeros(xs.shape) # placeholder

# %%
'''
Preprocessing

'''

# apply load weighting strategy
sw = ld.get_sample_weights(xs, batch_sizes)
#sw = np.ones(np.sum(batch_sizes))


# %%
"""
Model calibration

"""

t1 = now()
print(t1)

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
h = model.fit([xs], [ys, dys],
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
    'Biaxial calibration',
    'Pure shear calibration',
    'Uniaxial calibration',
    'Biaxial test',
    'Mixed test'
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
    xs, _, ys, [batch_size] = ld.load_stress_strain_data([path])
    
    # Evaluate the model on the test data using `evaluate`
    print("\nEvaluate on test data: {}".format(titles[i]))
    results = model.evaluate(xs, [ys, dys[:batch_size]])
    
    # predict using the trained model
    ys_pred, _ = model.predict(xs)
    P = ys
    P_pred = ys_pred
    
    # plot right Chauchy-Green tensor
    Cs = tf.einsum('ikj,ikl->ijl', xs, xs)
    pl.plot_right_cauchy_green_tensor(ld.reshape_C(Cs), titles[i], fnames[i])
    
    # plot stress tensor
    pl.plot_stress_tensor_prediction(P, P_pred, titles[i], fnames[i])

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