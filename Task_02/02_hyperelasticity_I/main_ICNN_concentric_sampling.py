"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity 1

==================

Authors: Jasper Schommartz, Toprak Kis
         
11/2022
"""


# %% [Import modules]
"""
Import modules

"""
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import datetime
import pandas as pd
now = datetime.datetime.now

# %% Own modules
import data as ld
import models as lm
import plots as pl
#from metrics import compute_metrics

# %%
"""
Load model

"""
lw = [0, 1]     # output_1 = function value, output_2 = gradient
model = lm.main(r_type='PhysicsAugmented', loss_weights=lw)
model.summary()

# %%
"""
Load calibration data

"""

# parameters
TEST_TRAIN_SPLIT = 0.8
FNUM = 100

# create training and test split
fnum = tf.random.shuffle(np.arange(1,FNUM + 1))
ftrain, ftest = tf.split(fnum, [int(TEST_TRAIN_SPLIT * (FNUM + 1)), -1])

paths = ld.generate_concentric_paths(ftrain)
xs, _, _, batch_sizes = ld.load_stress_strain_data(paths)

# compute potential and stress using analytical model
ys = ld.W(xs)
dys = ld.P(ld.W)(xs)

# %%
"""
Preprocessing

"""

# apply load weighting strategy
sw = ld.get_sample_weights(xs, batch_sizes)
 
# reshape inputs
ys = tf.reshape(ys,-1)

# %%
"""
Model calibration

"""

t1 = now()
print(t1)

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
h = model.fit([xs], [ys, dys],
              epochs=300,
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

# evaluate normalization criterion
ys_I, dys_I = model.predict(np.array([np.identity(3)]))
print(f'\nW(I) =\t{ys_I[0,0]}')
print(f'P(I) =\t{dys_I[0,0]}\n\t{dys_I[0,1]}\n\t{dys_I[0,2]}')

# evaluation on training data
def evaluate(paths, showplots=False):
    ''' Preform evaluation '''
    # initialization
    results = np.zeros([len(paths), 3])

    # iteration over load paths
    for i, path in enumerate(paths):
        xs, _, _, _ = ld.load_stress_strain_data([path])

        # compute potential and stress using analytical model
        ys = ld.W(xs)
        dys = ld.P(ld.W)(xs)

        # predict using the trained model
        ys_pred, dys_pred = model.predict(xs)
        P = dys
        P_pred = dys_pred

        # Potential correction - for P training
        # shift reference value ys by normalization offset
        # to ensure reasonable results from tensorflow evalutation function
        ys_eval = ys + ys_I[0,0]
        # ys_eval = ys
        ys_pred = ys_pred - ys_I[0,0]

        # Evaluate the model on the test data using `evaluate`
        print(f'\n{path}')
        results[i,:] = model.evaluate(xs, [ys_eval, dys], verbose=0)

        if showplots:
            # plot right Chauchy-Green tensor
            Cs = tf.einsum('ikj,ikl->ijl',xs,xs)
            # pl.plot_right_cauchy_green_tensor(ld.reshape_C(Cs), titles[i], fnames[i])
            pl.plot_right_cauchy_green_tensor(ld.reshape_C(Cs), title=path, fname=None)
            # plot potential
            pl.plot_potential_prediction(ys, ys_pred, title=path, fname=None)

            # plot stress tensor
            pl.plot_stress_tensor_prediction(P, P_pred, title=path, fname=None)

    return results

# evaluate training data
print('\nEvaluate on training data:')
paths = ld.generate_concentric_paths(ftrain)
results = evaluate(paths)
df_train = pd.DataFrame(results, columns=['loss', 'W loss', 'P loss'])
df_train['file'] = ftrain
df_train = df_train[['file', 'loss', 'W loss', 'P loss']]
df_train.to_csv('out/train.csv', index=False)

# evalutate test data
print('\nEvalutate on test data:')
paths = ld.generate_concentric_paths(ftest)
results = evaluate(paths)
df_test = pd.DataFrame(results, columns=['loss', 'W loss', 'P loss'])
df_test['file'] = ftest
df_test = df_test[['file', 'loss', 'W loss', 'P loss']]
df_test.to_csv('out/test.csv', index=False)


print(df_train.mean()[1:])
print(df_test.mean()[1:])

# %%
"""
Model parameters

"""

def print_model_parameters():
    ''' Displays model parameters '''
    model.summary()
    for idx, layer in enumerate(model.layers):
        print(layer.name, layer)
        #print(layer.weights, "\n")
        print(layer.get_weights())
        
#print_model_parameters()
# %%
