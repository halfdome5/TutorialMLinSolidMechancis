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
import pandas as pd
import datetime
now = datetime.datetime.now
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.keras.backend.floatx())
tf.keras.backend.set_floatx('float64')

# %% Own modules
import modules.data as ld
import modules.training as training
import modules.plots as pl

# %%
'''
Parameter definition

'''
#

FNUM = 100
test_train_split = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
#test_train_split = np.array([0.7, 0.8, 0.9])
#test_train_split = np.array([0.8])
#test_train_split = np.array([0.1, 0.2])
JMAX = 5
t_type = 'Naive' # 'Naive'

loss_weighting = True
loss_weights = [1, 1] # only used for physics augmented training
epochs = 10000
verbose = 0

# Alternative:
# paths = [
#     'data/calibration/biaxial.txt',
#     'data/calibration/pure_shear.txt',
#     'data/calibration/uniaxial.txt',
#     'data/test/biax_test.txt',
#     'data/test/mixed_test.txt'
#     ]

#%%
'''
Training

'''

NUMSPLITS = test_train_split.size

# initialization
train_losses = np.empty([NUMSPLITS, JMAX], dtype=object)
test_losses = np.empty([NUMSPLITS, JMAX], dtype=object)

t1 = now()
for i, split in enumerate(test_train_split):
    # create training and test split
    fnum = tf.random.shuffle(np.arange(1,FNUM + 1))
    ftrain, ftest = tf.split(fnum, [int(split * (FNUM + 1)), -1])

    # load test and train data paths
    train_paths = ld.generate_concentric_paths(ftrain)
    test_paths = ld.generate_concentric_paths(ftest)
    # Alternaive paths
    # train_paths = paths[:3]
    # test_paths = paths[3:]

    for j in range(JMAX):
        print(f'Model {i * JMAX + (j + 1)}/{NUMSPLITS * JMAX}')

        if t_type == 'TransverseIsotropy':
            tmodel = training.TransverseIsotropy(paths=train_paths, loss_weights=loss_weights, loss_weighting=True)
        else:
            tmodel = training.Naive(paths=train_paths, loss_weighting=loss_weighting)

        tmodel.calibrate(epochs=epochs, verbose=verbose)

        train_losses[i, j] = tmodel.evaluate(train_paths)
        test_losses[i, j] = tmodel.evaluate(test_paths)

        #print(f'\nW(I) =\t{ys[0,0]}')
        #print(f'P(I) =\t{dys[0,0]}\n\t{dys[0,1]}\n\t{dys_I[0,2]}')

        # Depending on the type of training the total loss might only contain the value
        # of either training on the gradient or the functional value, e.g. if loss_weights
        # is set to [0, 1]. To make sure that the total loss is always equal to the sum of
        # all output losses it is necessary to compute the sum.
        train_losses[i, j][:,0] = np.sum(train_losses[i,j][:,1:3], axis=1)
        test_losses[i, j][:,0] = np.sum(test_losses[i,j][:,1:3], axis=1)

#%%
# concat all splits because we cant to compoute the average loss of all runs
tmp = np.array([np.concatenate(train_losses[i,:]) for i in range(NUMSPLITS)])
mean_train_losses = np.array([np.mean(tmp[i], axis=0) for i in range(NUMSPLITS)])
median_train_losses = np.array([np.median(tmp[i], axis=0) for i in range(NUMSPLITS)])
max_train_losses = np.array([np.max(tmp[i], axis=0) for i in range(NUMSPLITS)])
min_train_losses = np.array([np.min(tmp[i], axis=0) for i in range(NUMSPLITS)])

tmp = np.array([np.concatenate(test_losses[i,:]) for i in range(NUMSPLITS)])
mean_test_losses = np.array([np.mean(tmp[i], axis=0) for i in range(NUMSPLITS)])
median_test_losses = np.array([np.median(tmp[i], axis=0) for i in range(NUMSPLITS)])
max_test_losses = np.array([np.max(tmp[i], axis=0) for i in range(NUMSPLITS)])
min_test_losses = np.array([np.min(tmp[i], axis=0) for i in range(NUMSPLITS)])

t2 = now()
print('it took', t2 - t1, '(sec) to calibrate and evaluate all models')


# %%
"""
Save results to file

"""

def save_to_df(means, medians, mins, maxs):
    df = pd.concat([pd.DataFrame(means),
                    pd.DataFrame(medians),
                    pd.DataFrame(mins),
                    pd.DataFrame(maxs)],
                    keys=('mean', 'median', 'min', 'max'))
    df.columns = ['total', 'function', 'gradient']
    df['split'] = np.tile(test_train_split, 4)
    df = df[['split', 'total', 'function', 'gradient']]
    return df

df_train = save_to_df(mean_train_losses,
                        median_train_losses,
                        min_train_losses,
                        max_train_losses)

df_test = save_to_df(mean_test_losses,
                        median_test_losses,
                        min_test_losses,
                        max_test_losses)

df_train.to_csv('out/train_losses.csv', index=False)
df_test.to_csv('out/test_losses.csv', index=False)

# %%
'''
Plot losses from csv file

'''

df_train = pd.read_csv('out/train_losses.csv')
df_test = pd.read_csv('out/test_losses.csv')

if t_type  == 'TransverseIsotropy':
    title = 'Evaluation on training data'
    pl.plot_loss_over_train_split_physics_augmented(df_train, title=title, fname='train_loss')
    title = 'Evaluation on test data'
    pl.plot_loss_over_train_split_physics_augmented(df_test, title=title, fname='test_loss')
elif t_type == 'Naive':
    title = 'Evaluation on training data'
    pl.plot_loss_over_train_split_naive(df_train, title=title, fname='train_loss')
    title = 'Evaluation on test data'
    pl.plot_loss_over_train_split_naive(df_test, title=title, fname='test_loss')

# %%
"""
Model parameters

"""

training.print_model_parameters(tmodel.model)

# %%
