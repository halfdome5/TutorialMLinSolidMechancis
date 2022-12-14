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

# %% Own modules
import modules.data as ld
import modules.training as training

# %%
'''
Training

'''


FNUM = 100
#test_train_split = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99])
test_train_split = np.array([0.7,0.8, 0.9])
JMAX = 3
t_type = 'Naive' # 'Naive'

loss_weighting = True
loss_weights = [1, 1] # only used for physics augmented training
epochs = 300
verbose = 0

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

    for j in range(JMAX):
        print(f'Model {i * JMAX + (j + 1)}/{NUMSPLITS * JMAX}')

        if t_type == 'PhysicsAugmented':
            tmodel = training.PhysicsAugmented(paths=train_paths, loss_weights=loss_weights, loss_weighting=True)
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

#%%
'''
Plots

'''

alpha = 0.3

# training
fig = plt.figure(dpi=600)
if t_type == 'PhysicsAugmented':
    plt.semilogy(test_train_split,mean_train_losses[:,0], color='firebrick', marker='o', label='mean')
    plt.semilogy(test_train_split,median_train_losses[:,0], color='firebrick', marker='s', label='median')
    plt.fill_between(test_train_split, max_train_losses[:,0], min_train_losses[:,0], color='firebrick', alpha=alpha)
    
    plt.semilogy(test_train_split,mean_train_losses[:,1], color='navy', marker='o', label='W mean')
    plt.semilogy(test_train_split,median_train_losses[:,1], color='navy', marker='s', label='W median')
    plt.fill_between(test_train_split, max_train_losses[:,1], min_train_losses[:,1], color='navy', alpha=alpha)

    plt.semilogy(test_train_split,mean_train_losses[:,2], color='darkorange', marker='o', label='P mean')
    plt.semilogy(test_train_split,median_train_losses[:,2], color='darkorange', marker='s', label='P median')
    plt.fill_between(test_train_split, max_train_losses[:,2], min_train_losses[:,2], color='darkorange', alpha=alpha)

else:
    plt.semilogy(test_train_split, mean_train_losses[:,0], color='firebrick', marker='o', label='mean')
    plt.semilogy(test_train_split, median_train_losses[:,0], color='firebrick', marker='s', label='median')
    plt.fill_between(test_train_split, max_train_losses[:,0], min_train_losses[:,0], color='firebrick', alpha=alpha)
plt.title('Evaluation on training data')
plt.xlabel('training fraction')
plt.ylabel('loss')
plt.xlim(np.min(test_train_split), np.max(test_train_split))
plt.grid(True, which='both')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('images/training_loss.png', dpi=fig.dpi, bbox_inches='tight')


# test
fig = plt.figure(dpi=600)
if t_type == 'PhysicsAugmented':
    plt.semilogy(test_train_split,mean_test_losses[:,0], color='firebrick', marker='o', label='mean')
    plt.semilogy(test_train_split,median_test_losses[:,0], color='firebrick', marker='s', label='median')
    plt.fill_between(test_train_split, max_test_losses[:,0], min_test_losses[:,0], color='firebrick', alpha=alpha)
    
    plt.semilogy(test_train_split,mean_test_losses[:,1], color='navy', marker='o', label='W mean')
    plt.semilogy(test_train_split,median_test_losses[:,1], color='navy', marker='s', label='W median')
    plt.fill_between(test_train_split, max_test_losses[:,1], min_test_losses[:,1], color='navy', alpha=alpha)

    plt.semilogy(test_train_split,mean_test_losses[:,2], color='darkorange', marker='o', label='P mean')
    plt.semilogy(test_train_split,median_test_losses[:,2], color='darkorange', marker='s', label='P median')
    plt.fill_between(test_train_split, max_test_losses[:,2], min_test_losses[:,2], color='darkorange', alpha=alpha)

else:
    plt.semilogy(test_train_split, mean_test_losses[:,0], color='firebrick', marker='o', label='mean')
    plt.semilogy(test_train_split, median_test_losses[:,0], color='firebrick', marker='s', label='median')
    plt.fill_between(test_train_split, max_test_losses[:,0], min_test_losses[:,0], color='firebrick', alpha=alpha)

plt.title('Evaluation on training data')
plt.xlabel('training fraction')
plt.ylabel('loss')
plt.xlim(np.min(test_train_split), np.max(test_train_split))
plt.grid(True, which='both')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('images/training_loss.png', dpi=fig.dpi, bbox_inches='tight')
plt.savefig('images/test_loss.png', dpi=fig.dpi, bbox_inches='tight')

plt.show()

# %%
"""
Save results to file

"""
df_train = pd.concat([pd.DataFrame(mean_train_losses),
                    pd.DataFrame(median_train_losses),
                    pd.DataFrame(min_train_losses),
                    pd.DataFrame(max_train_losses)],
                    keys=('mean', 'median', 'min', 'max'))
df_train.columns = ['total', 'function', 'gradient']
df_train['split'] = np.tile(test_train_split, 4)
df_train = df_train[['split', 'total', 'function', 'gradient']]

df_test = pd.concat([pd.DataFrame(mean_test_losses),
                    pd.DataFrame(median_test_losses),
                    pd.DataFrame(min_test_losses),
                    pd.DataFrame(max_test_losses)],
                    keys=('mean', 'median', 'min', 'max'))
df_test.columns = ['total', 'function', 'gradient']
df_test['split'] = np.tile(test_train_split, 4)
df_test = df_train[['split', 'total', 'function', 'gradient']]

df_train.to_csv('out/train_losses.csv', index=False)
df_test.to_csv('out/test_losses.csv', index=False)

# %%
"""
Model parameters

"""

training.print_model_parameters(tmodel.model)

# %%
