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
import data as ld
import training

# %%
'''
Training

'''


FNUM = 100
#test_train_split = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99])
test_train_split = np.array([0.1,0.2])
JMAX = 2
t_type = 'PhysicsAugmented' # 'Naive'

loss_weights = [0, 1] # only used for physics augmented training
epochs = 300
verbose = 0

# create training and test split
fnum = tf.random.shuffle(np.arange(1,FNUM + 1))
ftrain, ftest = tf.split(fnum, [int(split * (FNUM + 1)), -1])

# load test and train data paths
train_paths = ld.generate_concentric_paths(ftrain)
test_paths = ld.generate_concentric_paths(ftest)

# initialization
# mean_train_losses = np.zeros([test_train_split.size, 3])
# mean_test_losses = np.zeros([test_train_split.size, 3])

# median_train_losses = np.zeros([test_train_split.size, 3])
# median_test_losses = np.zeros([test_train_split.size, 3])

# max_train_losses = np.zeros([test_train_split.size, 3])
# max_test_losses = np.zeros([test_train_split.size, 3])

# min_train_losses = np.zeros([test_train_split.size, 3])
# min_test_losses = np.zeros([test_train_split.size, 3])

#tmp = pd.DataFrame(np.zeros([test_train_split.size, 3]),
#            columns=['total', 'function', 'gradient'])
#df_train = pd.concat([tmp, tmp, tmp, tmp], keys=('mean', 'median', 'max', 'min'))
#df_test = pd.concat([tmp, tmp, tmp, tmp], keys=('mean', 'median', 'max', 'min'))

train_losses = np.zeros([test_train_split.size, JMAX, len(train_paths), 3])
test_losses = np.zeros([test_train_split.size, JMAX, len(test_paths), 3])

t1 = now()
for i, split in enumerate(test_train_split):
    # mean_train_losses_tmp = np.zeros([JMAX,3])
    # mean_test_losses_tmp = np.zeros([JMAX,3])

    # median_train_losses_tmp = np.zeros([JMAX,3])
    # median_test_losses_tmp = np.zeros([JMAX,3])

    # max_train_losses_tmp = np.zeros([JMAX,3])
    # max_test_losses_tmp = np.zeros([JMAX,3])

    # min_train_losses_tmp = np.zeros([JMAX,3])
    # min_test_losses_tmp = np.zeros([JMAX,3])
    #train_losses_tmp = np.zeros([4, JMAX, 3])
    #test_losses_tmp = np.zeros([4, JMAX, 3])

    for j in range(JMAX):
        print(f'Model {i * JMAX + (j + 1)}/{test_train_split.size * JMAX}')

        if t_type == 'PhysicsAugmented':
            tmodel = training.PhysicsAugmented(paths=train_paths, loss_weights=loss_weights,
                        epochs=epochs,
                        verbose=verbose)
        else:
            tmodel = training.Naive(paths=train_paths, epochs=epochs, verbose=verbose)

        tmodel.preprocess()
        tmodel.calibrate()

        train_losses[i, j, :, :] = tmodel.evaluate(train_paths)
        test_losses[i, j, :, :] = tmodel.evaluate(test_paths)

        # evaluate
        # mean_train_losses_tmp[j,:] = np.mean(tmodel.evaluate(train_paths), axis=0)
        # mean_test_losses_tmp[j,:] = np.mean(tmodel.evaluate(test_paths), axis=0)

        # median_train_losses_tmp[j,:] = np.median(tmodel.evaluate(train_paths), axis=0)
        # median_test_losses_tmp[j,:] = np.median(tmodel.evaluate(test_paths), axis=0)

        # max_train_losses_tmp[j,:] = np.max(tmodel.evaluate(train_paths), axis=0)
        # max_test_losses_tmp[j,:] = np.max(tmodel.evaluate(test_paths), axis=0)

        # min_train_losses_tmp[j,:] = np.min(tmodel.evaluate(train_paths), axis=0)
        # min_test_losses_tmp[j,:] = np.min(tmodel.evaluate(test_paths), axis=0)

        # train_losses_tmp[0,j,:] = np.mean(tmodel.evaluate(train_paths), axis=0)
        # train_losses_tmp[1,j,:] = np.median(tmodel.evaluate(train_paths), axis=0)
        # train_losses_tmp[2,j,:] = np.max(tmodel.evaluate(train_paths), axis=0)
        # train_losses_tmp[3,j,:] = np.min(tmodel.evaluate(train_paths), axis=0)

        # test_losses_tmp[0,j,:] = np.mean(tmodel.evaluate(test_paths), axis=0)
        # test_losses_tmp[1,j,:] = np.median(tmodel.evaluate(test_paths), axis=0)
        # test_losses_tmp[2,j,:] = np.max(tmodel.evaluate(test_paths), axis=0)
        # test_losses_tmp[3,j,:] = np.min(tmodel.evaluate(test_paths), axis=0)

    # mean_train_losses[i,:] = np.mean(mean_train_losses_tmp, axis=0)
    # mean_test_losses[i,:] = np.mean(mean_test_losses_tmp, axis=0)

    # median_train_losses[i,:] = np.median(median_train_losses_tmp, axis=0)
    # median_test_losses[i,:] = np.median(median_test_losses_tmp, axis=0)

    # max_train_losses[i,:] = np.max(max_train_losses_tmp, axis=0)
    # max_test_losses[i,:] = np.max(max_test_losses_tmp, axis=0)

    # min_train_losses[i,:] = np.min(min_train_losses_tmp, axis=0)
    # min_test_losses[i,:] = np.min(min_test_losses_tmp, axis=0)

# Depending on the type of training the total loss might only contain the value
# of either training on the gradient or the functional value, e.g. if loss_weights
# is set to [0, 1]. To make sure that the total loss is always equal to the sum of
# all output losses it is necessary to compute the sum.
train_losses[:,:,:,0] = np.sum(train_losses[:,:,:,1:3], axis=3)
test_losses[:,:,:,0] = np.sum(test_losses[:,:,:,1:3], axis=3)

mean_train_losses = np.mean(train_losses, axis=(1,2))
median_train_losses = np.median(train_losses, axis=(1,2))
min_train_losses = np.min(train_losses, axis=(1,2))
max_train_losses = np.max(train_losses, axis=(1,2))

mean_test_losses = np.mean(test_losses, axis=(1,2))
median_test_losses = np.median(test_losses, axis=(1,2))
min_test_losses = np.min(test_losses, axis=(1,2))
max_test_losses = np.max(test_losses, axis=(1,2))



t2 = now()
print('it took', t2 - t1, '(sec) to calibrate and evaluate all models')

#%%
'''
Plots

'''

# training
fig = plt.figure(dpi=600)
if t_type == 'PhysicsAugmented':
    plt.semilogy(test_train_split,mean_train_losses[:,0], color='firebrick', marker='o', label='mean')
    plt.semilogy(test_train_split,median_train_losses[:,0], color='firebrick', marker='s', label='median')
    plt.fill_between(test_train_split, max_train_losses[:,0], min_train_losses[:,0], color='firebrick', alpha=0.5)
    
    plt.semilogy(test_train_split,mean_train_losses[:,1], color='navy', marker='o', label='W loss')
    plt.semilogy(test_train_split,median_train_losses[:,1], color='firebrick', marker='s', label='median loss')
    plt.fill_between(test_train_split, max_train_losses[:,1], min_train_losses[:,1], color='navy', alpha=0.5)

    plt.semilogy(test_train_split,mean_train_losses[:,2], color='darkorange', marker='o', label='mean')
    plt.semilogy(test_train_split,median_train_losses[:,2], color='darkorange', marker='s', label='median')
    plt.fill_between(test_train_split, max_train_losses[:,2], min_train_losses[:,2], color='darkorange', alpha=0.5)

else:
    plt.semilogy(test_train_split, mean_train_losses[:,0], color='firebrick', marker='o', label='loss')
    plt.fill_between(test_train_split, max_train_losses[:,0], min_train_losses[:,0], color='firebrick', alpha=0.5)
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
    plt.fill_between(test_train_split, max_test_losses[:,0], min_test_losses[:,0], color='firebrick', alpha=0.5)
    
    plt.semilogy(test_train_split,mean_test_losses[:,1], color='navy', marker='o', label='W loss')
    plt.semilogy(test_train_split,median_test_losses[:,1], color='firebrick', marker='s', label='median loss')
    plt.fill_between(test_train_split, max_test_losses[:,1], min_test_losses[:,1], color='navy', alpha=0.5)

    plt.semilogy(test_train_split,mean_test_losses[:,2], color='darkorange', marker='o', label='mean')
    plt.semilogy(test_train_split,median_test_losses[:,2], color='darkorange', marker='s', label='median')
    plt.fill_between(test_train_split, max_test_losses[:,2], min_test_losses[:,2], color='darkorange', alpha=0.5)

else:
    plt.semilogy(test_train_split, mean_test_losses[:,0], color='firebrick', marker='o', label='loss')
    plt.fill_between(test_train_split, max_test_losses[:,0], min_train_losses[:,0], color='firebrick', alpha=0.5)

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
