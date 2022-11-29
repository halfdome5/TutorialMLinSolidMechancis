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
test_train_split = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99])
#test_train_split = np.array([0.1,0.2])
JMAX = 5

loss_weights = [0, 1] # only used for physics augmented training
epochs = 2500
verbose = 0

# initialization
mean_train_losses = np.zeros([test_train_split.size, 3])
mean_test_losses = np.zeros([test_train_split.size, 3])

t1 = now()
for i, split in enumerate(test_train_split):
    train_losses_tmp = np.zeros([JMAX,3])
    test_losses_tmp = np.zeros([JMAX,3])

    for j in range(JMAX): 
        print(f'Model {i * JMAX + (j + 1)}/{test_train_split.size * JMAX}')
        # create training and test split
        fnum = tf.random.shuffle(np.arange(1,FNUM + 1))
        ftrain, ftest = tf.split(fnum, [int(split * (FNUM + 1)), -1])

        # load test and train data
        train_paths = ld.generate_concentric_paths(ftrain)
        test_paths = ld.generate_concentric_paths(ftest)

        # tmodel = training.PhysicsAugmented(paths=train_paths, loss_weights=loss_weights,
        #             epochs=epochs,
        #             verbose=verbose)
        tmodel = training.Naive(paths=train_paths, epochs=epochs, verbose=verbose)

        tmodel.preprocess()
        tmodel.calibrate()

        # evaluate
        train_losses_tmp[j,:] = np.mean(tmodel.evaluate(train_paths), axis=0)
        test_losses_tmp[j,:] = np.mean(tmodel.evaluate(test_paths), axis=0)
    
    mean_train_losses[i,:] = np.mean(train_losses_tmp, axis=0)
    mean_test_losses[i,:] = np.mean(test_losses_tmp, axis=0)

t2 = now()
print('it took', t2 - t1, '(sec) to calibrate and evaluate all models')

#%%
'''
Plots

'''
fig = plt.figure(dpi=600)
plt.semilogy(test_train_split,mean_train_losses[:,1] + mean_train_losses[:,2], marker='o', label='loss')
plt.semilogy(test_train_split,mean_train_losses[:,1], marker='o', label='W loss')
plt.semilogy(test_train_split,mean_train_losses[:,2], marker='o', label='P loss')
plt.title('Evaluation on training data')
plt.xlabel('training fraction')
plt.ylabel('loss')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('images/training_loss.png', dpi=fig.dpi, bbox_inches='tight')

fig = plt.figure(dpi=600)
plt.semilogy(test_train_split,mean_test_losses[:,1] + mean_test_losses[:,2], marker='o', label='loss')
plt.semilogy(test_train_split,mean_test_losses[:,1], marker='o', label='W loss')
plt.semilogy(test_train_split,mean_test_losses[:,2], marker='o', label='P loss')
plt.title('Evalutation on test data')
plt.xlabel('training fraction')
plt.ylabel('loss')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('images/test_loss.png', dpi=fig.dpi, bbox_inches='tight')

plt.show()

# %%
"""
Save results to file

"""
df_train = pd.DataFrame(mean_train_losses, columns=['total', 'function', 'gradient'])
df_test = pd.DataFrame(mean_test_losses, columns=['total', 'function', 'gradient'])

df_train.to_csv('out/train_losses.csv', index=False)
df_test.to_csv('out/test_losses.csv', index=False)
# %%
"""
Model parameters

"""

training.print_model_parameters(tmodel.model)

# %%
