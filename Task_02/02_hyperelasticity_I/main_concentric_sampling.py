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
import training

# %%
'''
Load calibration data
'''
# parameters
TEST_TRAIN_SPLIT = 0.8
FNUM = 100

# create training and test split
fnum = tf.random.shuffle(np.arange(1,FNUM + 1))
ftrain, ftest = tf.split(fnum, [int(TEST_TRAIN_SPLIT * (FNUM + 1)), -1])

train_paths = ld.generate_concentric_paths(ftrain)
test_paths = ld.generate_concentric_paths(ftest)

tmodel = training.PhysicsAugmented()
tmodel.load(train_paths)

#%%
'''
Pre-processing

'''
tmodel.preprocess()

#%%
'''
Model calibration

'''
tmodel.calibrate()

#%%
''' 
Evalutation

'''
# evaluate training data
print('\nEvaluate on training data:')
results = tmodel.evaluate(train_paths)
df_train = pd.DataFrame(results, columns=['loss', 'W loss', 'P loss'])
df_train['file'] = ftrain
df_train = df_train[['file', 'loss', 'W loss', 'P loss']]
df_train.to_csv('out/train.csv', index=False)

# evalutate test data
print('\nEvalutate on test data:')
results = tmodel.evaluate(test_paths)
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

tmodel.print_model_parameters()

# %%
