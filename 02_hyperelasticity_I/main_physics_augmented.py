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
import tensorflow as tf
import numpy as np
import datetime
import pandas as pd
now = datetime.datetime.now
tf.keras.backend.set_floatx('float64')

# %% Own modules
import modules.data as ld
import modules.training as training

# %% Training

paths = [
    'data/calibration/biaxial.txt',
    'data/calibration/pure_shear.txt',
    'data/calibration/uniaxial.txt',
    'data/test/biax_test.txt',
    'data/test/mixed_test.txt'
    ]

#Alternative: concentric data
FNUM = 100
fnum = tf.random.shuffle(np.arange(1,FNUM + 1))
paths = ld.generate_concentric_paths(fnum)

lw = [1, 1]
loss_weighting=True

tmodel = training.PhysicsAugmented(paths=paths[:10],
                                loss_weights=lw,
                                loss_weighting=loss_weighting)

tmodel.calibrate(epochs=5000, verbose=2)

# %% Evalutation of normalization criterion

ys_I, dys_I = tmodel.evaluate_normalization()
print(f'\nW(I) =\t{ys_I[0,0]}')
print(f'P(I) =\t{dys_I[0,0]}\n\t{dys_I[0,1]}\n\t{dys_I[0,2]}')

# %% Loss evalutation

results = tmodel.evaluate(paths[10:], showplots=False)
loss = pd.DataFrame(results, columns=['total', 'W', 'P'])
loss['total'] = loss['W'] + loss['P'] # in case some loss weights != 0
loss['paths'] = paths[10:]
loss

# %% Model parameters

training.print_model_parameters(tmodel.model)