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
    'data/02_hyperelasticity_I/calibration/biaxial.txt',
    'data/02_hyperelasticity_I/calibration/pure_shear.txt',
    'data/02_hyperelasticity_I/calibration/uniaxial.txt',
    'data/02_hyperelasticity_I/test/biax_test.txt',
    'data/02_hyperelasticity_I/test/mixed_test.txt'
    ]

#Alternative: concentric data
FNUM = 100
fnum = tf.random.shuffle(np.arange(1,FNUM + 1))
paths = ld.generate_concentric_paths(fnum)

loss_weighting=True

tmodel = training.Naive(paths=paths[:20],
                        loss_weighting=loss_weighting)

tmodel.calibrate(epochs=10000, verbose=2)

# %% Evalutation of normalization criterion

# in the naive approach ys is already the stress, therefore dys is ignored
ys_I = tmodel.evaluate_normalization()[0]
print(f'P(I) =\t{ys_I[0, 0]}\n\t{ys_I[0, 1]}\n\t{ys_I[0, 2]}')

# %% Loss evalutation

# in the naive approach ys is already the stress, therefore dys is ignored
results = tmodel.evaluate(paths[:20], showplots=True)
loss = pd.DataFrame(results[:,0], columns=['total'])
loss['paths'] = paths[:20]
loss
# %% Model parameters

training.print_model_parameters(tmodel.model)