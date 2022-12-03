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
import datetime
import pandas as pd
now = datetime.datetime.now

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
# fnums = np.array([5, 27, 13, 49])
# paths = ld.generate_concentric_paths(fnums)

loss_weighting=True

tmodel = training.Naive(paths=paths[:3],
                        loss_weighting=loss_weighting)

tmodel.calibrate(epochs=2500, verbose=2)

# %% Evalutation of normalization criterion

# in the naive approach ys is already the stress, therefore dys is ignored
ys_I = tmodel.evaluate_normalization()[0]
print(f'P(I) =\t{ys_I[0, 0]}\n\t{ys_I[0, 1]}\n\t{ys_I[0, 2]}')

# %% Loss evalutation

# in the naive approach ys is already the stress, therefore dys is ignored
results = tmodel.evaluate(paths, showplots=True)
loss = pd.DataFrame(results[:,0], columns=['total'])
loss['paths'] = paths
loss
# %% Model parameters

training.print_model_parameters(tmodel.model)