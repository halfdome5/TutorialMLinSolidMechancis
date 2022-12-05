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
fnums = np.array([5, 27, 13, 49])
paths = ld.generate_concentric_paths(fnums)

lw = [1, 1]
loss_weighting=True

tmodel = training.PhysicsAugmented(paths=paths[:3],
                                loss_weights=lw,
                                loss_weighting=loss_weighting)

tmodel.calibrate(epochs=2500, verbose=2)

# %% Evalutation of normalization criterion

ys_I, dys_I = tmodel.evaluate_normalization()
print(f'\nW(I) =\t{ys_I[0,0]}')
print(f'P(I) =\t{dys_I[0,0]}\n\t{dys_I[0,1]}\n\t{dys_I[0,2]}')

# %% Loss evalutation

results = tmodel.evaluate(paths, showplots=True)
loss = pd.DataFrame(results, columns=['total', 'W', 'P'])
loss['total'] = loss['W'] + loss['P'] # in case some loss weights != 0
loss['paths'] = paths
loss

# %% Model parameters

training.print_model_parameters(tmodel.model)