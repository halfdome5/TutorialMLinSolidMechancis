"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 3: Hyperelasticity 2

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
    'data/03_hyperelasticity_II/soft_beam_lattice_metamaterials/data/BCC_uniaxial.txt',
    'data/03_hyperelasticity_II/soft_beam_lattice_metamaterials/data/BCC_biaxial.txt',
    'data/03_hyperelasticity_II/soft_beam_lattice_metamaterials/data/BCC_shear.txt',
    'data/03_hyperelasticity_II/soft_beam_lattice_metamaterials/data/BCC_volumetric.txt',
    'data/03_hyperelasticity_II/soft_beam_lattice_metamaterials/data/BCC_planar.txt',
    'data/03_hyperelasticity_II/soft_beam_lattice_metamaterials/data/BCC_test1.txt',
    'data/03_hyperelasticity_II/soft_beam_lattice_metamaterials/data/BCC_test2.txt',
    'data/03_hyperelasticity_II/soft_beam_lattice_metamaterials/data/BCC_test3.txt'
    ]

lw = [1, 1]
scaling = False # scale stresses to range [-1, 1]

tmodel = training.CubicAnisotropy(paths=paths[:4],
                                loss_weights=lw,
                                loss_weighting=True,
                                scaling=scaling)

tmodel.calibrate(epochs=5000, verbose=2)

# %% Evalutation of normalization criterion

ys_I, dys_I = tmodel.evaluate_normalization()
print(f'\nW(I) =\t{ys_I[0,0]}')
print(f'P(I) =\t{dys_I[0,0]}\n\t{dys_I[0,1]}\n\t{dys_I[0,2]}')

# %% Loss evalutation

results = tmodel.evaluate(paths[:], showplots=True)
loss = pd.DataFrame(results, columns=['total', 'W', 'P'])
loss['total'] = loss['W'] + loss['P'] # in case some loss weights != 0
loss['paths'] = paths[:]
loss

# %% Model parameters

training.print_model_parameters(tmodel.model)