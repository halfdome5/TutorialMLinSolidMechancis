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
from scipy.spatial.transform import Rotation as R
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
scaling = False

tmodel = training.DefGradBased(paths=paths[:4],
                                loss_weights=lw,
                                loss_weighting=True,
                                scaling=True)

tmodel.calibrate(epochs=1000, verbose=2)

# %% Evalutation of normalization criterion

ys_I, dys_I = tmodel.evaluate_normalization()
print(f'\nW(I) =\t{ys_I[0,0]}')
print(f'P(I) =\t{dys_I[0,0]}\n\t{dys_I[0,1]}\n\t{dys_I[0,2]}')

# %% Loss evalutation
results = tmodel.evaluate(paths[:], 
                        qobj=R.identity().as_matrix(),
                        qmat=R.identity().as_matrix(),
                        showplots=True)
loss = pd.DataFrame(results, columns=['total', 'W', 'P'])
loss['total'] = loss['W'] + loss['P'] # in case some loss weights != 0
loss['paths'] = paths[:]
loss


# %% Evaluate objectivity
robjs = R.identity()
robjs = R.concatenate([robjs, R.random(9)])

# for Qobj in robjs.as_matrix(): # iteration over material symmetry group
#     #print(Qobj)
#     results = tmodel.evaluate(paths[4:],  
#                             qobj=Qobj,
#                             qmat=R.identity().as_matrix(),
#                             showplots=False)
#     print(results)

tmodel.evaluate_objectivity(paths, robjs)

#def plot_observers()

# %% Evaluate material symmetry

# create material symmetry group
rmats = R.identity()
rmats = R.concatenate([rmats, R.from_euler('x', [np.pi/2, np.pi, 3*np.pi/2])])
rmats = R.concatenate([rmats, R.from_euler('y', [np.pi/2, np.pi, 3*np.pi/2])])
rmats = R.concatenate([rmats, R.from_euler('z', [np.pi/2, np.pi, 3*np.pi/2])])
rmats = R.concatenate([rmats, R.from_rotvec( np.pi/np.sqrt(2) * np.array([[1, 1, 0],
                                                                    [-1, 1, 0],
                                                                    [1, 0, 1],
                                                                    [-1, 0, 1],
                                                                    [0, 1, 1],
                                                                    [0, -1, 1]]))])
rmats = R.concatenate([rmats, R.from_rotvec( 2/3 * np.pi/np.sqrt(3) * np.array([[1, 1, 1],
                                                                        [-1, 1, 1],
                                                                        [1, -1, 1],
                                                                        [-1, -1, 1]]))])
rmats = R.concatenate([rmats, R.from_rotvec( 4/3 * np.pi/np.sqrt(3) * np.array([[1, 1, 1],
                                                                        [-1, 1, 1],
                                                                        [1, -1, 1],
                                                                        [-1, -1, 1]]))])

for Qmat in rmats.as_matrix(): # iteration over material symmetry group
    #print(Qmat)
    results = tmodel.evaluate(paths[7:8],  
                            qobj=R.identity().as_matrix(),
                            qmat=Qmat,
                            showplots=False)
    #print(results)



# %% Model parameters

training.print_model_parameters(tmodel.model)