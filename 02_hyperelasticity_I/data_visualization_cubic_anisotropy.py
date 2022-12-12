#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:24:45 2022

@author: jasper
"""

#%%
'''
import modules

'''
import os

# user defined modules
import modules.data as ld
from modules.plots import plot_imported_data

# %%
'''
Evaluate imported F, P and W data

'''

# load deformation, stress and energy data
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
F, P, W = ld.read_txt(paths[0])

plot_imported_data(F, P, W)
