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
dir_path = os.path.join('data', 'soft_beam_lattice_metamaterials', 'data')
path = os.path.join(dir_path, 'BCC_volumetric.txt')
F, P, W = ld.read_txt(path)

plot_imported_data(F, P, W)
