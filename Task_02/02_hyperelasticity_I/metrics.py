#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:26:51 2022

@author: jasper
"""

# %%
'''
import modules

'''

import tensorflow as tf

# %%
'''
metrics

'''

def compute_metrics(x1, x2):
    mse = tf.keras.metrics.mean_squared_error(x1, x2)
    mae = tf.keras.metrics.mean_absolute_error(x1, x2)
    return mse, mae