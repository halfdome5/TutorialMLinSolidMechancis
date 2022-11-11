"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Jasper Schommartz, Troprak Kis
         
11/2022
"""


# %%   
"""
Import modules

"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import non_neg
import datetime
now = datetime.datetime.now


# %%   
"""
_x_to_y: custom trainable layer

"""

def makeLayer(xs, r_type, **kwargs):
    cf = {
        'FeedForward': FeedForwardLayer,
        'InputConvex': InputConvexLayer
          }
    class_obj = cf.get(r_type, None)
    if class_obj:
        return class_obj(**kwargs)(xs)
    raise ValueError('Class object not found')
    
    
class FeedForwardLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        self.ls = [layers.Dense(16, 'softplus')]
        self.ls += [layers.Dense(16, 'softplus')]
        self.ls += [layers.Dense(16, 'softplus')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]
        
    def __call__(self, x):     
        #  create weights by calling on input
        for l in self.ls:
            x = l(x)
        return x
    
class InputConvexLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(16, 'softplus')]
        self.ls += [layers.Dense(16, 'softplus', kernel_constraint=non_neg())]
        self.ls += [layers.Dense(16, 'softplus', kernel_constraint=non_neg())]
        # scalar-valued output function
        self.ls += [layers.Dense(1, kernel_constraint=non_neg())]
        
    def __call__(self, x):     
        #  create weights by calling on input
        for l in self.ls:
            x = l(x)
        return x


# %%   
"""
main: construction of the NN model

"""

def main(**kwargs):
    # define input shape
    xs = tf.keras.Input(shape=[1])
    # define which (custom) layers the model uses
    ys = makeLayer(xs, **kwargs)
    # connect input and output
    model = tf.keras.Model(inputs = [xs], outputs = [ys])
    # define optimizer and loss function
    model.compile('adam', 'mse')
    return model