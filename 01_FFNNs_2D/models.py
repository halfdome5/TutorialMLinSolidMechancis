"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Jasper Schommartz, Toprak Kis
         
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
_x_to_y: custom trainable and non-trainable layer

"""

def makeLayer(r_type, **kwargs):
    cf = {
        'FeedForward': FeedForwardLayer,
        'InputConvex': InputConvexLayer
          }
    class_obj = cf.get(r_type, None)
    if class_obj:
        return class_obj(**kwargs)
    raise ValueError("Unknown class object")
    
    
class FeedForwardLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        self.ls = [layers.Dense(4, 'softplus')]
        self.ls += [layers.Dense(4, 'softplus')]
        self.ls += [layers.Dense(4, 'softplus')]
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
        self.ls = [layers.Dense(4, 'softplus')]
        self.ls += [layers.Dense(4, 'softplus', kernel_constraint=non_neg())]
        self.ls += [layers.Dense(4, 'softplus', kernel_constraint=non_neg())]
        # scalar-valued output function
        self.ls += [layers.Dense(1, kernel_constraint=non_neg())]
        
    def __call__(self, x):     
        #  create weights by calling on input
        for l in self.ls:
            x = l(x)
        return x
    
class f1(layers.Layer):
    def __init__(self):
        super().__init__()
        pass
    
    def __call__(self, x):
        x = x ** 2
        x = x[:,0] - x[:,1]
        return x
        
class f2(layers.Layer):
    def __init__(self):
        super().__init__()
        pass
    
    def __call__(self, x):
        x = x ** 2
        x = x[:,0] + 0.5 * x[:,1]
        return x


# %%   
"""
main: construction of the NN model

"""

def main(**kwargs):
    # define input shape
    xs = tf.keras.Input(shape=(2,))
    # define which (custom) layers the model uses
    ys = makeLayer(**kwargs)(xs)
    # connect input and output
    model = tf.keras.Model(inputs = [xs], outputs = [ys])
    # define optimizer and loss function
    model.compile('adam', 'mse')
    return model