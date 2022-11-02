"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein
         
08/2022
"""


# %%   
"""
Import modules

"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import non_neg
import datetime
import numpy as np
now = datetime.datetime.now


# %%   
"""
_x_to_y: custom trainable and non-trainable layer

"""

# class _x_to_y(layers.Layer):
#     def __init__(self, model_type, **kwargs):
#         super(_x_to_y, self).__init__()
#         if model_type == 'FFNN':
#             self.__create_FFNN()
#         if model_type == 'ICNN':
#             self.__create_ICNN()
#         if model_type == 'f':
#             self.__create_f_NN(**kwargs)
        
#     def __create_FFNN(self):
#         # define hidden layers with activation functions
#         self.ls = [layers.Dense(4, 'softplus')]
#         self.ls += [layers.Dense(4, 'softplus')]
#         # scalar-valued output function
#         self.ls += [layers.Dense(1)]
        
#     def __create_ICNN(self):
#         # define hidden layers with activation functions
#         self.ls = [layers.Dense(4, 'softplus')]
#         self.ls += [layers.Dense(4, 'softplus', kernel_constraint=non_neg())]
#         # scalar-valued output function
#         self.ls += [layers.Dense(1, 'relu', kernel_constraint=non_neg())]
        
#     def __create_f_NN(self, weights, bias):
#         # define hidden layer with square function
#         self.ls = [layers.Lambda(lambda x: x **2)]
#         # scalar-valued output layer
#         l = layers.Dense(1, trainable=False)
#         l.build(input_shape=(2,))
#         l.set_weights([weights, bias])
#         self.ls += [l]
        
            
#     def __call__(self, x):     
        
#         for l in self.ls:
#             x = l(x)
#         return x

def makeLayer(r_type, **kwargs):
    cf = {
        'FastForward': FastForwardLayer,
        'InputConvex': InputConvexLayer,
        'InputConvexSobolev': InputConvexSobolevLayer,
        'f1': f1,
        'f2': f2,
        'f2Sobolev': f2Sobolev
          }
    class_obj = cf.get(r_type, None)
    if class_obj:
        return class_obj(**kwargs)
    raise ValueError
    
    
class FastForwardLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        self.ls = [layers.Dense(8, 'softplus')]
        self.ls += [layers.Dense(8, 'softplus')]
        self.ls += [layers.Dense(8, 'softplus')]
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
        self.ls = [layers.Dense(8, 'softplus')]
        self.ls += [layers.Dense(8, 'softplus', kernel_constraint=non_neg())]
        self.ls += [layers.Dense(8, 'softplus', kernel_constraint=non_neg())]
        # scalar-valued output function
        self.ls += [layers.Dense(1, kernel_constraint=non_neg())]
        
    def __call__(self, x):     
        #  create weights by calling on input
        for l in self.ls:
            x = l(x)
        return x
    
class InputConvexSobolevLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(8, 'softplus')]
        self.ls += [layers.Dense(8, 'softplus', kernel_constraint=non_neg())]
        self.ls += [layers.Dense(8, 'softplus', kernel_constraint=non_neg())]
        # scalar-valued output function
        self.ls += [layers.Dense(1, kernel_constraint=non_neg())]
    
    def call(self, x):
        #  create weights by calling on input
        def loop(x):
            for l in self.ls:
                x = l(x)
            return x
            
        with tf.GradientTape() as g:
            g.watch(x)
            dx = loop(x)
        return loop(x), dx
        
    
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
    
class f2Sobolev(layers.Layer):
    def __init__(self):
        super().__init__()
        pass
    
    def __call__(self, x):
        # compute function values
        y = x ** 2
        y = y[:,0] + 0.5 * y[:,1]
        
        # compute gradient
        dy = tf.stack([2 * x[:,0], x[:,1]], 1) 
        
        return y, dy


# %%   
"""
main: construction of the NN model

"""

def main(**kwargs):
    # define input shape
    xs = tf.keras.Input(shape=(2,))
    # define which (custom) layers the model uses
    ys, dys = makeLayer(**kwargs)(xs)
    # connect input and output
    model = tf.keras.Model(inputs = [xs], outputs = [ys, dys])
    # define optimizer and loss function
    model.compile('adam', 'mse')
    return model