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
now = datetime.datetime.now


# %%   
"""
_x_to_y: custom trainable layer

"""

class _x_to_y(layers.Layer):
    def __init__(self, model_type, **kwargs):
        super(_x_to_y, self).__init__()
        if model_type == 'FFNN':
            self.__create_FFNN()
        if model_type == 'ICNN':
            self.__create_ICNN()
        if model_type == 'nontrainable':
            self.__create_non_trainable_NN(**kwargs)
        
    def __create_FFNN(self):
        # define hidden layers with activation functions
        self.ls = [layers.Dense(4, 'softplus')]
        self.ls += [layers.Dense(4, 'softplus')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]
        
    def __create_ICNN(self):
        # define hidden layers with activation functions
        self.ls = [layers.Dense(4, 'softplus')]
        self.ls += [layers.Dense(4, 'softplus', kernel_constraint=non_neg())]
        # scalar-valued output function
        self.ls += [layers.Dense(1, 'relu', kernel_constraint=non_neg())]
        
    def __create_non_trainable_NN(self, weights, bias):
        # scalar-valued output layer
        l = layers.Dense(1)
        l.build(input_shape=(2,))
        l.set_weights([weights, bias])
        l.trainable = False
        self.ls = [l]
        
            
    def __call__(self, x):     
        
        for l in self.ls:
            x = l(x)
        return x


# %%   
"""
main: construction of the NN model

"""

def main(**kwargs):
    # define input shape
    xs = tf.keras.Input(shape=(2,))
    # define which (custom) layers the model uses
    ys = _x_to_y(**kwargs)(xs)
    # connect input and output
    model = tf.keras.Model(inputs = [xs], outputs = [ys])
    # define optimizer and loss function
    model.compile('adam', 'mse')
    return model