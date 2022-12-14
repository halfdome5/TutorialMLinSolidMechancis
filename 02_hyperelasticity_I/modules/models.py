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
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import non_neg
import datetime
now = datetime.datetime.now


# %%
'''
factory method

'''

def make_layer(r_type, **kwargs):
    """ Calls and returns layer object """
    cf = {
        'Naive': NaiveLayer,
        'PhysicsAugmented': PhysicsAugmentedLayer
        }
    class_obj = cf.get(r_type, None)
    if class_obj:
        return class_obj(**kwargs)
    raise ValueError('Unknown class type')

# %%
'''
wrapper layers

'''

class NaiveLayer(layers.Layer):
    """ Wrapper layer for naive neural network model """
    def __init__(self):
        super().__init__()
        # define non-trainable layers
        self.ls = [RightCauchyGreenLayer()]
        self.ls += [layers.Flatten()]
        self.ls += [IndependentValuesLayer()]
        self.ls += [FeedForwardLayer()]
        self.ls += [layers.Reshape((3,3))]
  
    def __call__(self, x):
        for l in self.ls:
            x = l(x)
        return x

class PhysicsAugmentedLayer(layers.Layer):
    """ Wrapper layer invariante based physics augmented neural network """
    def __init__(self):
        super().__init__()
        # define non-trainable layers
        self.lC = RightCauchyGreenLayer()
        self.lI = InvariantLayer()
        # define neural network
        self.lNN = InputConvexLayer()

    def __call__(self, x):
        y = self.lC(x)
        y = self.lI(x, y)
        y = self.lNN(y)
        return y

# %%   
"""
_x_to_y: custom trainable layers

"""

class SobolevLayer(layers.Layer):
    ''' Layer that computes the gradient '''
    def __init__(self, l):
        super().__init__()
        self.l = l

    def call(self, x):        
        with tf.GradientTape() as g:
            g.watch(x)
            y = self.l(x)
        return g.gradient(y, x)

class FeedForwardLayer(layers.Layer):
    ''' Layer that implements a feed forward neural network '''
    def __init__(self):
        super().__init__()
        # define hidden layers with activation function
        self.ls = [layers.Dense(8, 'softplus')]
        self.ls += [layers.Dense(8, 'softplus')]
        self.ls += [layers.Dense(8, 'softplus')]
        # scalar-valued output function
        self.ls += [layers.Dense(9)]

    def call(self, x):     
        #  create weights by calling on input
        for l in self.ls:
            x = l(x)
        return x

class InputConvexLayer(layers.Layer):
    """ Layer that implements an input convex neural network """
    def __init__(self):
        super().__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(8, 'softplus', kernel_constraint=non_neg())]
        self.ls += [layers.Dense(8, 'softplus', kernel_constraint=non_neg())]
        self.ls += [layers.Dense(8, 'softplus', kernel_constraint=non_neg())]
        # scalar-valued output function
        self.ls += [layers.Dense(1, kernel_constraint=non_neg())]

    def call(self, x):
        #  create weights by calling on input
        for l in self.ls:
            x = l(x)
        return x

# %%
'''
custom non-trainable layers

'''

class RightCauchyGreenLayer(layers.Layer):
    ''' Layer that computes the right Cauchy-Green tensor '''
    def __init__(self):
        super().__init__()
 
    def __call__(self, F):
        return tf.einsum('ikj,ikl->ijl', F, F)


class InvariantLayer(layers.Layer):
    ''' Layer that computes four invariants of a given deformatioin gradient '''
    def __init__(self):
        super().__init__()

    def __call__(self, F, C):
        # transversely isotropic structural tensor
        G_ti = np.array([[4, 0, 0],
                      [0, 0.5, 0],
                      [0, 0, 0.5]])
        # compute invariants
        I1 = tf.linalg.trace(C)
        J = tf.linalg.det(F)
        I4 = tf.linalg.trace(C @ G_ti)

        C_inv = tf.linalg.inv(C)
        I3 = tf.linalg.det(C)
        # catch error if a KerasTensor is passed
        Cof_C = I3[:, tf.newaxis, tf.newaxis] * C_inv
        I5 = tf.linalg.trace(Cof_C @ G_ti)
        return tf.stack([I1, J, -J, I4, I5], axis=1)
    

class IndependentValuesLayer(layers.Layer):
    ''' Layer that extracts six independent values of the right Cauchy green tensor '''
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        split1, _, split2, _,split3 = tf.split(x, [3, 1, 2, 2, 1], axis=1)
        return tf.concat([split1, split2, split3], 1)

    
# %%   
"""
main: construction of the NN model

"""

def main(loss_weights, **kwargs):
    """ This creates a Keras model """
    # define input shape
    xs = tf.keras.Input(shape=(3,3))
    # define which (custom) layers the model uses
    l_nn = make_layer(**kwargs)
    ys = l_nn(xs)
    # create and build sobolev layer
    dys = SobolevLayer(l_nn)(xs)

    model = tf.keras.Model(inputs=[xs], outputs=[ys, dys])
    # define optimizer and loss function
    model.compile('adam', 'mse', loss_weights=loss_weights)
    return model