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
"""
_x_to_y: custom trainable layers

"""

# factory function for custom layer creation
def makeLayer(r_type, **kwargs):
    cf = {
        'FeedForward': FeedForwardLayer,
        'InputConvex': InputConvexLayer,
          }
    class_obj = cf.get(r_type, None)
    if class_obj:
        return class_obj(**kwargs)
    raise ValueError('Unknown class type')
    

class PotentialLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        # define non-trainable layers
        self.lC = RightCauchyGreenLayer()
        self.lI = InvariantLayer()
        # define neural network
        self.lNN = FeedForwardLayer()
    
    def __call__(self, x):
        y = self.lC(x)
        y = self.lI(x, y)
        y = self.lNN(y)
        return y
    
# layer that computes the gradient of a custom layer
class SobolevLayer(layers.Layer):
    def __init__(self, l):
        super().__init__()
        self.l = l
    
    def call(self, x):        
        with tf.GradientTape() as g:
            g.watch(x)
            y = self.l(x)
        return g.gradient(y, x)
    
    
class FeedForwardLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        # define hidden layers with activation function
        self.ls = [layers.Dense(8, 'softplus')]
        self.ls += [layers.Dense(8, 'softplus')]
        self.ls += [layers.Dense(8, 'softplus')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]
        
    def call(self, x):     
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
        self.ls += [layers.Dense(9, kernel_constraint=non_neg())]
        
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
    def __init__(self):
        super().__init__()
        
    def __call__(self, F):
        return tf.einsum('ikj,ikl->ijl', F, F)

class InvariantLayer(layers.Layer):
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
        #Cof_C = tf.constant(np.array([I3i * C_inv[i,:,:] for i,I3i in enumerate(I3)]))
        #Cof_C = tf.tensordot(I3, C_inv, axis=0)
        Cof_C = C_inv
        try:
            for i, I3i in I3:
                Cof_C[i,:,:] = C_inv[i,:,:] * I3i
        except:
            pass
        #Cof_C = I3 * C_inv
        #Cof_C = tf.multiply(C_inv, I3)
        I5 = tf.linalg.trace(Cof_C @ G_ti)
        return tf.stack([I1, J, -J, I4, I5], axis=1)
    
# class InvariantLayer(layers.Layer):
#     def __init__(self):
#         super().__init__()
        
#     def __call__(self,x):
#         # transversely isotropic structural tensor
#         G_ti = np.array([[4, 0, 0],
#                       [0, 0.5, 0],
#                       [0, 0, 0.5]])
#         # transpose F and compute right Cauchy-Green tensor
#         C = tf.einsum('ikj,ikl->ijl',x,x)
#         # compute invariants
#         I1 = tf.linalg.trace(C)
#         J = tf.linalg.det(x)
#         I4 = tf.linalg.trace(C @ G_ti)
        
#         C_inv = tf.linalg.inv(C)
#         I3 = tf.linalg.det(C)
#         Cof_C = tf.constant(np.array([I3i * C_inv[i,:,:] for i,I3i in enumerate(I3)]))
#         I5 = tf.linalg.trace(Cof_C @ G_ti)
#         return I1, J, I4, I5
    
# %%   
"""
main: construction of the NN model

"""

def main(loss_weights, **kwargs):
    # define input shape
    xs = tf.keras.Input(shape=(3,3))
    # self.ls = RightCauchyGreenLayer()(xs)
    # self.ls = InvariantLayer()(xs, ys)
    # define which (custom) layers the model uses
    #l_nn = makeLayer(**kwargs)
    l_nn = PotentialLayer()
    ys = l_nn(xs)
    # create and build sobolev layer
    # The sobolev layer computes the gradient and takes a custom layer as
    # constructor input
    dys = SobolevLayer(l_nn)(xs)
    
    model = tf.keras.Model(inputs=[xs], outputs=[ys, dys])
    
    # define optimizer and loss function
    model.compile('adam', 'mse', loss_weights=loss_weights)
    return model