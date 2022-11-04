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
_x_to_y: custom trainable layers

"""

    
def makeLayer(r_type, **kwargs):
    cf = {
        'FeedForward': FeedForwardLayer,
        'InputConvex': InputConvexLayer,
          }
    class_obj = cf.get(r_type, None)
    if class_obj:
        return class_obj(**kwargs)
    raise ValueError('Unknown class type')
    
    # if sobolev_training: 
    #       l = SobolevLayer(l)
    #return l
    
class SobolevLayer(layers.Layer):
    def __init__(self, l):
        super().__init__()
        self.l = l
    
    def call(self, x):
        if tf.keras.backend.is_keras_tensor(x):
            return x
        
        with tf.GradientTape() as g:
            g.watch(x)
            y = self.l(x)
        return g.gradient(y, x)

    
class FeedForwardLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        self.ls = [layers.Dense(4, 'softplus')]
        self.ls += [layers.Dense(4, 'softplus')]
        # self.ls += [layers.Dense(8, 'softplus')]
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
        self.ls = [layers.Dense(8, 'softplus')]
        self.ls += [layers.Dense(8, 'softplus', kernel_constraint=non_neg())]
        # self.ls += [layers.Dense(8, 'softplus', kernel_constraint=non_neg())]
        # scalar-valued output function
        self.ls += [layers.Dense(1, kernel_constraint=non_neg())]
        
    def call(self, x):    
        #  create weights by calling on input
        for l in self.ls:
            x = l(x)
        return x
    
#%%
"""
  
_x_to_y: custom non-trainable layers      

"""
    
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

def main(sobolev_training=False, **kwargs):
    # define input shape
    xs = tf.keras.Input(shape=(2,))
    # define which (custom) layers the model uses
    l_nn = makeLayer(**kwargs)
    ys = l_nn(xs)
    # apply sobolev training
    if sobolev_training: 
        dys = SobolevLayer(l_nn)(xs)
        # l_sobolev = SobolevLayer(l_nn)
        # dys = l_sobolev(xs)
        # dys = l_sobolev(xs)
        # dys = SobolevLayer(l_nn).build()(xs)
    # connect input and output
    model = tf.keras.Model(inputs = [xs], outputs = [ys, dys])
    # define optimizer and loss function
    model.compile('adam', 'mse')
    return model