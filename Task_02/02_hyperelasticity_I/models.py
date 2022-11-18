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
        self.ls = [layers.Dense(4, 'softplus')]
        self.ls += [layers.Dense(4, 'softplus')]
        self.ls += [layers.Dense(4, 'softplus')]
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
        self.ls += [layers.Dense(1, kernel_constraint=non_neg())]
        
    def call(self, x):    
        #  create weights by calling on input
        for l in self.ls:
            x = l(x)
        return x
    
# %%   
"""
main: construction of the NN model

"""

def main(loss_weights, **kwargs):
    # define input shape
    xs = tf.keras.Input(shape=(2,))
    # define which (custom) layers the model uses
    l_custom = makeLayer(**kwargs)
    ys = l_custom(xs)
    # create and build sobolev layer
    # The sobolev layer computes the gradient and takes a custom layer as
    # constructor input
    dys = SobolevLayer(l_custom)(xs)
    
    model = tf.keras.Model(inputs=[xs], outputs=[ys, dys])
    
    # define optimizer and loss function
    model.compile('adam', 'mse', loss_weights=loss_weights)
    return model