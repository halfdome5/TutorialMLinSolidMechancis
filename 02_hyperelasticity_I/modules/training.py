# %% [Load modules]
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import datetime
now = datetime.datetime.now

# user-defined models
import modules.models as lm
import modules.plots as pl
import modules.data as ld

# %% class definition

# Training class structure
# class Training:
#     def __init__(self, paths **kwargs):
#         ''' Load model and calibration data '''
#         self.paths = paths
#         self.xs, self.ys, self.dys = self.__load(path)
#         self.model = ...

#     def __load(self, paths):
#         ...
#         return xs, ys, dys, batch_sizes

#     def preprocess(self):
#         pass

#     def calibrate(self):
#         """ Fit model to calibration data """
#
#     def evaluate(self, test_paths):
#         """ Evaluate model prediction on test reference cases """

class CubicAnisotropy:
    ''' Class for training a physics augmented neutral network
    calling order: initialize --> preprocess --> calibrate --> evaluate '''
    def __init__(self, paths, loss_weights, loss_weighting):
        # initialize variables
        self.loss_weights = loss_weights
        self.loss_weighting = loss_weighting

        # load calibration data
        self.xs, self.ys, self.dys, self.batch_sizes = self.__load(paths)
        
        # preform pre-preocessing
        self.sample_weights = np.ones(np.sum(self.batch_sizes))
        self.__preprocess()

        # create model
        self.model = lm.main(r_type='TransverseIsotropy', loss_weights=self.loss_weights)

    def __load(self, paths):
        ''' Load data for training or testing from a path '''
        xs, _, _, batch_sizes = ld.load_stress_strain_data(paths)

        # compute potential and stress using analytical model
        ys = ld.W(xs)
        dys = ld.P(ld.W)(xs)
        return xs, ys, dys, batch_sizes

    def __preprocess(self):
        ''' Preforms necessary pre-preocessing steps before model calibration '''
        # apply load weighting strategy
        if self.loss_weighting:
            self.sample_weights = ld.get_sample_weights(self.xs, self.batch_sizes)
        # reshape inputs
        self.ys = tf.reshape(self.ys,-1)

    def calibrate(self, epochs, verbose=2):
        ''' Preform model training '''
        t1 = now()
        print(t1)

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, 0.002)
        h = self.model.fit([self.xs], [self.ys, self.dys],
                    epochs=epochs,
                    verbose=verbose,
                    sample_weight=self.sample_weights)

        t2 = now()
        print('it took', t2 - t1, '(sec) to calibrate the model')
        pl.plot_calibration_loss(h)

    def evaluate_normalization(self):
        ''' Calls evaluate_normalization static function '''
        return evaluate_normalization(self.model)

    def evaluate(self, paths, **kwargs):
        ''' Perform evaluation '''
        # initialization
        losses = np.zeros([len(paths), 3])

        # iteration over load paths
        for i, path in enumerate(paths):
            losses[i,:] = self.__evaluate_single_load_path(path, **kwargs)
        return losses

    def __evaluate_single_load_path(self, path, showplots=False):
        ''' Perform evaluation of a single load path '''
        xs, ys, dys, batch_size = self.__load([path])

        # predict using the trained model
        ys_pred, dys_pred = self.model.predict(xs)

        # Potential correction - for P training
        # shift reference value ys by normalization offset
        # to ensure reasonable results from tensorflow evalutation function
        if self.loss_weights == [0, 1]:
            ys_I, _ = self.evaluate_normalization()
            # normalization
            ys_eval = ys + ys_I[0,0]
            ys_pred = ys_pred - ys_I[0,0]
        else:
            # no normalization
            ys_eval = ys

        if self.loss_weighting:
            sample_weights = ld.get_sample_weights(xs, batch_size)
        else:
            sample_weights = np.ones(np.sum(batch_size))

        loss = self.model.evaluate(xs, [ys_eval, dys], 
                                    verbose=0,
                                    sample_weight=sample_weights)

        if showplots:
            # reshape right Cauchy-Green-Tensor
            #Cs = tf.einsum('ikj,ikl->ijl',xs,xs)
            cs = lm.IndependentValues()(layers.Flatten()(lm.RightCauchyGreenTensor()(xs)))
            # plots
            fname = path.split('/')[-1].split('.')[0]
            pl.plot_right_cauchy_green_tensor(cs, title=path, fname=fname)
            pl.plot_potential_prediction(ys, ys_pred, title=path, fname=fname)
            pl.plot_stress_tensor_prediction(dys, dys_pred, title=path, fname=fname)

        return loss

class TransverseIsotropy:
    ''' Class for training a physics augmented neutral network
    calling order: initialize --> preprocess --> calibrate --> evaluate '''
    def __init__(self, paths, loss_weights, loss_weighting):
        # initialize variables
        self.loss_weights = loss_weights
        self.loss_weighting = loss_weighting

        # load calibration data
        self.xs, self.ys, self.dys, self.batch_sizes = self.__load(paths)
        
        # preform pre-preocessing
        self.sample_weights = np.ones(np.sum(self.batch_sizes))
        self.__preprocess()

        # create model
        self.model = lm.main(r_type='TransverseIsotropy', loss_weights=self.loss_weights)

    def __load(self, paths):
        ''' Load data for training or testing from a path '''
        xs, _, _, batch_sizes = ld.load_stress_strain_data(paths)

        # compute potential and stress using analytical model
        ys = ld.W(xs)
        dys = ld.P(ld.W)(xs)
        return xs, ys, dys, batch_sizes

    def __preprocess(self):
        ''' Preforms necessary pre-preocessing steps before model calibration '''
        # apply load weighting strategy
        if self.loss_weighting:
            self.sample_weights = ld.get_sample_weights(self.xs, self.batch_sizes)
        # reshape inputs
        self.ys = tf.reshape(self.ys,-1)

    def calibrate(self, epochs, verbose=2):
        ''' Preform model training '''
        t1 = now()
        print(t1)

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, 0.002)
        h = self.model.fit([self.xs], [self.ys, self.dys],
                    epochs=epochs,
                    verbose=verbose,
                    sample_weight=self.sample_weights)

        t2 = now()
        print('it took', t2 - t1, '(sec) to calibrate the model')
        pl.plot_calibration_loss(h)

    def evaluate_normalization(self):
        ''' Calls evaluate_normalization static function '''
        return evaluate_normalization(self.model)

    def evaluate(self, paths, **kwargs):
        ''' Perform evaluation '''
        # initialization
        losses = np.zeros([len(paths), 3])

        # iteration over load paths
        for i, path in enumerate(paths):
            losses[i,:] = self.__evaluate_single_load_path(path, **kwargs)
        return losses

    def __evaluate_single_load_path(self, path, showplots=False):
        ''' Perform evaluation of a single load path '''
        xs, ys, dys, batch_size = self.__load([path])

        # predict using the trained model
        ys_pred, dys_pred = self.model.predict(xs)

        # Potential correction - for P training
        # shift reference value ys by normalization offset
        # to ensure reasonable results from tensorflow evalutation function
        if self.loss_weights == [0, 1]:
            ys_I, _ = self.evaluate_normalization()
            # normalization
            ys_eval = ys + ys_I[0,0]
            ys_pred = ys_pred - ys_I[0,0]
        else:
            # no normalization
            ys_eval = ys

        if self.loss_weighting:
            sample_weights = ld.get_sample_weights(xs, batch_size)
        else:
            sample_weights = np.ones(np.sum(batch_size))

        loss = self.model.evaluate(xs, [ys_eval, dys], 
                                    verbose=0,
                                    sample_weight=sample_weights)

        if showplots:
            # reshape right Cauchy-Green-Tensor
            #Cs = tf.einsum('ikj,ikl->ijl',xs,xs)
            cs = lm.IndependentValues()(layers.Flatten()(lm.RightCauchyGreenTensor()(xs)))
            # plots
            fname = path.split('/')[-1].split('.')[0]
            pl.plot_right_cauchy_green_tensor(cs, title=path, fname=fname)
            pl.plot_potential_prediction(ys, ys_pred, title=path, fname=fname)
            pl.plot_stress_tensor_prediction(dys, dys_pred, title=path, fname=fname)

        return loss

class Naive:
    ''' Class for training a feed forward neutral network
    calling order: initialize --> preprocess --> calibrate --> evaluate '''
    def __init__(self, paths, loss_weighting):
        # initialzation
        self.loss_weights = [1, 0]
        self.loss_weighting = loss_weighting

        # load calibration data
        self.xs, self.ys, self.dys, self.batch_sizes = self.__load(paths)
        
        # preprocessing
        self.sample_weights = np.ones(np.sum(self.batch_sizes))
        self.__preprocess()

        # create model
        self.model = lm.main(r_type='Naive', loss_weights=self.loss_weights)

    def __load(self, paths):
        ''' Load data for training or testing from a path '''
        xs, _, _, batch_sizes = ld.load_stress_strain_data(paths)

        # compute potential and stress using analytical model
        ys = ld.P(ld.W)(xs)
        dys = np.zeros(xs.shape) # placeholder
        return xs, ys, dys, batch_sizes

    def __preprocess(self):
        ''' Preforms necessary pre-preocessing steps before model calibration '''
        # apply load weighting strategy
        if self.loss_weighting:
            self.sample_weights = ld.get_sample_weights(self.xs, self.batch_sizes)

    def calibrate(self, epochs, verbose=2):
        ''' Preform model training '''
        t1 = now()
        print(t1)

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, 0.002)
        h = self.model.fit([self.xs], [self.ys, self.dys],
                    epochs=epochs,
                    verbose=verbose,
                    sample_weight=self.sample_weights)

        t2 = now()
        print('it took', t2 - t1, '(sec) to calibrate the model')
        pl.plot_calibration_loss(h)

    def evaluate_normalization(self):
        ''' Calls evaluate_normalization static function '''
        return evaluate_normalization(self.model)

    def evaluate(self, paths, **kwargs):
        ''' Perform evaluation '''
        # initialization
        losses = np.zeros([len(paths), 3])

        # iteration over load paths
        for i, path in enumerate(paths):
            losses[i,:] = self.__evaluate_single_load_path(path, **kwargs)
        return losses

    def __evaluate_single_load_path(self, path, showplots=False):
        xs, ys, dys, batch_size = self.__load([path])

        # predict using the trained model
        ys_pred, _ = self.model.predict(xs)
        if self.loss_weighting:
            sample_weights = ld.get_sample_weights(xs, batch_size)
        else:
            sample_weights = np.ones(np.sum(batch_size))
        
        loss = self.model.evaluate(xs, [ys, dys], 
                                verbose=0,
                                sample_weight=sample_weights)

        if showplots:
            # plot right Chauchy-Green tensor
            cs = lm.IndependentValues()(layers.Flatten()(lm.RightCauchyGreenTensor()(xs)))
            fname = path.split('/')[-1].split('.')[0]
            pl.plot_right_cauchy_green_tensor(cs, title=path, fname=fname)
            pl.plot_stress_tensor_prediction(ys, ys_pred, title=path, fname=fname)

        return loss


#%% Static function

def evaluate_normalization(model):
    ''' Returns model prediction for Deformation Gradient = Identity '''
    ys, dys = model.predict(np.array([np.identity(3)]))
    return ys, dys

def print_model_parameters(model):
    ''' Displays model parameters '''
    model.summary()
    for layer in model.layers:
        print(layer.name, layer)
        print(layer.get_weights())