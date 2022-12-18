# %% [Load modules]
import tensorflow as tf
from tensorflow.keras import layers
from scipy.spatial.transform import Rotation as R
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

class DefGradBased:
    ''' Class for training a physics augmented neutral network
    calling order: initialize --> preprocess --> calibrate --> evaluate '''
    def __init__(self, paths, loss_weights, loss_weighting, scaling, rmats, robjs):
        # initialize variables
        self.loss_weights = loss_weights
        self.loss_weighting = loss_weighting
        self.scaling = scaling
        self.rmats = rmats
        self.robjs = robjs
        self.paths = paths # train paths
        self.scalefactor = 1

        # create model
        self.model = lm.main(r_type='DefGradBased', loss_weights=self.loss_weights)

        # load calibration data
        self.xs, self.ys, self.dys, self.batch_sizes = self.__load(self.paths)
        
        # perform pre-preocessing
        self.sample_weights = np.ones(np.sum(self.batch_sizes))
        self.__preprocess()

    def __load(self, paths):
        ''' Load data for training or testing from a path '''
        xs, _, _, batch_sizes = ld.load_stress_strain_data(paths)

        # compute potential and stress using analytical model
        ys = ld.W(xs)
        dys = ld.P(ld.W)(xs)
        return xs, ys, dys, batch_sizes

    def __preprocess(self):
        ''' Preforms necessary pre-preocessing steps before model calibration '''
        self.__scale_calibration_data()
        self.__apply_loss_weighting(self.xs, self.batch_sizes)
        self.ys = tf.reshape(self.ys,-1)

    def __scale_calibration_data(self):
        if self.scaling: self.scalefactor = tf.math.reduce_max(tf.math.abs(self.dys)) ** (-1)
        self.ys = self.ys * self.scalefactor
        self.dys = self.dys * self.scalefactor

    def __apply_loss_weighting(self, xs, batch_sizes):
        if self.loss_weighting:
            self.sample_weights = ld.get_sample_weights(xs, batch_sizes)
        else:
            self.sample_weights = np.ones(batch_sizes.sum())


    def augment_data(self, a_type):
        ''' Performs preprocessing for a second calibration by applying data augmentation '''
        # Data is reloaded to make it easier to recalibrate the model
        # load calibration data
        self.xs, self.ys, self.dys, self.batch_sizes = self.__load(self.paths)
        
        # perform pre-preocessing
        self.sample_weights = np.ones(np.sum(self.batch_sizes))
        self.__preprocess()
        
        if a_type == 'obj':
            n_rots = self.robjs.__len__()
        elif a_type == 'mat':
            n_rots = self.rmats.__len__()
        elif a_type == 'successive':
            n_rots = self.robjs.__len__() + self.rmats.__len__()
        elif a_type == 'concurrent':
            n_rots = self.robjs.__len__() * self.rmats.__len__()
        else:
            raise ValueError(f'a_type "{a_type}" is unknown')

        # initialize augmented dataset
        bs = self.batch_sizes.sum()
        xs = np.zeros([bs * n_rots, 3, 3])
        ys = np.zeros([bs * n_rots])
        dys = np.zeros([bs * n_rots, 3, 3])

        if a_type == 'obj':
            for i, Qobj in enumerate(self.robjs.as_matrix()):
                xs[i*bs:(i+1)*bs,:,:] = Qobj @ self.xs
                ys[i*bs:(i+1)*bs] = self.ys
                dys[i*bs:(i+1)*bs,:,:] = Qobj @ self.dys
            print(f'Dataset augmented by factor: {i + 1}')
            
        elif a_type == 'mat':
            for j, Qmat in enumerate(self.rmats.as_matrix()):
                xs[j*bs:(j+1)*bs,:,:] = self.xs @ Qmat
                ys[j*bs:(j+1)*bs] = self.ys
                dys[j*bs:(j+1)*bs,:,:] = self.dys @ Qmat
            print(f'Dataset augmented by factor: {j + 1}')

        elif a_type == 'successive':
            for i, Qobj in enumerate(self.robjs.as_matrix()):
                xs[i*bs:(i+1)*bs,:,:] = Qobj @ self.xs
                ys[i*bs:(i+1)*bs] = self.ys
                dys[i*bs:(i+1)*bs,:,:] = Qobj @ self.dys
            for j, Qmat in enumerate(self.rmats.as_matrix()):
                xs[(i+j+1)*bs:(i+j+2)*bs,:,:] = self.xs @ Qmat
                ys[(i+j+1)*bs:(i+j+2)*bs] = self.ys
                dys[(i+j+1)*bs:(i+j+2)*bs,:,:] = self.dys @ Qmat
            print(f'Dataset augmented by factor: {i + j + 2}')

        elif a_type == 'concurrent':
            n_mat = self.rmats.__len__()
            for i, Qobj in enumerate(self.robjs.as_matrix()):
                for j, Qmats in enumerate(self.rmats.as_matrix()):
                    xs[(i*n_mat+j)*bs:(i*n_mat+j+1)*bs,:,:] = Qobj @ self.xs @ Qmats
                    ys[(i*n_mat+j)*bs:(i*n_mat+j+1)*bs] = self.ys
                    dys[(i*n_mat+j)*bs:(i*n_mat+j+1)*bs,:,:] = Qobj @ self.dys @ Qmats
            print(f'Dataset augmented by factor: {i * n_mat + j + 1}')

        # update
        self.xs = xs
        self.ys = ys
        self.dys = dys

        self.__apply_loss_weighting(self.xs, np.tile(self.batch_sizes, n_rots))

    def calibrate(self, **kwargs):
        ''' Preform model training '''
        calibrate(self.model, [self.xs], [self.ys, self.dys], 
                    self.sample_weights, **kwargs)

    def evaluate_normalization(self):
        ''' Calls evaluate_normalization static function '''
        return evaluate_normalization(self.model)

    def evaluate_objectivity(self, paths, robjs, qmat):
        ''' Evaluate objectivity using rotations "robjs" for one symmetry case'''
        # get batch sizes to initiliaze correct array dimensions
        batch_sizes = ld.load_stress_strain_data(paths)[3]

        losses = np.zeros([robjs.__len__(), len(paths), 3])

        for i, path in enumerate(paths):
            potentials = np.zeros([robjs.__len__(), batch_sizes[i], 1])
            stresses = np.zeros([robjs.__len__(), batch_sizes[i], 3, 3])

            for j, Qobj in enumerate(robjs.as_matrix()):
                losses[j, i,:], ref, pred, _ = self.__evaluate_single_load_path(path, Qobj, qmat)
                potentials[j,:,:] = pred[0]
                stresses[j,:,:] = Qobj.T @ pred[1]
                P = Qobj.T @ ref[2]

            P_pred = [np.mean(stresses, axis=0),
                np.median(stresses, axis=0),
                np.min(stresses, axis=0),
                np.max(stresses, axis=0)]
            
            fname = path.split('/')[-1].split('.')[0]
            pl.plot_stress_objectivity(P, P_pred, title=path, fname=fname)
        return losses

    def evaluate_matsymmetry(self, paths, rmats, qobj, showplot=True):
        ''' Evaluate material symmetry using rotations "rmats" for one observer '''
        losses = np.zeros([rmats.__len__(), len(paths), 3])

        for i, path in enumerate(paths):
            for j, Qmat in enumerate(rmats.as_matrix()):
                losses[j,i,:] = self.__evaluate_single_load_path(path, qobj, Qmat)[0]

        if showplot: 
            loss = np.array([np.mean(losses, axis=1)[:,0],
                        np.median(losses, axis=1)[:,0],
                        np.min(losses, axis=1)[:,0],
                        np.max(losses, axis=1)[:,0]])
            pl.matsym_loss(loss, title='Material symmetry for one observer', fname='')
        return losses

    def evaluate_concurrent(self, paths, robjs, rmats, showplot=True):
        ''' Concurrent evaluation of material symmetry for each observer '''
        losses = np.zeros([robjs.__len__(), rmats.__len__(), len(paths), 3])
        for i, Qobj in enumerate(robjs.as_matrix()):
            losses[i,:,:,:] = self.evaluate_matsymmetry(paths, rmats, Qobj, showplot=False)

        loss = np.array([np.mean(losses, axis=(0,2))[:,0],
                        np.median(losses, axis=(0,2))[:,0],
                        np.min(losses, axis=(0,2))[:,0],
                        np.max(losses, axis=(0,2))[:,0]])

        if showplot: pl.matsym_loss(loss, title='Material symmetry for multiple observers', fname='concurrent')
        return losses

        

    def evaluate(self, paths, **kwargs):
        ''' Perform evaluation '''
        # initialization
        losses = np.zeros([len(paths), 3])

        # iteration over load paths
        for i, path in enumerate(paths):
            losses[i,:]  = self.__evaluate_single_load_path(path, **kwargs)[0]
        return losses

    def __evaluate_single_load_path(self, path, qobj, qmat, showplots=False):
        ''' Perform evaluation of a single load path '''
        xs, ys, dys, batch_size = self.__load([path])

        # rescale
        ys = ys * self.scalefactor
        dys = dys * self.scalefactor

        if self.loss_weighting:
            sample_weights = ld.get_sample_weights(xs, batch_size)
        else:
            sample_weights = np.ones(np.sum(batch_size))

        # evaluate objectivity and material symmetry condition
        xs = qobj @ xs @ qmat
        dys = qobj @ dys @ qmat

        # predict using the trained model
        ys_pred, dys_pred = self.model.predict(xs)

        # Potential correction - for P training
        # shift reference value ys by normalization offset
        # to ensure reasonable results from tensorflow evalutation function
        # if self.loss_weights == [0, 1]:
        #     ys_I, _ = self.evaluate_normalization()
        #     # normalization
        #     ys_eval = ys + ys_I[0,0]
        #     ys_pred = ys_pred - ys_I[0,0]
        # else:
        #     # no normalization
        #     ys_eval = ys

        loss = self.model.evaluate(xs, [ys, dys], 
                                    verbose=0,
                                    sample_weight=sample_weights)

        if showplots:
            # reshape right Cauchy-Green-Tensor
            cs = lm.IndependentValues()(layers.Flatten()(lm.RightCauchyGreenTensor()(xs)))
            # plots
            fname = path.split('/')[-1].split('.')[0]
            pl.plot_right_cauchy_green_tensor(cs, title=path, fname=fname)
            pl.plot_potential_prediction(ys, ys_pred, title=path, fname=fname)
            pl.plot_stress_tensor_prediction(dys, dys_pred, title=path, fname=fname)

        return loss, [xs, ys, dys], [ys_pred, dys_pred], batch_size

class CubicAnisotropy:
    ''' Class for training a physics augmented neutral network
    calling order: initialize --> preprocess --> calibrate --> evaluate '''
    def __init__(self, paths, loss_weights, loss_weighting, scaling):
        # initialize variables
        self.loss_weights = loss_weights
        self.loss_weighting = loss_weighting
        self.scaling = scaling

        # load calibration data
        self.xs, self.ys, self.dys, self.batch_sizes = self.__load(paths)
        
        # preform pre-preocessing
        self.sample_weights = np.ones(np.sum(self.batch_sizes))
        if self.scaling: self.scalefactor = tf.math.reduce_max(tf.math.abs(self.dys)) ** (-1)
        else: self.scalefactor = 1
        self.__preprocess()

        # create model
        self.model = lm.main(r_type='CubicAnisotropy', loss_weights=self.loss_weights)

    def __load(self, paths):
        ''' Load data for training or testing from a path '''
        xs, _, _, batch_sizes = ld.load_stress_strain_data(paths)

        # compute potential and stress using analytical model
        ys = ld.W(xs)
        dys = ld.P(ld.W)(xs)
        return xs, ys, dys, batch_sizes

    def __preprocess(self):
        ''' Preforms necessary pre-preocessing steps before model calibration '''
        # rescale
        self.ys = self.ys * self.scalefactor
        self.dys = self.dys * self.scalefactor

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

        # scaling
        ys = ys * self.scalefactor
        dys = dys * self.scalefactor

        if self.loss_weighting:
            sample_weights = ld.get_sample_weights(xs, batch_size)
        else:
            sample_weights = np.ones(np.sum(batch_size))

        # predict using the trained model
        ys_pred, dys_pred = self.model.predict(xs)

        # Potential correction - for P training
        # shift reference value ys by normalization offset
        # to ensure reasonable results from tensorflow evalutation function
        # if self.loss_weights == [0, 1]:
        #     ys_I, _ = self.evaluate_normalization()
        #     # normalization
        #     ys_eval = ys + ys_I[0,0]
        #     ys_pred = ys_pred - ys_I[0,0]
        # else:
        #     # no normalization
        #     ys_eval = ys

        loss = self.model.evaluate(xs, [ys, dys], 
                                    verbose=0,
                                    sample_weight=sample_weights)

        if showplots:
             # rescale to originale scale for plotting
            ys_pred = ys_pred
            dys_pred = dys_pred

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

def calibrate(model, inp, out, sw, epochs, verbose=2):
    ''' Preform model training '''
    t1 = now()
    print(t1)

    tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
    h = model.fit(inp, out,
                epochs=epochs,
                verbose=verbose,
                sample_weight=sw)

    t2 = now()
    print('it took', t2 - t1, '(sec) to calibrate the model')
    pl.plot_calibration_loss(h)

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