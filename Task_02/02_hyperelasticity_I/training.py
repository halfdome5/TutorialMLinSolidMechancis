# %% [Load models]
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

# user-defined models
import models as lm
import plots as pl
import data as ld

# %% class definition
class PhysicsAugmented:
    ''' Class for training a physics augmented neutral network
    calling order: initialize --> load --> preprocess --> calibrate --> evaluate '''
    def __init__(self):
        self.lw = [0, 1]     # output_1 = function value, output_2 = gradient
        self.model = lm.main(r_type='PhysicsAugmented', loss_weights=self.lw)

    def load(self, paths):
        ''' Load data for training or testing from a path '''
        self.xs, _, _, self.batch_sizes = ld.load_stress_strain_data(paths)

        # compute potential and stress using analytical model
        self.ys = ld.W(self.xs)
        self.dys = ld.P(ld.W)(self.xs)

    def preprocess(self):
        ''' Preforms necessary pre-preocessing steps before model calibration '''
        # apply load weighting strategy
        self.sw = ld.get_sample_weights(self.xs, self.batch_sizes)

        # reshape inputs
        self.ys = tf.reshape(self.ys,-1)


    def calibrate(self):
        ''' Preform model training '''
        t1 = datetime.now()
        print(t1)

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, 0.002)
        h = self.model.fit([self.xs], [self.ys, self.dys],
                    epochs=300,
                    verbose=2,
                    sample_weight=self.sw)

        t2 = datetime.now()
        print('it took', t2 - t1, '(sec) to calibrate the model')

        # plot some results
        fig = plt.figure(1, dpi=600)
        plt.semilogy(h.history['loss'], label='training loss')
        plt.grid(which='both')
        plt.xlabel('calibration epoch')
        plt.ylabel('log$_{10}$ MSE')
        plt.legend()
        fig.savefig('images/loss.png', dpi=fig.dpi, bbox_inches='tight')


    def evaluate(self, paths, showplots=False):
        ''' Perform evaluation '''
        # initialization
        losses = np.zeros([len(paths), 3])

        # evaluate normalization criterion
        ys_I, dys_I = self.model.predict(np.array([np.identity(3)]))
        print(f'\nW(I) =\t{ys_I[0,0]}')
        print(f'P(I) =\t{dys_I[0,0]}\n\t{dys_I[0,1]}\n\t{dys_I[0,2]}')

        # iteration over load paths
        for i, path in enumerate(paths):
            xs, _, _, _ = ld.load_stress_strain_data([path])

            # compute potential and stress using analytical model
            ys = ld.W(xs)
            dys = ld.P(ld.W)(xs)

            # predict using the trained model
            ys_pred, dys_pred = self.model.predict(xs)
            P = dys
            P_pred = dys_pred

            # Potential correction - for P training
            # shift reference value ys by normalization offset
            # to ensure reasonable results from tensorflow evalutation function
            ys_eval = ys + ys_I[0,0]
            # ys_eval = ys
            ys_pred = ys_pred - ys_I[0,0]

            # Evaluate the model on the test data using `evaluate`
            print(f'\n{path}')
            losses[i,:] = self.model.evaluate(xs, [ys_eval, dys], verbose=0)

            if showplots:
                # plot right Chauchy-Green tensor
                Cs = tf.einsum('ikj,ikl->ijl',xs,xs)
                # pl.plot_right_cauchy_green_tensor(ld.reshape_C(Cs), titles[i], fnames[i])
                pl.plot_right_cauchy_green_tensor(ld.reshape_C(Cs), title=path, fname=None)
                # plot potential
                pl.plot_potential_prediction(ys, ys_pred, title=path, fname=None)

                # plot stress tensor
                pl.plot_stress_tensor_prediction(P, P_pred, title=path, fname=None)

        return losses

    def print_model_parameters(self):
        ''' Displays model parameters '''
        self.model.summary()
        for idx, layer in enumerate(self.model.layers):
            print(layer.name, layer)
            #print(layer.weights, "\n")
            print(layer.get_weights())
