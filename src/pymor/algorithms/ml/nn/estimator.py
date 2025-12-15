# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('TORCH')

from numbers import Number

import torch
import torch.optim as optim

from pymor.algorithms.ml.nn.neural_networks import FullyConnectedNN
from pymor.algorithms.ml.nn.train import multiple_restarts_training
from pymor.core.base import BasicObject
from pymor.core.defaults import defaults
from pymor.core.exceptions import NeuralNetworkTrainingError
from pymor.tools.random import get_rng, get_seed_seq


class NeuralNetworkEstimator(BasicObject):
    """Scikit-learn-style estimator using neural networks from PyTorch.

    Parameters
    ----------
    neural_network
        Neural network to use in the estimator. The neural network should
        implement a function `set_input_output_dimensions` that allows to
        change the input and the output dimension also after initialization.
    validation_ratio
        Ratio of training data to use for validation during training.
    tol
        Prescribed tolerance for the neural network training. If `None`, the
        neural network with the smallest validation loss is used (usually,
        multiple restarts of the training using different initial guesses
        for the weights and biases are performed).
    training_parameters
        Additional training parameters passed to the training algorithm.
        For training,
        :func:`~pymor.algorithms.ml.nn.train.multiple_restarts_training`
        is used.
    """

    @defaults('neural_network', 'validation_ratio', 'tol')
    def __init__(self, neural_network=FullyConnectedNN([30, 30, 30]), validation_ratio=0.1, tol=None,
                 training_parameters={'optimizer': optim.LBFGS, 'epochs': 1000, 'batch_size': 20, 'learning_rate': 1.,
                                      'loss_function': None, 'restarts': 10, 'lr_scheduler': optim.lr_scheduler.StepLR,
                                      'lr_scheduler_params': {'step_size': 10, 'gamma': 0.7},
                                      'es_scheduler_params': {'patience': 10, 'delta': 0.}, 'weight_decay': 0.,
                                      'log_loss_frequency': 0}):
        assert 0 < validation_ratio < 1

        self.__auto_init(locals())

    def fit(self, X, Y, **kwargs):
        """Fit neural network using PyTorch optimization algorithms.

        Parameters
        ----------
        X
            Training inputs.
        Y
            Training targets.
        kwargs
            Additional training parameters that can replace
            the parameters passed to the constructor.

        Returns
        -------
        The trained estimator.
        """
        for key, value in kwargs.items():
            self.training_parameters[key] = value

        assert len(X) == len(Y)

        assert self.training_parameters['restarts'] >= 0
        assert self.training_parameters['epochs'] > 0
        assert self.training_parameters['batch_size'] > 0
        assert self.training_parameters['learning_rate'] > 0.
        assert self.training_parameters['weight_decay'] >= 0.

        torch.manual_seed(get_seed_seq().spawn(1)[0].generate_state(1).item())

        self.dim_inputs = X[0].shape[0]
        self.dim_outputs = Y[0].shape[0]

        self.logger.info('Initializing neural network ...')
        # initialize the neural network
        self.neural_network.set_input_output_dimensions(input_dimension=self.dim_inputs,
                                                        output_dimension=self.dim_outputs)

        self.training_data = [(x, y) for x, y in zip(X, Y, strict=False)]
        number_validation_snapshots = int(len(self.training_data) * self.validation_ratio)
        get_rng().shuffle(self.training_data)
        # split training snapshots into validation and training snapshots
        self.validation_data = self.training_data[0:number_validation_snapshots]
        self.training_data = self.training_data[number_validation_snapshots:]

        # run the actual training of the neural network
        with self.logger.block('Training of neural network ...'):
            # run training algorithm with multiple restarts
            self.neural_network, self.losses = multiple_restarts_training(self.training_data, self.validation_data,
                self.neural_network, self.tol, self.training_parameters['restarts'],
                self.training_parameters['log_loss_frequency'], self.training_parameters)

        with self.logger.block('Checking tolerances for error of neural network ...'):
            if isinstance(self.tol, Number):
                if self.losses['full'] > self.tol:
                    raise NeuralNetworkTrainingError('Could not train a neural network that '
                                                      'guarantees the prescribed tolerance!')
            elif self.tol is None:
                self.logger.info('Using neural network with smallest validation error ...')
                self.logger.info(f'Finished training with a validation loss of {self.losses["val"]} ...')
            else:
                raise ValueError('Unknown value for mean squared error of neural network')

        return self

    def predict(self, X):
        """Predict the target for the input `X`.

        Parameters
        ----------
        X
            Input for which to compute the prediction.

        Returns
        -------
        Prediction obtained by the neural network.
        """
        return self.neural_network(torch.DoubleTensor(X)).detach().numpy()
