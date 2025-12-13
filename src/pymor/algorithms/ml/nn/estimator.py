# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('TORCH')

from numbers import Number

import torch
import torch.optim as optim

from pymor.algorithms.ml.nn.neural_networks import FullyConnectedNN, LongShortTermMemoryNN
from pymor.algorithms.ml.nn.train import multiple_restarts_training
from pymor.core.base import BasicObject
from pymor.core.defaults import defaults
from pymor.core.exceptions import NeuralNetworkTrainingError
from pymor.tools.random import get_rng, get_seed_seq


class NeuralNetworkEstimator(BasicObject):

    @defaults('validation_ratio', 'tol', 'neural_network_type')
    def __init__(self, validation_ratio=0.1, tol=None, neural_network_type='FullyConnectedNN',
                 training_parameters={'hidden_layers': '[(N+P)*3, (N+P)*3]', 'activation_function': torch.tanh,
                                      'optimizer': optim.LBFGS, 'epochs': 1000, 'batch_size': 20, 'learning_rate': 1.,
                                      'loss_function': None, 'restarts': 10, 'lr_scheduler': optim.lr_scheduler.StepLR,
                                      'lr_scheduler_params': {'step_size': 10, 'gamma': 0.7},
                                      'es_scheduler_params': {'patience': 10, 'delta': 0.}, 'weight_decay': 0.,
                                      'log_loss_frequency': 0}):
        assert 0 < validation_ratio < 1

        self.__auto_init(locals())

    def fit(self, X, Y, **kwargs):
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

        # compute layer sizes
        self.training_parameters['layer_sizes'] = self._compute_layer_sizes(self.training_parameters['hidden_layers'])

        self.logger.info('Initializing neural network ...')
        # initialize the neural network
        neural_network = self._initialize_neural_network(self.training_parameters)

        self.training_data = [(x, y) for x, y in zip(X, Y, strict=False)]
        number_validation_snapshots = int(len(self.training_data) * self.validation_ratio)
        get_rng().shuffle(self.training_data)
        # split training snapshots into validation and training snapshots
        self.validation_data = self.training_data[0:number_validation_snapshots]
        self.training_data = self.training_data[number_validation_snapshots:]

        # run the actual training of the neural network
        with self.logger.block('Training of neural network ...'):
            target_loss = self._compute_target_loss()

            # run training algorithm with multiple restarts
            self.neural_network, self.losses = multiple_restarts_training(self.training_data, self.validation_data,
                neural_network, target_loss, self.training_parameters['restarts'],
                self.training_parameters['log_loss_frequency'], self.training_parameters)

        self._check_tolerances()

    def _compute_layer_sizes(self, hidden_layers):
        # determine the numbers of neurons in the hidden layers
        if isinstance(hidden_layers, str):
            hidden_layers = eval(hidden_layers, {'N': self.dim_outputs, 'P': self.dim_inputs})
        # input and output size of the neural network are prescribed by the
        # dimension of the parameter space and the reduced basis size
        assert isinstance(hidden_layers, list)
        return [self.dim_inputs, ] + hidden_layers + [self.dim_outputs, ]

    def _initialize_neural_network(self, params):
        neural_network_parameters = {'layer_sizes': params['layer_sizes'],
                                     'activation_function': params['activation_function']}
        if self.neural_network_type == 'FullyConnectedNN':
            neural_network = FullyConnectedNN(**neural_network_parameters).double()
        elif self.neural_network_type == 'LongShortTermMemoryNN':
            assert len(params['layer_sizes']) >= 3
            number_layers = len(params['layer_sizes']) - 2
            neural_network = LongShortTermMemoryNN(input_dimension=params['layer_sizes'][0],
                                                   hidden_dimension=params['layer_sizes'][1],
                                                   output_dimension=params['layer_sizes'][-1],
                                                   number_layers=number_layers).double()
        else:
            raise NotImplementedError(f'Unknown neural network type {self.neural_network_type}!')
        return neural_network

    def _compute_target_loss(self):
        target_loss = None
        if isinstance(self.tol, Number):
            target_loss = self.tol
        return target_loss

    def _check_tolerances(self):
        with self.logger.block('Checking tolerances for error of neural network ...'):

            if isinstance(self.tol, Number):
                if self.losses['full'] > self.tol:
                    raise NeuralNetworkTrainingError('Could not train a neural network that '
                                                      'guarantees prescribed tolerance!')
            elif self.tol is None:
                self.logger.info('Using neural network with smallest validation error ...')
                self.logger.info(f'Finished training with a validation loss of {self.losses["val"]} ...')
            else:
                raise ValueError('Unknown value for mean squared error of neural network')

    def predict(self, X):
        return self.neural_network(torch.DoubleTensor(X)).detach().numpy()
