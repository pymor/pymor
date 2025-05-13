# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('TORCH')


import inspect
from numbers import Number

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from pymor.algorithms.pod import pod
from pymor.algorithms.projection import project
from pymor.core.base import BasicObject
from pymor.core.exceptions import NeuralNetworkTrainingError
from pymor.core.logger import getLogger
from pymor.models.neural_network import (
    FullyConnectedNN,
    LongShortTermMemoryNN,
    NeuralNetworkInstationaryModel,
    NeuralNetworkInstationaryStatefreeOutputModel,
    NeuralNetworkModel,
    NeuralNetworkStatefreeOutputModel,
)
from pymor.tools.random import get_rng, get_seed_seq


class NeuralNetworkReductor(BasicObject):
    """Reduced Basis reductor relying on artificial neural networks.

    This is a reductor that constructs a reduced basis using proper
    orthogonal decomposition and trains a neural network that approximates
    the mapping from parameter space to coefficients of the full-order
    solution in the reduced basis.
    The approach is described in :cite:`HU18`.

    Parameters
    ----------
    fom
        The full-order |Model| to reduce. If `None`, the `training_parameters` has
        to consist of pairs of |parameter values| and corresponding solution
        |VectorArrays|.
    reduced_basis
        Prescribed reduced basis of the full-order |Model|. If `None`, the
        reduced basis is computed using the :meth:`~pymor.algorithms.pod.pod` method.
    training_parameters
        |Parameter values| to use for POD (in case no `reduced_basis` is provided) and
        training of the neural network.
    training_snapshots
        |VectorArray| to use for POD and training of the
        neural network. Contains the solutions to the parameters of the
        `training_parameters` and can be `None` when `fom` is not `None`.
    validation_parameters
        |Parameter values| to use for validation in the training
        of the neural network.
    validation_snapshots
        |VectorArray| to use for validation in the training
        of the neural network. Contains the solutions to the parameters of
        the `validation_parameters` and can be `None` when `fom` is not `None`.
    validation_ratio
        Fraction of the training parameters to use for validation in the training
        of the neural network (only used if no validation parameters are provided).
        Either validation parameters or a positive validation ratio is required.
    basis_size
        Desired size of the reduced basis. If `None`, rtol, atol or l2_err must
        be provided.
    rtol
        Relative tolerance the basis should guarantee on the training set.
    atol
        Absolute tolerance the basis should guarantee on the training set.
    l2_err
        L2-approximation error the basis should not exceed on the training
        set.
    pod_params
        Dict of additional parameters for the POD-method.
    ann_mse
        If `'like_basis'`, the mean squared error of the neural network on
        the training parameters should not exceed the error of projecting onto the basis.
        If `None`, the neural network with smallest validation error is
        used to build the ROM.
        If a tolerance is prescribed, the mean squared error of the neural
        network on the training parameters should not exceed this threshold.
        Training is interrupted if a neural network that undercuts the
        error tolerance is found.
    scale_inputs
        Determines whether or not to scale the inputs of the neural networks.
    scale_outputs
        Determines whether or not to scale the outputs/targets of the neural
        networks.
    """

    def __init__(self, fom=None, reduced_basis=None, training_parameters=None, validation_parameters=None, training_snapshots=None,
                 validation_snapshots=None, validation_ratio=0.1, T=None, nt=1, basis_size=None, rtol=0., atol=0.,
                 l2_err=0., pod_params={}, ann_mse='like_basis', scale_inputs=True, scale_outputs=False):
        assert 0 < validation_ratio < 1 or validation_parameters

        self.scaling_parameters = {'min_inputs': None, 'max_inputs': None,
                                   'min_targets': None, 'max_targets': None}
        self.training_data = None
        self.validation_data = None

        if not fom:
            assert training_parameters is not None
            assert len(training_parameters) > 0
            assert training_snapshots is not None
            self.parameters_dim = training_parameters[0].parameters().dim
            self.nt = int(len(training_snapshots) / len(training_parameters))
            if self.nt > 1:  # instationary
                assert T is not None
                self.T = T
                self.is_stationary = False
            else:  # stationary
                self.is_stationary = True
        else:
            self.parameters_dim = fom.parameters.dim
            if hasattr(fom, 'time_stepper'):  # instationary
                self.nt = fom.time_stepper.nt + 1
                self.T = fom.T
                self.is_stationary = False
            else:  # stationary
                self.nt = 1
                self.is_stationary = True

        self.__auto_init(locals())

    def reduce(self, hidden_layers='[(N+P)*3, (N+P)*3]', activation_function=torch.tanh,
               optimizer=optim.LBFGS, epochs=1000, batch_size=20, learning_rate=1.,
               loss_function=None, restarts=10, lr_scheduler=optim.lr_scheduler.StepLR,
               lr_scheduler_params={'step_size': 10, 'gamma': 0.7},
               es_scheduler_params={'patience': 10, 'delta': 0.}, weight_decay=0.,
               log_loss_frequency=0):
        """Reduce by training artificial neural networks.

        Parameters
        ----------
        hidden_layers
            Number of neurons in the hidden layers. Can either be fixed or
            a Python expression string depending on the reduced basis size
            respectively output dimension `N` and the total dimension of
            the |Parameters| `P`.
        activation_function
            Activation function to use between the hidden layers.
        optimizer
            Algorithm to use as optimizer during training.
        epochs
            Maximum number of epochs for training.
        batch_size
            Batch size to use if optimizer allows mini-batching.
        learning_rate
            Step size to use in each optimization step.
        loss_function
            Loss function to use for training. If `'weighted MSE'`, a weighted
            mean squared error is used as loss function, where the weights are
            given as the singular values of the corresponding reduced basis
            functions. If `None`, the usual mean squared error is used.
        restarts
            Number of restarts of the training algorithm. Since the training
            results highly depend on the initial starting point, i.e. the
            initial weights and biases, it is advisable to train multiple
            neural networks by starting with different initial values and
            choose that one performing best on the validation set.
        lr_scheduler
            Algorithm to use as learning rate scheduler during training.
            If `None`, no learning rate scheduler is used.
        lr_scheduler_params
            A dictionary of additional parameters passed to the init method of
            the learning rate scheduler. The possible parameters depend on the
            chosen learning rate scheduler.
        es_scheduler_params
            A dictionary of additional parameters passed to the init method of
            the early stopping scheduler. For the possible parameters,
            see :class:`EarlyStoppingScheduler`.
        weight_decay
            Weighting parameter for the l2-regularization of the weights and
            biases in the neural network. This regularization is not available
            for all optimizers; see the PyTorch documentation for more details.
        log_loss_frequency
            Frequency of epochs in which to log the current validation and
            training loss during training of the neural networks.
            If `0`, no intermediate logging of losses is done.

        Returns
        -------
        rom
            Reduced-order |NeuralNetworkModel|.
        """
        assert restarts >= 0
        assert epochs > 0
        assert batch_size > 0
        assert learning_rate > 0.
        assert weight_decay >= 0.

        torch.manual_seed(get_seed_seq().spawn(1)[0].generate_state(1).item())

        # compute training snapshots
        if self.training_snapshots is None:
            self.compute_training_snapshots()

        # build a reduced basis using POD if necessary
        if self.reduced_basis is None:
            self.compute_reduced_basis()

        # compute training data
        # i.e. pairs of parameters (potentially including time) and reduced coefficients
        if self.training_data is None:
            self.compute_training_data()
        assert self.training_data is not None
        assert len(self.training_data) == len(self.training_parameters) * self.nt

        if self.validation_parameters is None:
            number_validation_snapshots = int(len(self.training_data) * self.validation_ratio)
            if self.is_stationary:
                # randomly shuffle training data before splitting into two sets
                get_rng().shuffle(self.training_data)
                # split training snapshots into validation and training snapshots
                self.validation_data = self.training_data[0:number_validation_snapshots]
                self.validation_parameters = [data[0] for data in self.validation_data]
                self.training_data = self.training_data[number_validation_snapshots:]
            else:
                # Create blocks of timesteps for each paraneter
                blocksize = self.nt
                blocks = [self.training_data[i:i + blocksize] for i in range(0, len(self.training_data), blocksize)]
                # shuffle the blocks
                get_rng().shuffle(blocks)
                # concatenate the shuffled blocks into a single list
                self.training_data[:] = [timesteps for parameter in blocks for timesteps in parameter]
                # split training snapshots into validation and training snapshots
                self.validation_data = self.training_data[0:number_validation_snapshots]
                self.validation_parameters = [data[0] for data in self.validation_data[::blocksize]]
                self.training_data = self.training_data[number_validation_snapshots:]

        # compute validation snapshots if not given as input
        if self.validation_data is None:
            if self.fom is None:
                assert self.validation_snapshots is not None
            else:
                if self.validation_snapshots is None:
                    self.compute_validation_snapshots()

            # compute validation data
            self.compute_validation_data()
        assert self.validation_data is not None
        assert len(self.validation_data) == len(self.validation_parameters) * self.nt

        # compute layer sizes
        layer_sizes = self._compute_layer_sizes(hidden_layers)

        # run the actual training of the neural network
        with self.logger.block('Training of neural network ...'):
            target_loss = self._compute_target_loss()
            # set parameters for training and neural network
            if loss_function == 'weighted MSE':
                if hasattr(self, 'weights'):
                    weights = self.weights

                    def weighted_mse_loss_function(inputs, targets):
                        return (weights * (inputs - targets) ** 2).mean()

                    loss_function = weighted_mse_loss_function
                    self.logger.info('Using weighted MSE loss function ...')
                else:
                    raise RuntimeError('No weights for weighted MSE loss available!')
            training_parameters = {'optimizer': optimizer, 'epochs': epochs,
                                   'batch_size': batch_size, 'learning_rate': learning_rate,
                                   'lr_scheduler': lr_scheduler, 'lr_scheduler_params': lr_scheduler_params,
                                   'es_scheduler_params': es_scheduler_params, 'weight_decay': weight_decay,
                                   'loss_function': loss_function}

            self.logger.info('Initializing neural network ...')
            # initialize the neural network
            neural_network = self._initialize_neural_network(locals())

            # run training algorithm with multiple restarts
            self.neural_network, self.losses = multiple_restarts_training(self.training_data, self.validation_data,
                                                                          neural_network, target_loss, restarts,
                                                                          log_loss_frequency, training_parameters,
                                                                          self.scaling_parameters)

        self._check_tolerances()

        return self._build_rom()

    def _initialize_neural_network(self, params):
        """Initialize the neural network using the required parameters."""
        neural_network_parameters = {'layer_sizes': params['layer_sizes'],
                                     'activation_function': params['activation_function']}
        neural_network = FullyConnectedNN(**neural_network_parameters).double()
        return neural_network

    def compute_training_snapshots(self):
        """Compute training snapshots for the neural network."""
        # compute snapshots for POD and training of neural networks
        with self.logger.block('Computing training snapshots ...'):
            self.training_snapshots = self.fom.solution_space.empty()
            for mu in self.training_parameters:
                u = self.fom.solve(mu)
                self.training_snapshots.append(u)

    def compute_reduced_basis(self):
        """Compute a reduced basis using proper orthogonal decomposition."""
        # compute reduced basis via POD
        with self.logger.block('Building reduced basis ...'):
            self.reduced_basis, svals = pod(self.training_snapshots, modes=self.basis_size, rtol=self.rtol / 2.,
                                            atol=self.atol / 2., l2_err=self.l2_err / 2.,
                                            **(self.pod_params or {}))

            # set singular values as weights for the weighted MSE loss
            self.weights = torch.Tensor(svals)

            # compute mean square loss
            self.mse_basis = (sum(self.training_snapshots.norm2()) - sum(svals ** 2)) / len(self.training_snapshots)

    def compute_training_data(self):
        """Compute training data for the neural network using the reduced basis."""
        # compute training samples
        with self.logger.block('Computing training samples ...'):
            self.training_data = []
            for i, mu in enumerate(self.training_parameters):
                samples = self._compute_sample(mu, self.training_snapshots[i*self.nt:(i+1)*self.nt])
                # compute minimum and maximum of outputs/targets for scaling
                for sample in samples:
                    self._update_scaling_parameters(sample)
                self.training_data.extend(samples)

    def compute_validation_snapshots(self):
        """Compute validation data for the neural network."""
        # compute snapshots for POD and validation of neural networks
        with self.logger.block('Computing validation snapshots ...'):
            self.validation_snapshots = self.fom.solution_space.empty()
            for mu in self.validation_parameters:
                u = self.fom.solve(mu)
                self.validation_snapshots.append(u)

    def compute_validation_data(self):
            assert self.validation_parameters is not None
            with self.logger.block('Computing validation samples ...'):
                validation_data_iterable = zip(self.validation_parameters, self.validation_snapshots)

                self.validation_data = []
                for i, (mu, u) in enumerate(validation_data_iterable):
                    samples = self._compute_sample(mu, u.base[i * self.nt:(i + 1) * self.nt])
                    self.validation_data.extend(samples)

    def _update_scaling_parameters(self, sample):
        """Update the quantities for scaling of inputs and outputs."""
        assert len(sample) == 2 or (len(sample) == 1 and len(sample[0]) == 2)
        if len(sample) == 1:
            sample = sample[0]

        def prepare_datum(datum):
            if not (isinstance(datum, (torch.DoubleTensor, np.ndarray))):
                return datum.to_numpy()
            return datum
        sample = (torch.DoubleTensor(prepare_datum(sample[0])), torch.DoubleTensor(prepare_datum(sample[1])))

        if self.scale_inputs:
            if sample[0].ndim > 1:
                s_in_min, _ = torch.min(sample[0], 0)
            else:
                s_in_min = sample[0]
            if self.scaling_parameters['min_inputs'] is not None:

                self.scaling_parameters['min_inputs'] = torch.min(self.scaling_parameters['min_inputs'], s_in_min)
            else:
                self.scaling_parameters['min_inputs'] = s_in_min
            if sample[0].ndim > 1:
                s_in_max, _ = torch.max(sample[0], 0)
            else:
                s_in_max = sample[0]
            if self.scaling_parameters['max_inputs'] is not None:
                self.scaling_parameters['max_inputs'] = torch.max(self.scaling_parameters['max_inputs'], s_in_max)
            else:
                self.scaling_parameters['max_inputs'] = s_in_max

        if self.scale_outputs:
            if sample[1].ndim > 1:
                s_out_min, _ = torch.min(sample[1], 0)
            else:
                s_out_min = sample[1]
            if self.scaling_parameters['min_targets'] is not None:
                self.scaling_parameters['min_targets'] = torch.min(self.scaling_parameters['min_targets'],
                                                                   s_out_min)
            else:
                self.scaling_parameters['min_targets'] = s_out_min
            if sample[1].ndim > 1:
                s_out_max, _ = torch.max(sample[1], 0)
            else:
                s_out_max = sample[1]
            if self.scaling_parameters['max_targets'] is not None:
                self.scaling_parameters['max_targets'] = torch.max(self.scaling_parameters['max_targets'],
                                                                   s_out_max)
            else:
                self.scaling_parameters['max_targets'] = s_out_max

    def _compute_sample(self, mu, u=None):
        """Transform parameter and corresponding solution to |NumPy arrays|."""
        # determine the coefficients of the full-order solutions in the reduced basis to obtain
        # the training data
        if u is None:
            assert self.fom is not None
            u = self.fom.solve(mu)

        product = self.pod_params.get('product')

        # conditional expression to check for instationary solution to return self.nt solutions
        parameters = [mu] if self.is_stationary else [mu.with_(t=t) for t in np.linspace(0, self.T, self.nt)]
        samples = [(mu, self.reduced_basis.inner(u_t, product=product)[:, 0]) for mu, u_t in
                   zip(parameters, u)]

        return samples

    def _compute_layer_sizes(self, hidden_layers):
        """Compute the number of neurons in the layers of the neural network.

        The input dimension is increased by one to account for the time in
        case of instationary data.
        """

        # determine the numbers of neurons in the hidden layers
        if isinstance(hidden_layers, str):
            hidden_layers = eval(hidden_layers, {'N': len(self.reduced_basis), 'P': self.parameters_dim})
        # input and output size of the neural network are prescribed by the
        # dimension of the parameter space and the reduced basis size
        assert isinstance(hidden_layers, list)
        if self.is_stationary:
            return [self.parameters_dim, ] + hidden_layers + [len(self.reduced_basis), ]
        else:
            return [self.parameters_dim + 1, ] + hidden_layers + [len(self.reduced_basis), ]

    def _compute_target_loss(self):
        """Compute target loss depending on value of `ann_mse`."""
        target_loss = None
        if isinstance(self.ann_mse, Number):
            target_loss = self.ann_mse
        elif self.ann_mse == 'like_basis':
            target_loss = self.mse_basis
        return target_loss

    def _check_tolerances(self):
        """Check if trained neural network is sufficient to guarantee certain error bounds."""
        with self.logger.block('Checking tolerances for error of neural network ...'):

            if isinstance(self.ann_mse, Number):
                if self.losses['full'] > self.ann_mse:
                    raise NeuralNetworkTrainingError('Could not train a neural network that '
                                                      'guarantees prescribed tolerance!')
            elif self.ann_mse == 'like_basis':
                if self.losses['full'] > self.mse_basis:
                    raise NeuralNetworkTrainingError('Could not train a neural network with an error as small as '
                                                      'the reduced basis error! Maybe you can try a different '
                                                      'neural network architecture or change the value of '
                                                      '`ann_mse`.')
            elif self.ann_mse is None:
                self.logger.info('Using neural network with smallest validation error ...')
                self.logger.info(f'Finished training with a validation loss of {self.losses["val"]} ...')
            else:
                raise ValueError('Unknown value for mean squared error of neural network')

    def _build_rom(self):
        """Construct the reduced order model."""
        if self.fom:
            projected_output_functional = project(self.fom.output_functional, None, self.reduced_basis)
            parameters = self.fom.parameters
            name = self.fom.name
        else:
            projected_output_functional = None
            parameters = self.training_parameters[0].parameters()
            name = 'data_driven'

        with self.logger.block('Building ROM ...'):
            if self.is_stationary:
                rom = NeuralNetworkModel(self.neural_network, parameters,
                                     scaling_parameters=self.scaling_parameters,
                                     output_functional=projected_output_functional,
                                     name=f'{name}_reduced')
            else:
                rom = NeuralNetworkInstationaryModel(self.T, self.nt, self.neural_network,
                                                     parameters=parameters,
                                                     scaling_parameters=self.scaling_parameters,
                                                     output_functional=projected_output_functional,
                                                     name=f'{name}_reduced')
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        assert hasattr(self, 'reduced_basis')
        return self.reduced_basis.lincomb(u.to_numpy())


class NeuralNetworkStatefreeOutputReductor(NeuralNetworkReductor):
    """Output reductor relying on artificial neural networks.

    This is a reductor that trains a neural network that approximates
    the mapping from parameter space to output space.

    Parameters
    ----------
    fom
        The full-order |Model| to reduce. If `None`, the `training_parameters` has
        to consist of pairs of |parameter values| and corresponding outputs.
    training_parameters
        List of |Parameter values| to use for training of the
        neural network.
    training_outputs
        Set of outputs corresponding to a set of |Parameter values| used
        for training of the neural network. These are the outputs to the
        parameters of the`training_parameters` and can be `None` when `fom` is
        not `None`.
    validation_parameters
        List of |Parameter values| to use for validation in the training
        of the neural network.
    validation_outputs
        Set of outputs corresponding to a set of |Parameter values| used
        for validation of the neural network. These are the outputs to the
        parameters of the`validation_parameters` and can be `None` when `fom` is
        not `None`.
    validation_ratio
        See :class:`~pymor.reductors.neural_network.NeuralNetworkReductor`.
    validation_loss
        The validation loss to reach during training. If `None`, the neural
        network with the smallest validation loss is returned.
    scale_inputs
        See :class:`~pymor.reductors.neural_network.NeuralNetworkReductor`.
    scale_outputs
        See :class:`~pymor.reductors.neural_network.NeuralNetworkReductor`.
    """

    def __init__(self, fom=None, training_parameters=None, validation_parameters=None, training_outputs=None,
                 validation_outputs=None, validation_ratio=0.1, T=None, nt=1, validation_loss=None,
                 scale_inputs=True, scale_outputs=False):
        assert 0 < validation_ratio < 1 or validation_parameters

        self.scaling_parameters = {'min_inputs': None, 'max_inputs': None,
                                   'min_targets': None, 'max_targets': None}

        super().__init__(fom=fom, training_parameters=training_parameters, validation_parameters=validation_parameters,
                         training_snapshots=training_outputs, validation_snapshots=validation_outputs,
                         validation_ratio=validation_ratio, T=T, nt=nt, scale_inputs=scale_inputs,
                         scale_outputs=scale_outputs)
        if not fom:
            self.dim_output = training_outputs[0].size
        else:
            self.dim_output = fom.dim_output

        self.__auto_init(locals())

    def compute_training_data(self):
        """Compute the training samples (the outputs to the parameters of the training set)."""
        with self.logger.block('Computing training samples ...'):
            self.training_data = []
            for i, mu in enumerate(self.training_parameters):
                if not self.fom:
                    samples = self._compute_sample(mu, output=self.training_outputs[i*self.nt:(i+1)*self.nt])
                else:
                    samples = self._compute_sample(mu)
                for sample in samples:
                    self._update_scaling_parameters(sample)
                self.training_data.extend(samples)

    def compute_validation_data(self):
        """Compute the training samples (the outputs to the parameters of the training set)."""
        with self.logger.block('Computing validation samples ...'):

            self.validation_data = []
            for i, mu in enumerate(self.validation_parameters):
                if not self.fom:
                    samples = self._compute_sample(mu, output=self.validation_outputs[i*self.nt:(i+1)*self.nt])
                else:
                    samples = self._compute_sample(mu)
                for sample in samples:
                    self._update_scaling_parameters(sample)
                self.validation_data.extend(samples)

    def compute_reduced_basis(self):
        """empty function to avoid computing a reduced basis."""

    def compute_training_snapshots(self):
        """empty function to avoid computing training_snapshots."""

    def compute_validation_snapshots(self):
        """empty function to avoid computing validation_snapshots."""

    def _compute_sample(self, mu, output=None):
        """Transform parameter and corresponding output to tensors."""
        if output is None:
            output = self.fom.output(mu).flatten()
        else:
            output = output.flatten()

        if self.is_stationary:
            samples = [(mu, output)]
        else:
            output_size = output.shape[0]
            # conditional expression to check for instationary solution to return self.nt solutions
            parameters = [mu.with_(t=t) for t in np.linspace(0, self.T, output_size)] if output_size > 1 else [mu]
            samples = [(param, np.array([out])) for param, out in zip(parameters, output)]
        return samples


    def _compute_layer_sizes(self, hidden_layers):
        """Compute the number of neurons in the layers of the neural network."""
        # determine the numbers of neurons in the hidden layers
        if isinstance(hidden_layers, str):
            hidden_layers = eval(hidden_layers, {'N': self.dim_output, 'P': self.parameters_dim})
        # input and output size of the neural network are prescribed by the
        # dimension of the parameter space and the output dimension
        assert isinstance(hidden_layers, list)
        if self.is_stationary:
            return [self.parameters_dim, ] + hidden_layers + [self.dim_output, ]
        else:
            return [self.parameters_dim + 1, ] + hidden_layers + [self.dim_output, ]

    def _compute_target_loss(self):
        """Compute target loss depending on value of `ann_mse`."""
        return self.validation_loss

    def _check_tolerances(self):
        """Check if trained neural network is sufficient to guarantee certain error bounds."""
        self.logger.info('Using neural network with smallest validation error ...')
        self.logger.info(f'Finished training with a validation loss of {self.losses["val"]} ...')

    def _build_rom(self):
        """Construct the reduced order model."""
        if self.fom:
            parameters = self.fom.parameters
            name = self.fom.name
        else:
            parameters = self.training_parameters[0].parameters()
            name = 'data_driven'

        with self.logger.block('Building ROM ...'):
            if self.is_stationary:
                rom = NeuralNetworkStatefreeOutputModel(self.neural_network, parameters=parameters,
                                                    scaling_parameters=self.scaling_parameters,
                                                    name=f'{name}_output_reduced')
            else:
                rom = NeuralNetworkInstationaryStatefreeOutputModel(self.T, self.nt, self.neural_network,
                                                                    parameters=parameters,
                                                                    scaling_parameters=self.scaling_parameters,
                                                                    name=f'{name}_output_reduced')

        return rom

class NeuralNetworkLSTMReductor(NeuralNetworkReductor):

    """Reduced Basis reductor for instationary problems relying on LSTM neural networks.

    This is a reductor that constructs a reduced basis using proper
    orthogonal decomposition and trains an LSTM neural network that approximates
    the mapping from parameter to coefficients of the full-order solution
    in the reduced basis for a fixed number of timesteps.
    """

    def reduce(self, hidden_dimension='3*N + P', number_layers=1, optimizer=optim.LBFGS,
               epochs=1000, batch_size=20, learning_rate=1., loss_function=None, restarts=10,
               lr_scheduler=None, lr_scheduler_params={},
               es_scheduler_params={'patience': 10, 'delta': 0.}, weight_decay=0.,
               log_loss_frequency=0):
        """Reduce by LSTM neural networks.

        Parameters
        ----------
        hidden_dimension
            Number of neurons in the hidden state of the LSTM. Can either
            be fixed or a Python expression string depending on the reduced
            basis size respectively output dimension `N` and the total
            dimension of the |Parameters| `P`.
        number_layers
            Number of recurred layers, i.e. number of stacked LSTM cells in the
            neural network.
        optimizer
            See :class:`~pymor.reductors.neural_network.NeuralNetworkReductor`.
        epochs
            See :class:`~pymor.reductors.neural_network.NeuralNetworkReductor`.
        batch_size
            See :class:`~pymor.reductors.neural_network.NeuralNetworkReductor`.
        learning_rate
            See :class:`~pymor.reductors.neural_network.NeuralNetworkReductor`.
        loss_function
            See :class:`~pymor.reductors.neural_network.NeuralNetworkReductor`.
        restarts
            See :class:`~pymor.reductors.neural_network.NeuralNetworkReductor`.
        lr_scheduler
            See :class:`~pymor.reductors.neural_network.NeuralNetworkReductor`.
        lr_scheduler_params
            See :class:`~pymor.reductors.neural_network.NeuralNetworkReductor`.
        es_scheduler_params
            See :class:`~pymor.reductors.neural_network.NeuralNetworkReductor`.
        weight_decay
            See :class:`~pymor.reductors.neural_network.NeuralNetworkReductor`.
        log_loss_frequency
            See :class:`~pymor.reductors.neural_network.NeuralNetworkReductor`.
        """
        hidden_layers = [hidden_dimension, number_layers]
        return super().reduce(hidden_layers=hidden_layers, optimizer=optimizer, epochs=epochs,
                              batch_size=batch_size, learning_rate=learning_rate, loss_function=loss_function,
                              restarts=restarts, lr_scheduler=lr_scheduler, lr_scheduler_params=lr_scheduler_params,
                              es_scheduler_params=es_scheduler_params, weight_decay=weight_decay,
                              log_loss_frequency=log_loss_frequency)

    def _initialize_neural_network(self, params):
        """Initialize the neural network using the required parameters."""
        layer_sizes = params['layer_sizes']
        hidden_layers = params['hidden_layers']
        number_layers = hidden_layers[1]

        neural_network_parameters = {'input_dimension': layer_sizes[0],
                                     'hidden_dimension': layer_sizes[1],
                                     'output_dimension': layer_sizes[2],
                                     'number_layers': number_layers}
        neural_network = LongShortTermMemoryNN(**neural_network_parameters).double()
        return neural_network

    def _compute_layer_sizes(self, hidden_layers):
        """Compute the number of neurons in the layers of the neural network."""
        hidden_dimension = hidden_layers[0]
        if isinstance(hidden_dimension, str):
            hidden_dimension = eval(hidden_dimension, {'N': len(self.reduced_basis), 'P': self.fom.parameters.dim})

        assert isinstance(hidden_dimension, int)
        # input and output size of the neural network are prescribed by the
        # dimension of the parameter space and the reduced basis size
        if self.is_stationary:
            return [self.fom.parameters.dim, hidden_dimension, len(self.reduced_basis), ]
        else:
            return [self.fom.parameters.dim + 1, hidden_dimension, len(self.reduced_basis), ]

class NeuralNetworkLSTMStatefreeOutputReductor(NeuralNetworkStatefreeOutputReductor,
                                               NeuralNetworkLSTMReductor):

    """Output reductor relying on LSTM neural networks.

    This is a reductor that trains an LSTM neural network that approximates
    the mapping from parameter space to output space.
    """

    reduce = NeuralNetworkLSTMReductor.reduce

    _initialize_neural_network = NeuralNetworkLSTMReductor._initialize_neural_network

    def _compute_layer_sizes(self, hidden_layers):
        """Compute the number of neurons in the layers of the neural network."""
        hidden_dimension = hidden_layers[0]
        if isinstance(hidden_dimension, str):
            hidden_dimension = eval(hidden_dimension, {'N': self.dim_output, 'P': self.parameters_dim})

        assert isinstance(hidden_dimension, int)
        # input and output size of the neural network are prescribed by the
        # dimension of the parameter space and the reduced basis size
        if self.is_stationary:
            return [self.fom.parameters_dim, hidden_dimension, self.dim_output, ]
        else:
            return [self.fom.parameters.dim + 1, hidden_dimension, self.dim_output, ]


class EarlyStoppingScheduler(BasicObject):
    """Class for performing early stopping in training of neural networks.

    If the validation loss does not decrease over a certain amount of epochs, the
    training should be aborted to avoid overfitting the training data.
    This class implements an early stopping scheduler that recommends to stop the
    training process if the validation loss did not decrease by at least `delta`
    over `patience` epochs.

    Parameters
    ----------
    size_training_validation_parameters
        Size of both, training and validation parameters together.
    patience
        Number of epochs of non-decreasing validation loss allowed, before early
        stopping the training process.
    delta
        Minimal amount of decrease in the validation loss that is required to reset
        the counter of non-decreasing epochs.
    """

    def __init__(self, size_training_validation_parameters, patience=10, delta=0.):
        self.__auto_init(locals())

        self.best_losses = None
        self.best_neural_network = None
        self.counter = 0

    def __call__(self, losses, neural_network=None):
        """Returns `True` if early stopping of training is suggested.

        Parameters
        ----------
        losses
            Dictionary of losses on the validation and the training parameters in
            the current epoch.
        neural_network
            Neural network that produces the current validation loss.

        Returns
        -------
        `True` if early stopping is suggested, `False` otherwise.
        """
        import copy
        if self.best_losses is None:
            self.best_losses = losses
            self.best_losses['full'] /= self.size_training_validation_parameters
            self.best_neural_network = copy.deepcopy(neural_network)
        elif self.best_losses['val'] - self.delta <= losses['val']:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_losses = losses
            self.best_losses['full'] /= self.size_training_validation_parameters
            self.best_neural_network = copy.deepcopy(neural_network)
            self.counter = 0

        return False


class CustomDataset(utils.data.Dataset):
    """Class that represents the dataset to use in PyTorch.

    Parameters
    ----------
    training_data
        Set of training parameters and the respective coefficients of the
        solution in the reduced basis.
    """

    def __init__(self, training_data):
        self.training_data = training_data

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        t = self.training_data[idx]
        return t


def train_neural_network(training_data, validation_data, neural_network,
                         training_parameters={}, scaling_parameters={}, log_loss_frequency=0):
    """Training algorithm for artificial neural networks.

    Trains a single neural network using the given training and validation data.

    Parameters
    ----------
    training_data
        Data to use during the training phase. Has to be a list of tuples,
        where each tuple consists of two elements that are either
        PyTorch-tensors (`torch.DoubleTensor`) or |NumPy arrays| or pyMOR data
        structures that have `to_numpy()` implemented.
        The first element contains the input data, the second element contains
        the target values.
    validation_data
        Data to use during the validation phase. Has to be a list of tuples,
        where each tuple consists of two elements that are either
        PyTorch-tensors (`torch.DoubleTensor`) or |NumPy arrays| or pyMOR data
        structures that have `to_numpy()` implemented.
        The first element contains the input data, the second element contains
        the target values.
    neural_network
        The neural network to train (can also be a pre-trained model).
        Has to be a PyTorch-Module.
    training_parameters
        Dictionary with additional parameters for the training routine like
        the type of the optimizer, the (maximum) number of epochs, the batch
        size, the learning rate or the loss function to use.
        Possible keys are `'optimizer'` (an optimizer from the PyTorch `optim`
        package; if not provided, the LBFGS-optimizer is taken as default),
        `'epochs'` (an integer that determines the number of epochs to use
        for training the neural network (if training is not interrupted
        prematurely due to early stopping); if not provided, 1000 is taken as
        default value), `'batch_size'` (an integer that determines the number
        of samples to pass to the optimizer at once; if not provided, 20 is
        taken as default value; not used in the case of the LBFGS-optimizer
        since LBFGS does not support mini-batching), `'learning_rate'` (a
        positive real number used as the (initial) step size of the optimizer;
        if not provided, 1 is taken as default value), `'loss_function'`
        (a loss function from PyTorch; if not provided, the MSE loss is taken
        as default), `'lr_scheduler'` (a learning rate scheduler from the
        PyTorch `optim.lr_scheduler` package; if not provided or `None`,
        no learning rate scheduler is used), `'lr_scheduler_params'`
        (a dictionary of additional parameters for the learning rate
        scheduler), `'es_scheduler_params'` (a dictionary of additional
        parameters for the early stopping scheduler), and `'weight_decay'`
        (non-negative real number that determines the strength of the
        l2-regularization; if not provided or 0., no regularization is applied).
    scaling_parameters
        Dict of tensors that determine how to scale inputs before passing them
        through the neural network and outputs after obtaining them from the
        neural network. If not provided or each entry is `None`, no scaling is
        applied. Required keys are `'min_inputs'`, `'max_inputs'`, `'min_targets'`,
        and `'max_targets'`.
    log_loss_frequency
        Frequency of epochs in which to log the current validation and
        training loss. If `0`, no intermediate logging of losses is done.

    Returns
    -------
    best_neural_network
        The best trained neural network with respect to validation loss.
    losses
        The corresponding losses as a dictionary with keys `'full'` (for the
        full loss containing the training and the validation average loss),
        `'train'` (for the average loss on the training set), and `'val'`
        (for the average loss on the validation set).
    """
    assert isinstance(neural_network, nn.Module)
    assert isinstance(log_loss_frequency, int)

    for data in training_data, validation_data:
        assert isinstance(data, list)
        assert all(isinstance(datum, tuple) and len(datum) == 2 for datum in data)

    def prepare_datum(datum):
        if not (isinstance(datum, (torch.DoubleTensor, np.ndarray))):
            return datum.to_numpy()
        return datum

    training_data = [(prepare_datum(datum[0]), prepare_datum(datum[1])) for datum in training_data]
    validation_data = [(prepare_datum(datum[0]), prepare_datum(datum[1])) for datum in validation_data]

    optimizer = training_parameters.get('optimizer', optim.LBFGS)
    epochs = training_parameters.get('epochs', 1000)
    assert isinstance(epochs, int)
    assert epochs > 0
    batch_size = training_parameters.get('batch_size', 20)
    assert isinstance(batch_size, int)
    assert batch_size > 0
    learning_rate = training_parameters.get('learning_rate', 1.0)
    assert learning_rate > 0.
    loss_function = (nn.MSELoss() if (training_parameters.get('loss_function') is None)
                     else training_parameters['loss_function'])

    logger = getLogger('pymor.algorithms.neural_network.train_neural_network')

    # LBFGS-optimizer does not support mini-batching, so the batch size needs to be adjusted
    if optimizer == optim.LBFGS:
        batch_size = max(len(training_data), len(validation_data))

    # initialize optimizer, early stopping scheduler and learning rate scheduler
    weight_decay = training_parameters.get('weight_decay', 0.)
    assert weight_decay >= 0.
    if weight_decay > 0. and 'weight_decay' not in inspect.getfullargspec(optimizer).args:
        optimizer = optimizer(neural_network.parameters(), lr=learning_rate)
        logger.warning(f'Optimizer {optimizer.__class__.__name__} does not support weight decay! '
                       'Continuing without regularization!')
    elif 'weight_decay' in inspect.getfullargspec(optimizer).args:
        optimizer = optimizer(neural_network.parameters(), lr=learning_rate,
                              weight_decay=weight_decay)
    else:
        optimizer = optimizer(neural_network.parameters(), lr=learning_rate)

    if 'es_scheduler_params' in training_parameters:
        es_scheduler = EarlyStoppingScheduler(len(training_data) + len(validation_data),
                                              **training_parameters['es_scheduler_params'])
    else:
        es_scheduler = EarlyStoppingScheduler(len(training_data) + len(validation_data))
    if training_parameters.get('lr_scheduler'):
        lr_scheduler = training_parameters['lr_scheduler'](optimizer, **training_parameters['lr_scheduler_params'])

    # create the training and validation sets as well as the respective data loaders
    training_dataset = CustomDataset(training_data)
    validation_dataset = CustomDataset(validation_data)
    training_loader = utils.data.DataLoader(training_dataset, batch_size=batch_size)
    validation_loader = utils.data.DataLoader(validation_dataset, batch_size=batch_size)
    dataloaders = {'train':  training_loader, 'val': validation_loader}

    phases = ['train', 'val']

    logger.info('Starting optimization procedure ...')

    if 'min_inputs' in scaling_parameters and 'max_inputs' in scaling_parameters:
        min_inputs = scaling_parameters['min_inputs']
        max_inputs = scaling_parameters['max_inputs']
    else:
        min_inputs = None
        max_inputs = None
    if 'min_targets' in scaling_parameters and 'max_targets' in scaling_parameters:
        min_targets = scaling_parameters['min_targets']
        max_targets = scaling_parameters['max_targets']
    else:
        min_targets = None
        max_targets = None

    # perform optimization procedure
    for epoch in range(epochs):
        losses = {'full': 0.}

        # alternate between training and validation phase
        for phase in phases:
            if phase == 'train':
                neural_network.train()
            else:
                neural_network.eval()

            running_loss = 0.0

            # iterate over batches
            for batch in dataloaders[phase]:
                # scale inputs and outputs if desired
                if min_inputs is not None and max_inputs is not None:
                    diff = max_inputs - min_inputs
                    diff[diff == 0] = 1.
                    inputs = (batch[0] - min_inputs) / diff
                else:
                    inputs = batch[0]
                if min_targets is not None and max_targets is not None:
                    diff = max_targets - min_targets
                    diff[diff == 0] = 1.
                    targets = (batch[1] - min_targets) / diff
                else:
                    targets = batch[1]

                with torch.set_grad_enabled(phase == 'train'):
                    def closure(inputs=inputs, targets=targets):
                        if torch.is_grad_enabled():
                            optimizer.zero_grad()
                        outputs = neural_network(inputs)
                        loss = loss_function(outputs, targets)
                        if loss.requires_grad:
                            loss.backward()
                        return loss

                    # perform optimization step
                    if phase == 'train':
                        optimizer.step(closure)

                    # compute loss of current batch
                    loss = closure()

                # update overall absolute loss
                running_loss += loss.item() * len(batch[0])

            # compute average loss
            if len(dataloaders[phase].dataset) > 0:
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
            else:
                epoch_loss = np.inf

            losses[phase] = epoch_loss

            losses['full'] += running_loss

            if log_loss_frequency > 0 and epoch % log_loss_frequency == 0:
                logger.info(f'Epoch {epoch}: Current {phase} loss of {losses[phase]:.3e}')

            if training_parameters.get('lr_scheduler'):
                lr_scheduler.step()

            # check for early stopping
            if phase == 'val' and es_scheduler(losses, neural_network):
                logger.info(f'Stopping training process early after {epoch + 1} epochs with validation loss '
                            f'of {es_scheduler.best_losses["val"]:.3e} ...')
                return es_scheduler.best_neural_network, es_scheduler.best_losses

    return es_scheduler.best_neural_network, es_scheduler.best_losses


def multiple_restarts_training(training_data, validation_data, neural_network,
                               target_loss=None, max_restarts=10, log_loss_frequency=0,
                               training_parameters={}, scaling_parameters={}):
    """Algorithm that performs multiple restarts of neural network training.

    This method either performs a predefined number of restarts and returns
    the best trained network or tries to reach a given target loss and
    stops training when the target loss is reached.

    See :func:`train_neural_network` for more information on the parameters.

    Parameters
    ----------
    training_data
        Data to use during the training phase.
    validation_data
        Data to use during the validation phase.
    neural_network
        The neural network to train (parameters will be reset after each
        restart).
    target_loss
        Loss to reach during training (if `None`, the network with the
        smallest loss is returned).
    max_restarts
        Maximum number of restarts to perform.
    log_loss_frequency
        Frequency of epochs in which to log the current validation and
        training loss. If `0`, no intermediate logging of losses is done.
    training_parameters
        Additional parameters for the training algorithm,
        see :func:`train_neural_network` for more information.
    scaling_parameters
        Additional parameters for scaling inputs respectively outputs,
        see :func:`train_neural_network` for more information.

    Returns
    -------
    best_neural_network
        The best trained neural network.
    losses
        The corresponding losses.

    Raises
    ------
    NeuralNetworkTrainingError
        Raised if prescribed loss can not be reached within the given number
        of restarts.
    """
    assert isinstance(training_parameters, dict)
    assert isinstance(max_restarts, int)
    assert max_restarts >= 0

    logger = getLogger('pymor.algorithms.neural_network.multiple_restarts_training')

    torch.manual_seed(get_seed_seq().spawn(1)[0].generate_state(1).item())

    # in case no training data is provided, return a neural network
    # that always returns zeros independent of the input
    if len(training_data) == 0 or len(training_data[0]) == 0:
        for layers in neural_network.children():
            for layer in layers:
                torch.nn.init.zeros_(layer.weight)
                layer.bias.data.fill_(0.)
        return neural_network, {'full': None, 'train': None, 'val': None}

    if target_loss:
        logger.info(f'Performing up to {max_restarts} restart{"s" if max_restarts > 1 else ""} '
                    f'to train a neural network with a loss below {target_loss:.3e} ...')
    else:
        logger.info(f'Performing up to {max_restarts} restart{"s" if max_restarts > 1 else ""} '
                    'to find the neural network with the lowest loss ...')

    with logger.block('Training neural network #0 ...'):
        best_neural_network, losses = train_neural_network(training_data, validation_data,
                                                           neural_network, training_parameters,
                                                           scaling_parameters, log_loss_frequency)

    # perform multiple restarts
    for run in range(1, max_restarts + 1):

        if target_loss and losses['full'] <= target_loss:
            logger.info(f'Finished training after {run - 1} restart{"s" if run - 1 != 1 else ""}, '
                        f'found neural network with loss of {losses["full"]:.3e} ...')
            return neural_network, losses

        with logger.block(f'Training neural network #{run} ...'):
            # reset parameters of layers to start training with a new and untrained network
            def reset_parameters_nn(component):
                if hasattr(component, 'children'):
                    for child in component.children():
                        reset_parameters_nn(child)
                try:
                    for child in component:
                        reset_parameters_nn(child)
                except TypeError:
                    pass
                if hasattr(component, 'reset_parameters'):
                    component.reset_parameters()

            reset_parameters_nn(neural_network)

            # perform training
            current_nn, current_losses = train_neural_network(training_data, validation_data,
                                                              neural_network, training_parameters,
                                                              scaling_parameters, log_loss_frequency)

        if current_losses['full'] < losses['full']:
            logger.info(f'Found better neural network (loss of {current_losses["full"]:.3e} '
                        f'instead of {losses["full"]:.3e}) ...')
            best_neural_network = current_nn
            losses = current_losses
        else:
            logger.info(f'Rejecting neural network with loss of {current_losses["full"]:.3e} '
                        f'(instead of {losses["full"]:.3e}) ...')

    if target_loss and losses['full'] > target_loss:
        raise NeuralNetworkTrainingError(f'Could not find neural network with prescribed loss of '
                                         f'{target_loss:.3e} (best one found was {losses["full"]:.3e})!')
    logger.info(f'Found neural network with error of {losses["full"]:.3e} ...')
    return best_neural_network, losses
