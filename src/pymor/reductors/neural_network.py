# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Remark on the documentation:

Due to an issue in autoapi, the classes `NeuralNetworkStatefreeOutputReductor`,
`NeuralNetworkInstationaryReductor`, `NeuralNetworkInstationaryStatefreeOutputReductor`,
`EarlyStoppingScheduler` and `CustomDataset` do not appear in the documentation,
see https://github.com/pymor/pymor/issues/1343.
"""

from pymor.core.config import config


if config.HAVE_TORCH:
    from numbers import Number

    import numpy as np

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils as utils

    from pymor.algorithms.pod import pod
    from pymor.algorithms.projection import project
    from pymor.core.base import BasicObject
    from pymor.core.exceptions import NeuralNetworkTrainingFailed
    from pymor.core.logger import getLogger
    from pymor.models.neural_network import (FullyConnectedNN, NeuralNetworkModel,
                                             NeuralNetworkStatefreeOutputModel,
                                             NeuralNetworkInstationaryModel,
                                             NeuralNetworkInstationaryStatefreeOutputModel)

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
            The full-order |Model| to reduce.
        training_set
            Set of |parameter values| to use for POD and training of the
            neural network.
        validation_set
            Set of |parameter values| to use for validation in the training
            of the neural network.
        validation_ratio
            Fraction of the training set to use for validation in the training
            of the neural network (only used if no validation set is provided).
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
            the training set should not exceed the error of projecting onto the basis.
            If `None`, the neural network with smallest validation error is
            used to build the ROM.
            If a tolerance is prescribed, the mean squared error of the neural
            network on the training set should not exceed this threshold.
            Training is interrupted if a neural network that undercuts the
            error tolerance is found.
        """

        def __init__(self, fom, training_set, validation_set=None, validation_ratio=0.1,
                     basis_size=None, rtol=0., atol=0., l2_err=0., pod_params=None,
                     ann_mse='like_basis'):
            assert 0 < validation_ratio < 1 or validation_set
            self.__auto_init(locals())

        def reduce(self, hidden_layers='[(N+P)*3, (N+P)*3]', activation_function=torch.tanh,
                   optimizer=optim.LBFGS, epochs=1000, batch_size=20, learning_rate=1.,
                   restarts=10, seed=0):
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
            restarts
                Number of restarts of the training algorithm. Since the training
                results highly depend on the initial starting point, i.e. the
                initial weights and biases, it is advisable to train multiple
                neural networks by starting with different initial values and
                choose that one performing best on the validation set.
            seed
                Seed to use for various functions in PyTorch. Using a fixed seed,
                it is possible to reproduce former results.

            Returns
            -------
            rom
                Reduced-order |NeuralNetworkModel|.
            """
            assert restarts > 0
            assert epochs > 0
            assert batch_size > 0
            assert learning_rate > 0.

            # set a seed for the PyTorch initialization of weights and biases
            # and further PyTorch methods
            torch.manual_seed(seed)

            # build a reduced basis using POD and compute training data
            if not hasattr(self, 'training_data'):
                self.compute_training_data()

            layer_sizes = self._compute_layer_sizes(hidden_layers)

            # compute validation data
            if not hasattr(self, 'validation_data'):
                with self.logger.block('Computing validation snapshots ...'):

                    if self.validation_set:
                        self.validation_data = []
                        for mu in self.validation_set:
                            sample = self._compute_sample(mu)
                            self.validation_data.extend(sample)
                    else:
                        number_validation_snapshots = int(len(self.training_data)*self.validation_ratio)
                        # randomly shuffle training data before splitting into two sets
                        np.random.shuffle(self.training_data)
                        # split training data into validation and training set
                        self.validation_data = self.training_data[0:number_validation_snapshots]
                        self.training_data = self.training_data[number_validation_snapshots+1:]

            # run the actual training of the neural network
            with self.logger.block('Training of neural network ...'):
                target_loss = self._compute_target_loss()
                # set parameters for neural network and training
                neural_network_parameters = {'layer_sizes': layer_sizes,
                                             'activation_function': activation_function}
                training_parameters = {'optimizer': optimizer, 'epochs': epochs,
                                       'batch_size': batch_size, 'learning_rate': learning_rate}

                self.logger.info('Initializing neural network ...')
                # initialize the neural network
                neural_network = FullyConnectedNN(**neural_network_parameters).double()
                # run training algorithm with multiple restarts
                self.neural_network, self.losses = multiple_restarts_training(self.training_data, self.validation_data,
                                                                              neural_network, target_loss, restarts,
                                                                              training_parameters, seed)

            self._check_tolerances()

            return self._build_rom()

        def compute_training_data(self):
            """Compute a reduced basis using proper orthogonal decomposition."""
            # compute snapshots for POD and training of neural networks
            with self.logger.block('Computing training snapshots ...'):
                U = self.fom.solution_space.empty()
                for mu in self.training_set:
                    U.append(self.fom.solve(mu))

            # compute reduced basis via POD
            with self.logger.block('Building reduced basis ...'):
                self.reduced_basis, svals = pod(U, modes=self.basis_size, rtol=self.rtol / 2.,
                                                atol=self.atol / 2., l2_err=self.l2_err / 2.,
                                                **(self.pod_params or {}))

            # compute training samples
            with self.logger.block('Computing training samples ...'):
                self.training_data = []
                for mu, u in zip(self.training_set, U):
                    sample = self._compute_sample(mu, u)
                    self.training_data.extend(sample)

            # compute mean square loss
            self.mse_basis = (sum(U.norm2()) - sum(svals**2)) / len(U)

        def _compute_sample(self, mu, u=None):
            """Transform parameter and corresponding solution to |NumPy arrays|."""
            # determine the coefficients of the full-order solutions in the reduced basis to obtain
            # the training data
            if u is None:
                u = self.fom.solve(mu)
            return [(mu, self.reduced_basis.inner(u)[:, 0])]

        def _compute_layer_sizes(self, hidden_layers):
            """Compute the number of neurons in the layers of the neural network."""
            # determine the numbers of neurons in the hidden layers
            if isinstance(hidden_layers, str):
                hidden_layers = eval(hidden_layers, {'N': len(self.reduced_basis), 'P': self.fom.parameters.dim})
            # input and output size of the neural network are prescribed by the
            # dimension of the parameter space and the reduced basis size
            assert isinstance(hidden_layers, list)
            return [self.fom.parameters.dim, ] + hidden_layers + [len(self.reduced_basis), ]

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
                        raise NeuralNetworkTrainingFailed('Could not train a neural network that '
                                                          'guarantees prescribed tolerance!')
                elif self.ann_mse == 'like_basis':
                    if self.losses['full'] > self.mse_basis:
                        raise NeuralNetworkTrainingFailed('Could not train a neural network with an error as small as '
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
            with self.logger.block('Building ROM ...'):
                projected_output_functional = (project(self.fom.output_functional, None, self.reduced_basis)
                                               if self.fom.output_functional else None)
                rom = NeuralNetworkModel(self.neural_network, parameters=self.fom.parameters,
                                         output_functional=projected_output_functional,
                                         name=f'{self.fom.name}_reduced')

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
            The full-order |Model| to reduce.
        training_set
            Set of |parameter values| to use for POD and training of the
            neural network.
        validation_set
            Set of |parameter values| to use for validation in the training
            of the neural network.
        validation_ratio
            Fraction of the training set to use for validation in the training
            of the neural network (only used if no validation set is provided).
        validation_loss
            The validation loss to reach during training. If `None`, the neural
            network with the smallest validation loss is returned.
        """

        def __init__(self, fom, training_set, validation_set=None, validation_ratio=0.1,
                     validation_loss=None):
            assert 0 < validation_ratio < 1 or validation_set
            self.__auto_init(locals())

        def compute_training_data(self):
            """Compute the training samples (the outputs to the parameters of the training set)."""
            with self.logger.block('Computing training samples ...'):
                self.training_data = []
                for mu in self.training_set:
                    sample = self._compute_sample(mu)
                    self.training_data.extend(sample)

        def _compute_sample(self, mu):
            """Transform parameter and corresponding output to tensors."""
            return [(mu, self.fom.output(mu).flatten())]

        def _compute_layer_sizes(self, hidden_layers):
            """Compute the number of neurons in the layers of the neural network."""
            # determine the numbers of neurons in the hidden layers
            if isinstance(hidden_layers, str):
                hidden_layers = eval(hidden_layers, {'N': self.fom.dim_output, 'P': self.fom.parameters.dim})
            # input and output size of the neural network are prescribed by the
            # dimension of the parameter space and the output dimension
            assert isinstance(hidden_layers, list)
            return [self.fom.parameters.dim, ] + hidden_layers + [self.fom.dim_output, ]

        def _compute_target_loss(self):
            """Compute target loss depending on value of `ann_mse`."""
            return self.validation_loss

        def _check_tolerances(self):
            """Check if trained neural network is sufficient to guarantee certain error bounds."""
            self.logger.info('Using neural network with smallest validation error ...')
            self.logger.info(f'Finished training with a validation loss of {self.losses["val"]} ...')

        def _build_rom(self):
            """Construct the reduced order model."""
            with self.logger.block('Building ROM ...'):
                rom = NeuralNetworkStatefreeOutputModel(self.neural_network, self.fom.parameters,
                                                        name=f'{self.fom.name}_output_reduced')

            return rom

    class NeuralNetworkInstationaryReductor(NeuralNetworkReductor):
        """Reduced Basis reductor for instationary problems relying on artificial neural networks.

        This is a reductor that constructs a reduced basis using proper
        orthogonal decomposition and trains a neural network that approximates
        the mapping from parameter and time space to coefficients of the
        full-order solution in the reduced basis.
        The approach is described in :cite:`WHR19`.

        Parameters
        ----------
        fom
            The full-order |Model| to reduce.
        training_set
            Set of |parameter values| to use for POD and training of the
            neural network.
        validation_set
            Set of |parameter values| to use for validation in the training
            of the neural network.
        validation_ratio
            Fraction of the training set to use for validation in the training
            of the neural network (only used if no validation set is provided).
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
            the training set should not exceed the error of projecting onto the basis.
            If `None`, the neural network with smallest validation error is
            used to build the ROM.
            If a tolerance is prescribed, the mean squared error of the neural
            network on the training set should not exceed this threshold.
            Training is interrupted if a neural network that undercuts the
            error tolerance is found.
        """

        def __init__(self, fom, training_set, validation_set=None, validation_ratio=0.1,
                     basis_size=None, rtol=0., atol=0., l2_err=0., pod_params=None,
                     ann_mse='like_basis'):
            assert 0 < validation_ratio < 1 or validation_set
            self.__auto_init(locals())

        def compute_training_data(self):
            """Compute a reduced basis using proper orthogonal decomposition."""
            # compute snapshots for POD and training of neural networks
            with self.logger.block('Computing training snapshots ...'):
                U = self.fom.solution_space.empty()
                for mu in self.training_set:
                    u = self.fom.solve(mu)
                    if hasattr(self, 'nt'):
                        assert self.nt == len(u)
                    else:
                        self.nt = len(u)
                    U.append(u)

            # compute reduced basis via POD
            with self.logger.block('Building reduced basis ...'):
                self.reduced_basis, svals = pod(U, modes=self.basis_size, rtol=self.rtol / 2.,
                                                atol=self.atol / 2., l2_err=self.l2_err / 2.,
                                                **(self.pod_params or {}))

            # compute training samples
            with self.logger.block('Computing training samples ...'):
                self.training_data = []
                for i, mu in enumerate(self.training_set):
                    sample = self._compute_sample(mu, U[i*self.nt:(i+1)*self.nt])
                    self.training_data.extend(sample)

            # compute mean square loss
            self.mse_basis = (sum(U.norm2()) - sum(svals**2)) / len(U)

        def _compute_sample(self, mu, u=None):
            """Transform parameter and corresponding solution to |NumPy arrays|.

            This function takes care of including the time instances in the inputs.
            """
            if u is None:
                u = self.fom.solve(mu)

            parameters_with_time = [mu.with_(t=t) for t in np.linspace(0, self.fom.T, self.nt)]

            samples = [(mu, self.reduced_basis.inner(u_t)[:, 0])
                       for mu, u_t in zip(parameters_with_time, u)]

            return samples

        def _compute_layer_sizes(self, hidden_layers):
            """Compute the number of neurons in the layers of the neural network
            (make sure to increase the input dimension to account for the time).
            """
            # determine the numbers of neurons in the hidden layers
            if isinstance(hidden_layers, str):
                hidden_layers = eval(hidden_layers, {'N': len(self.reduced_basis), 'P': self.fom.parameters.dim})
            # input and output size of the neural network are prescribed by the
            # dimension of the parameter space and the reduced basis size
            assert isinstance(hidden_layers, list)
            return [self.fom.parameters.dim + 1, ] + hidden_layers + [len(self.reduced_basis), ]

        def _build_rom(self):
            """Construct the reduced order model."""
            with self.logger.block('Building ROM ...'):
                projected_output_functional = (project(self.fom.output_functional, None, self.reduced_basis)
                                               if self.fom.output_functional else None)
                rom = NeuralNetworkInstationaryModel(self.fom.T, self.nt, self.neural_network,
                                                     parameters=self.fom.parameters,
                                                     output_functional=projected_output_functional,
                                                     name=f'{self.fom.name}_reduced')

            return rom

    class NeuralNetworkInstationaryStatefreeOutputReductor(NeuralNetworkStatefreeOutputReductor):
        """Output reductor relying on artificial neural networks.

        This is a reductor that trains a neural network that approximates
        the mapping from parameter space to output space.

        Parameters
        ----------
        fom
            The full-order |Model| to reduce.
        nt
            Number of time steps in the reduced order model (does not have to
            coincide with the number of time steps in the full order model).
        training_set
            Set of |parameter values| to use for POD and training of the
            neural network.
        validation_set
            Set of |parameter values| to use for validation in the training
            of the neural network.
        validation_ratio
            Fraction of the training set to use for validation in the training
            of the neural network (only used if no validation set is provided).
        validation_loss
            The validation loss to reach during training. If `None`, the neural
            network with the smallest validation loss is returned.
        """

        def __init__(self, fom, nt, training_set, validation_set=None, validation_ratio=0.1,
                     validation_loss=None):
            assert 0 < validation_ratio < 1 or validation_set
            self.__auto_init(locals())

        def _compute_sample(self, mu):
            """Transform parameter and corresponding output to |NumPy arrays|.

            This function takes care of including the time instances in the inputs.
            """
            output_trajectory = self.fom.output(mu)
            output_size = output_trajectory.shape[0]
            samples = [(mu.with_(t=t), output.flatten())
                       for t, output in zip(np.linspace(0, self.fom.T, output_size), output_trajectory)]

            return samples

        def _compute_layer_sizes(self, hidden_layers):
            """Compute the number of neurons in the layers of the neural network."""
            # determine the numbers of neurons in the hidden layers
            if isinstance(hidden_layers, str):
                hidden_layers = eval(hidden_layers, {'N': self.fom.dim_output, 'P': self.fom.parameters.dim})
            # input and output size of the neural network are prescribed by the
            # dimension of the parameter space and the output dimension
            assert isinstance(hidden_layers, list)
            return [self.fom.parameters.dim + 1, ] + hidden_layers + [self.fom.dim_output, ]

        def _build_rom(self):
            """Construct the reduced order model."""
            with self.logger.block('Building ROM ...'):
                rom = NeuralNetworkInstationaryStatefreeOutputModel(self.fom.T, self.nt, self.neural_network,
                                                                    self.fom.parameters,
                                                                    name=f'{self.fom.name}_output_reduced')

            return rom

    class EarlyStoppingScheduler(BasicObject):
        """Class for performing early stopping in training of neural networks.

        If the validation loss does not decrease over a certain amount of epochs, the
        training should be aborted to avoid overfitting the training data.
        This class implements an early stopping scheduler that recommends to stop the
        training process if the validation loss did not decrease by at least `delta`
        over `patience` epochs.

        Parameters
        ----------
        size_training_validation_set
            Size of both, training and validation set together.
        patience
            Number of epochs of non-decreasing validation loss allowed, before early
            stopping the training process.
        delta
            Minimal amount of decrease in the validation loss that is required to reset
            the counter of non-decreasing epochs.
        """

        def __init__(self, size_training_validation_set, patience=10, delta=0.):
            self.__auto_init(locals())

            self.best_losses = None
            self.best_neural_network = None
            self.counter = 0

        def __call__(self, losses, neural_network=None):
            """Returns `True` if early stopping of training is suggested.

            Parameters
            ----------
            losses
                Dictionary of losses on the validation and the training set in
                the current epoch.
            neural_network
                Neural network that produces the current validation loss.

            Returns
            -------
            `True` if early stopping is suggested, `False` otherwise.
            """
            if self.best_losses is None:
                self.best_losses = losses
                self.best_losses['full'] /= self.size_training_validation_set
                self.best_neural_network = neural_network
            elif self.best_losses['val'] - self.delta <= losses['val']:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            else:
                self.best_losses = losses
                self.best_losses['full'] /= self.size_training_validation_set
                self.best_neural_network = neural_network
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
                             training_parameters={}):
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
            if not provided, 1 is taken as default value; thus far, no learning
            rate schedulers are supported in this implementation), and
            `'loss_function'` (a loss function from PyTorch; if not provided, the
            MSE loss is taken as default).

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

        for data in training_data, validation_data:
            assert isinstance(data, list)
            assert all(isinstance(datum, tuple) and len(datum) == 2 for datum in data)

        def prepare_datum(datum):
            if not (isinstance(datum, torch.DoubleTensor) or isinstance(datum, np.ndarray)):
                return datum.to_numpy()
            return datum

        training_data = [(prepare_datum(datum[0]), prepare_datum(datum[1])) for datum in training_data]
        validation_data = [(prepare_datum(datum[0]), prepare_datum(datum[1])) for datum in validation_data]

        optimizer = optim.LBFGS if 'optimizer' not in training_parameters else training_parameters['optimizer']
        epochs = 1000 if 'epochs' not in training_parameters else training_parameters['epochs']
        assert isinstance(epochs, int) and epochs > 0
        batch_size = 20 if 'batch_size' not in training_parameters else training_parameters['batch_size']
        assert isinstance(batch_size, int) and batch_size > 0
        learning_rate = 1. if 'learning_rate' not in training_parameters else training_parameters['learning_rate']
        assert learning_rate > 0.
        loss_function = (nn.MSELoss() if 'loss_function' not in training_parameters
                         else training_parameters['loss_function'])

        logger = getLogger('pymor.algorithms.neural_network.train_neural_network')

        # LBFGS-optimizer does not support mini-batching, so the batch size needs to be adjusted
        if optimizer == optim.LBFGS:
            batch_size = max(len(training_data), len(validation_data))

        # initialize optimizer and early stopping scheduler
        optimizer = optimizer(neural_network.parameters(), lr=learning_rate)
        early_stopping_scheduler = EarlyStoppingScheduler(len(training_data) + len(validation_data))

        # create the training and validation sets as well as the respective data loaders
        training_dataset = CustomDataset(training_data)
        validation_dataset = CustomDataset(validation_data)
        training_loader = utils.data.DataLoader(training_dataset, batch_size=batch_size)
        validation_loader = utils.data.DataLoader(validation_dataset, batch_size=batch_size)
        dataloaders = {'train':  training_loader, 'val': validation_loader}

        phases = ['train', 'val']

        logger.info('Starting optimization procedure ...')

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
                    inputs = batch[0]
                    targets = batch[1]

                    with torch.set_grad_enabled(phase == 'train'):
                        def closure():
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
                epoch_loss = running_loss / len(dataloaders[phase].dataset)

                losses[phase] = epoch_loss

                losses['full'] += running_loss

                # check for early stopping
                if phase == 'val' and early_stopping_scheduler(losses, neural_network):
                    logger.info(f'Stopping training process early after {epoch + 1} epochs with validation loss '
                                f'of {early_stopping_scheduler.best_losses["val"]:.3e} ...')
                    return early_stopping_scheduler.best_neural_network, early_stopping_scheduler.best_losses

        return early_stopping_scheduler.best_neural_network, early_stopping_scheduler.best_losses

    def multiple_restarts_training(training_data, validation_data, neural_network,
                                   target_loss=None, max_restarts=10,
                                   training_parameters={}, seed=None):
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
        target_loss
            Loss to reach during training (if `None`, the network with the
            smallest loss is returned).
        max_restarts
            Maximum number of restarts to perform.
        neural_network
            The neural network to train (parameters will be reset after each
            restart).

        Returns
        -------
        best_neural_network
            The best trained neural network.
        losses
            The corresponding losses.

        Raises
        ------
        NeuralNetworkTrainingFailed
            Raised if prescribed loss can not be reached within the given number
            of restarts.
        """
        assert isinstance(training_parameters, dict)
        assert isinstance(max_restarts, int) and max_restarts > 0

        logger = getLogger('pymor.algorithms.neural_network.multiple_restarts_training')

        # if applicable, set a common seed for the PyTorch initialization
        # of weights and biases and further PyTorch methods for all training runs
        if seed:
            torch.manual_seed(seed)

        if target_loss:
            logger.info(f'Performing up to {max_restarts} restart{"s" if max_restarts > 1 else ""} '
                        f'to train a neural network with a loss below {target_loss:.3e} ...')
        else:
            logger.info(f'Performing up to {max_restarts} restart{"s" if max_restarts > 1 else ""} '
                        'to find the neural network with the lowest loss ...')

        with logger.block('Training neural network #0 ...'):
            best_neural_network, losses = train_neural_network(training_data, validation_data,
                                                               neural_network, training_parameters)

        # perform multiple restarts
        for run in range(1, max_restarts + 1):

            if target_loss and losses['full'] <= target_loss:
                logger.info(f'Finished training after {run - 1} restart{"s" if run - 1 != 1 else ""}, '
                            f'found neural network with loss of {losses["full"]:.3e} ...')
                return neural_network, losses

            with logger.block(f'Training neural network #{run} ...'):
                # reset parameters of layers to start training with a new and untrained network
                for layers in neural_network.children():
                    for layer in layers:
                        if hasattr(layer, 'reset_parameters'):
                            layer.reset_parameters()
                # perform training
                current_nn, current_losses = train_neural_network(training_data, validation_data,
                                                                  neural_network, training_parameters)

            if current_losses['full'] < losses['full']:
                logger.info(f'Found better neural network (loss of {current_losses["full"]:.3e} '
                            f'instead of {losses["full"]:.3e}) ...')
                best_neural_network = current_nn
                losses = current_losses
            else:
                logger.info(f'Rejecting neural network with loss of {current_losses["full"]:.3e} '
                            f'(instead of {losses["full"]:.3e}) ...')

        if target_loss:
            raise NeuralNetworkTrainingFailed(f'Could not find neural network with prescribed loss of '
                                              f'{target_loss:.3e} (best one found was {losses["full"]:.3e})!')
        logger.info(f'Found neural network with error of {losses["full"]:.3e} ...')
        return best_neural_network, losses
