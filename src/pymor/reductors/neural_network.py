# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_TORCH:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils as utils

    from pymor.algorithms.pod import pod
    from pymor.core.base import BasicObject
    from pymor.models.neural_network import NeuralNetworkModel


    class FullyConnectedNN(nn.Module, BasicObject):
        """Class for neural networks with fully connected layers.

        This class implements neural networks consisting of linear and fully connected layers.
        Furthermore, the same activation function is used between each layer, except for the
        last one where no activation function is applied.

        Parameters
        ----------
        layers_sizes
            List of sizes (i.e. number of neurons) for the layers of the neural network.
        activation_function
            Function to use as activation function between the single layers.
        """

        def __init__(self, layers_sizes, activation_function=torch.tanh):
            super().__init__()

            if layers_sizes is None or not len(layers_sizes) > 1 or not all(size >= 1 for size in layers_sizes):
                raise ValueError

            self.input_dimension = layers_sizes[0]
            self.output_dimension = layers_sizes[-1]

            self.layers = nn.ModuleList()
            self.layers.extend([nn.Linear(int(layers_sizes[i]), int(layers_sizes[i+1])) for i in range(len(layers_sizes) - 1)])

            self.activation_function = activation_function

            if not self.logging_disabled:
                self._log_parameters()

        def forward(self, x):
            """Performs the forward pass through the neural network.

            Applies the weights in the linear layers and passes the outcomes to the
            activation function.

            Parameters
            ----------
            x
                Input for the neural network.

            Returns
            -------
            The output of the neural network for the input x.
            """
            for i in range(len(self.layers) - 1):
                x = self.activation_function(self.layers[i](x))
            return self.layers[len(self.layers)-1](x)

        def _log_parameters(self):
            self.logger.info(f'Architecture of the neural network:\n{self}')


    class EarlyStoppingScheduler(BasicObject):
        """Class for performing early stopping in training of neural networks.

        If the validation loss does not decrease over a certain amount of epochs, the
        training should be aborted to avoid overfitting the training data.
        This class implements an early stopping scheduler that recommends to stop the
        training process if the validation loss did not decrease by at least `delta`
        over `patience` epochs.

        Parameters
        ----------
        patience
            Number of epochs of non-decreasing validation loss allowed, before early
            stopping the training process.
        delta
            Minimal amount of decrease in the validation loss that is required to reset
            the counter of non-decreasing epochs.
        """

        def __init__(self, patience=10, delta=0.):
            self.__auto_init(locals())

            self.best_loss = None
            self.best_neural_network = None
            self.counter = 0

        def __call__(self, validation_loss, neural_network=None):
            """Returns `True` if early stopping of training is suggested.

            Parameters
            ----------
            validation_loss
                Loss on the validation set in the current epoch.
            neural_network
                Neural network that produces the current validation loss.

            Returns
            -------
            `True` if early stopping is suggested, `False` otherwise.
            """
            if self.best_loss is None:
                self.best_loss = validation_loss
                self.best_neural_network = neural_network
            elif self.best_loss - self.delta <= validation_loss:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            else:
                self.best_loss = validation_loss
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


    class NeuralNetworkReductor(BasicObject):
        """Reduced Basis reductor relying on artificial neural networks.

        This is a reductor that constructs a reduced basis using proper
        orthogonal decomposition and trains a neural networks that approximates
        the mapping from parameter space to coefficients of the full-order
        solution in the reduced basis by means of a neural network.
        The approach is described in [HU18]_.

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
        basis_size
            Desired size of the reduced basis. If `None`, rtol or atol most
            be provided.
        rtol
            Relative tolerance the model should guarantee on the validation
            set.
        atol
            Absolute tolerance the model should guarantee on the validation
            set.
        pod_params
            Additional parameters for the POD-method.
        hidden_layers
            Number of neurons in the hidden layers. Can either be fixed or
            depend on the reduced basis size `N` and dimension of the
            parameter space `P`.
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
        """

        def __init__(
            self,
            fom,
            training_set,
            validation_set,
            basis_size=None,
            rtol=None,
            atol=None,
            pod_params=None,
            hidden_layers='[(N+P)*2, (N+P)*2]',
            activation_function=torch.tanh,
            optimizer=optim.LBFGS,
            epochs=1000,
            batch_size=20,
            learning_rate=1.,
            restarts=10
        ):
            assert restarts > 0
            assert epochs > 0
            assert batch_size > 0
            assert learning_rate > 0.
            self.__auto_init(locals())

        def reduce(self, seed=0):
            # set a seed for the PyTorch initialization of weights and biases and further PyTorch methods
            torch.manual_seed(seed)

            # build a reduced basis using POD and compute training data
            self.reduced_basis = self.build_basis()

            # determine the numbers of neurons in the hidden layers
            layers = eval(self.hidden_layers, {'N': len(self.reduced_basis), 'P': self.fom.parameters.dim})
            # input and output size of the neural network are prescribed by the dimension of the parameter space and the reduced basis size
            self.layers = [len(self.fom.parameters),] + layers + [len(self.reduced_basis),]

            # compute validation data
            with self.logger.block('Computing validation snapshots ...'):

                validation_data = []
                for mu in self.validation_set:
                    mu_tensor = torch.DoubleTensor(mu.to_numpy())
                    u = self.fom.solve(mu)
                    u_tensor = torch.DoubleTensor(self.reduced_basis.inner(u)[:,0])
                    validation_data.append((mu_tensor, u_tensor))
                self.validation_data = validation_data

            # run the actual training of the neural network
            with self.logger.block(f'Performing {self.restarts} restarts for training ...'):

                for run in range(self.restarts):
                    neural_network, loss = self._train()
                    if not hasattr(self, 'validation_loss') or loss < self.validation_loss:
                        self.validation_loss = loss
                        self.neural_network = neural_network

            self.logger.info(f'Finished training with a validation loss of {self.validation_loss} ...')

            return self._build_rom()

        def _train(self):
            """Perform a single training iteration and return the resulting neural network."""
            if not hasattr(self, 'training_data'):
                self.logger.error('No data for training available ...')

            if not hasattr(self, 'validation_data'):
                self.logger.error('No data for validation available ...')

            # LBFGS-optimizer does not support mini-batching, so the batch size needs to be adjusted
            if self.optimizer == optim.LBFGS:
                self.batch_size = max(len(self.training_data), len(self.validation_data))

            with self.logger.block('Training the neural network ...'):

                # initialize the neural network
                neural_network = FullyConnectedNN(self.layers,
                                                  activation_function=self.activation_function).double()

                # initialize the optimizer
                optimizer = self.optimizer(neural_network.parameters(),
                                           lr=self.learning_rate)

                loss_function = nn.MSELoss()
                early_stopping_scheduler = EarlyStoppingScheduler()

                # create the training and validation sets as well as the respective data loaders
                training_dataset = CustomDataset(self.training_data)
                validation_dataset = CustomDataset(self.validation_data)
                phases = ['train', 'val']
                training_loader = utils.data.DataLoader(training_dataset,
                                                        batch_size=self.batch_size)
                validation_loader = utils.data.DataLoader(validation_dataset,
                                                          batch_size=self.batch_size)
                dataloaders = {'train':  training_loader, 'val': validation_loader}

                # perform optimization procedure
                for epoch in range(self.epochs):
                    losses = {}

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

                        # check for early stopping
                        if phase == 'val' and early_stopping_scheduler(losses['val'], neural_network):
                            if not self.logging_disabled:
                                self.logger.info(f'Early stopping training process after {epoch + 1} epochs ...')
                                self.logger.info(f'Minimum validation loss: {early_stopping_scheduler.best_loss}')
                            return early_stopping_scheduler.best_neural_network, early_stopping_scheduler.best_loss

            return early_stopping_scheduler.best_neural_network, early_stopping_scheduler.best_loss

        def _build_rom(self):
            with self.logger.block('Building ROM ...'):
                rom = self.build_rom()
                rom = rom.with_(name=f'{self.fom.name}_reduced')
                rom.disable_logging()

            return rom

        def build_rom(self):
            """Construct the reduced order model."""
            return NeuralNetworkModel(self.neural_network)

        def build_basis(self):
            """Compute a reduced basis using proper orthogonal decomposition."""
            snapshots = []
            self.training_data = []

            with self.logger.block('Building reduced basis ...'):

                # compute snapshots for POD and training of neural networks
                with self.logger.block('Computing training snapshots ...'):
                    U = self.fom.solution_space.empty()
                    for mu in self.training_set:
                        u = self.fom.solve(mu)
                        U.append(u)
                        snapshots.append({'mu': mu, 'u_full': u})

                # compute reduced basis via POD
                reduced_basis, _ = pod(U, modes=self.basis_size, rtol=self.rtol,
                                       atol=self.atol, **(self.pod_params or {}))

                # determine the coefficients of the full-order solutions in the reduced basis to obtain the training data; convert everything into tensors that are compatible with PyTorch
                for v in snapshots:
                    mu_tensor = torch.DoubleTensor(v['mu'].to_numpy())
                    u_tensor = torch.DoubleTensor(reduced_basis.inner(v['u_full'])[:,0])
                    self.training_data.append((mu_tensor, u_tensor))

            return reduced_basis

        def reconstruct(self, u):
            """Reconstruct high-dimensional vector from reduced vector `u`."""
            assert hasattr(self, 'reduced_basis')
            return self.reduced_basis.lincomb(u.to_numpy())
