# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_TORCH:
    import numpy as np

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils as utils

    from pymor.algorithms.pod import pod
    from pymor.core.base import BasicObject
    from pymor.models.neural_network import NeuralNetworkModel


    class FullyConnectedNN(nn.Module, BasicObject):
        def __init__(self, layers_sizes, activation_function=torch.tanh):
            super().__init__()

            if layers_sizes is None or not len(layers_sizes) > 1 or not all(size >= 1 for size in layers_sizes):
                raise ValueError

            self.layers = nn.ModuleList()
            self.layers.extend([nn.Linear(int(layers_sizes[i]), int(layers_sizes[i+1])) for i in range(len(layers_sizes) - 1)])

            self.activation_function = activation_function

            if not self.logging_disabled:
                self._log_parameters()

        def forward(self, x):
            for i in range(len(self.layers) - 1):
                x = self.activation_function(self.layers[i](x))
            return self.layers[len(self.layers)-1](x)

        def _log_parameters(self):
            self.logger.info(f'Architecture of the neural network:\n{self}')


    class EarlyStoppingScheduler(BasicObject):

        def __init__(self, patience=100, delta=0., maximum_loss=None):
            self.__auto_init(locals())

            self.best_loss = None
            self.counter = 0

        def __call__(self, validation_loss):
            if self.best_loss is None:
                self.best_loss = validation_loss
            elif self.best_loss - self.delta <= validation_loss:
                self.counter += 1
                if self.counter >= self.patience:
                    if self.maximum_loss:
                        if self.maximum_loss > self.best_loss:
                            return True
                    else:
                        return True
            else:
                self.best_loss = validation_loss
                self.counter = 0

            return False


    class NeuralNetworkReductor(BasicObject):

        def __init__(
            self,
            fom,
            training_set,
            validation_set,
            basis_size=None,
            basis_rtol=None,
            basis_atol=None,
            pod_params=None,
            hidden_layers='[(N+P)*2, (N+P)*2]',
            activation_function=torch.tanh,
            optimizer=optim.LBFGS,
            epochs=1000,
            batch_size=20,
            learning_rate=1.
        ):
            self.__auto_init(locals())

        def reduce(self):
            self.reduced_basis = self.build_basis()

            layers = eval(self.hidden_layers, {'N': len(self.reduced_basis), 'P': len(self.fom.parameters)})
            layers = [len(self.fom.parameters),] + layers + [len(self.reduced_basis),]

            with self.logger.block('Initialize neural network ...'):
                self.neural_network = FullyConnectedNN(layers, activation_function=self.activation_function).double()

            self.optimizer = self.optimizer(self.neural_network.parameters(), lr=self.learning_rate)

            self._train()

            return self._build_rom()

        def _train(self):
            with self.logger.block('Computing training snapshots ...'):
                training_data = []
                for mu in self.training_set:
                    training_data.append({'mu': mu, 'u_full': self.fom.solve(mu)})

            with self.logger.block('Computing validation snapshots ...'):
                validation_data = []
                for mu in self.validation_set:
                    validation_data.append({'mu': mu, 'u_full': self.fom.solve(mu)})

            if type(self.optimizer) == optim.LBFGS:
                self.batch_size = len(training_data)

            loss_function = nn.MSELoss()
            early_stopping_scheduler = EarlyStoppingScheduler()

            phases = ['train', 'val']
            training_loader = utils.data.DataLoader(training_data, batch_size=self.batch_size, collate_fn=lambda batch: self._prepare_batch(batch))
            validation_loader = utils.data.DataLoader(validation_data, batch_size=self.batch_size, collate_fn=lambda batch: self._prepare_batch(batch))
            dataloaders = {'train':  training_loader, 'val': validation_loader}

            with self.logger.block('Training the neural network ...'):

                for epoch in range(self.epochs):
                    losses = {}
                    for phase in phases:
                        if phase == 'train':
                            self.neural_network.train()
                        else:
                            self.neural_network.eval()

                        running_loss = 0.0

                        for batch in dataloaders[phase]:
                            inputs = torch.stack(batch['inputs'])
                            targets = torch.stack(batch['targets'])

                            with torch.set_grad_enabled(phase == 'train'):
                                def closure():
                                    if torch.is_grad_enabled():
                                        self.optimizer.zero_grad()
                                    outputs = self.neural_network(inputs)
                                    loss = loss_function(outputs, targets)
                                    if loss.requires_grad:
                                        loss.backward()
                                    return loss

                                if phase == 'train':
                                    self.optimizer.step(closure)

                                loss = closure()

                            running_loss += loss.item() * len(batch['inputs'])

                        epoch_loss = running_loss / len(dataloaders[phase].dataset)

                        losses[phase] = epoch_loss

                        if phase == 'val' and early_stopping_scheduler(losses['val']):
                            if not self.logging_disabled:
                                self.logger.info('Early stopping training process ...')
                                self.logger.info(f'Minimum validation loss: {early_stopping_scheduler.best_loss}')
                            return

        def _build_rom(self):
            with self.logger.block('Building ROM ...'):
                rom = self.build_rom()
                rom = rom.with_(name=f'{self.fom.name}_reduced')
                rom.disable_logging()

            return rom

        def build_rom(self):
            return NeuralNetworkModel(self.neural_network, self.reduced_basis)

        def build_basis(self):
            with self.logger.block('Building reduced basis ...'):

                with self.logger.block('Computing training snapshots ...'):
                    U = self.fom.solution_space.empty()
                    for mu in self.training_set:
                        U.append(self.fom.solve(mu))

                reduced_basis, _ = pod(U, modes=self.basis_size, rtol=self.basis_rtol, atol=self.basis_atol, **(self.pod_params or {}))

            return reduced_basis

        def _prepare_batch(self, batch):
            inputs = []
            targets = []

            for item in batch:
                u_proj = self.reduced_basis.inner(item['u_full'])[:,0]
                input_ = torch.DoubleTensor(np.fromiter(item['mu'].values(), dtype=float))
                target = torch.DoubleTensor(u_proj)
                inputs.append(input_)
                targets.append(target)

            return {'inputs': inputs, 'targets': targets}
