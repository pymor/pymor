# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('TORCH')

import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from pymor.algorithms.ml.nn.utils import CustomDataset, EarlyStoppingScheduler
from pymor.core.exceptions import NeuralNetworkTrainingError
from pymor.core.logger import getLogger
from pymor.tools.random import get_seed_seq


def train_neural_network(training_data, validation_data, neural_network,
                         training_parameters={}, log_loss_frequency=0):
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
        `'train'` (for the average loss on the training parameters), and `'val'`
        (for the average loss on the validation parameters).
    """
    assert isinstance(neural_network, nn.Module)
    assert isinstance(log_loss_frequency, int)

    for data in training_data, validation_data:
        assert isinstance(data, list)
        assert all(isinstance(datum, tuple) and len(datum) == 2 for datum in data)

    def prepare_datum(datum):
        if not (isinstance(datum, torch.DoubleTensor | np.ndarray)):
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

    # create the training and validation parameters as well as the respective data loaders
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
                # scale inputs and outputs if desired
                inputs = batch[0]
                targets = batch[1]

                with torch.set_grad_enabled(phase == 'train'):
                    def closure(inputs=inputs, targets=targets):
                        if torch.is_grad_enabled():
                            optimizer.zero_grad()
                        outputs = neural_network(inputs)
                        loss = loss_function(outputs, targets)
                        if loss.requires_grad:
                            loss.backward()
                        return loss.item()

                    # perform optimization step
                    if phase == 'train':
                        optimizer.step(closure)

                    # compute loss of current batch
                    loss = closure()

                # update overall absolute loss
                running_loss += loss * len(batch[0])

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
                               training_parameters={}):
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
                                                           log_loss_frequency)

    # perform multiple restarts
    for run in range(1, max_restarts + 1):

        if target_loss and losses['full'] <= target_loss:
            logger.info(f'Finished training after {run - 1} restart{"s" if run - 1 != 1 else ""}, '
                        f'found neural network with loss of {losses["full"]:.3e} ...')
            return best_neural_network, losses

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
                                                              log_loss_frequency)

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
