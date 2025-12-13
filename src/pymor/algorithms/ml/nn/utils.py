# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('TORCH')

import torch.utils as utils

from pymor.core.base import BasicObject


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
