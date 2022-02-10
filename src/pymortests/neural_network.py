# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import os
import pytest

from pymor.core.exceptions import TorchMissing

import numpy as np


def _skip_if_no_torch():
    if not os.environ.get('DOCKER_PYMOR', False):
        pytest.skip('skipped test due to missing Torch')

def test_linear_function_fitting():
    _skip_if_no_torch()

    from pymor.reductors.neural_network import multiple_restarts_training
    from pymor.models.neural_network import FullyConnectedNN

    import torch.optim as optim

    n = 1000
    d_in = 3
    d_out = 2

    a = np.random.rand(d_in, d_out)
    b = np.random.rand(d_out)

    points = np.random.rand(n, d_in)
    vals = points.dot(a) + b

    data = list(zip(points, vals))

    validation_ratio = 0.1

    training_data = data[0:int(len(data)*validation_ratio)]
    validation_data = data[int(len(data)*validation_ratio):]
    neural_network = FullyConnectedNN([d_in, 3 * (d_in + d_out), 3 * (d_in + d_out), d_out]).double()

    # without specifying training parameters
    tol = 1e-5
    _, best_losses = multiple_restarts_training(training_data, validation_data, neural_network)
    assert all(loss < tol for loss in best_losses.values())

    # with training parameters (that differ from the default values)
    optimizer = optim.Adam
    learning_rate = 1e-2
    epochs = 2000
    max_restarts = 1
    batch_size = 30
    lr_scheduler = optim.lr_scheduler.StepLR
    lr_scheduler_params = {'step_size': 50, 'gamma': 0.9}
    training_parameters = {'optimizer': optimizer, 'learning_rate': learning_rate, 'batch_size': batch_size,
                           'epochs': epochs, 'lr_scheduler': lr_scheduler,
                           'lr_scheduler_params': lr_scheduler_params}

    tol = 1e-4
    _, best_losses = multiple_restarts_training(training_data, validation_data, neural_network,
                                                training_parameters=training_parameters,
                                                max_restarts=max_restarts)
    assert all(loss < tol for loss in best_losses.values())

def test_no_training_data():
    _skip_if_no_torch()

    from pymor.reductors.neural_network import multiple_restarts_training
    from pymor.models.neural_network import FullyConnectedNN

    import torch

    n = 1000
    d_in = 3
    d_out = 2
    training_data = []
    validation_data = []
    neural_network = FullyConnectedNN([d_in, 3 * (d_in + d_out), 3 * (d_in + d_out), d_out]).double()
    best_neural_network, _ = multiple_restarts_training(training_data, validation_data, neural_network)
    assert np.allclose(best_neural_network(torch.DoubleTensor(np.random.rand(n, d_in))).detach(), np.zeros(d_out))
