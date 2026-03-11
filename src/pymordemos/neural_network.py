# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Demo of the VKOGA algorithm for function approximation."""

import matplotlib.pyplot as plt
import numpy as np
from typer import Option, run

from pymor.algorithms.ml.nn import NeuralNetworkRegressor
from pymor.core.config import config
from pymor.core.exceptions import TorchMissingError
from pymor.tools.random import new_rng
from pymor.tools.typer import Choices


def main(training_points_sampling: Choices('random uniform') = Option('random',
                                                                      help='Method for sampling the training points'),
         num_training_points: int = Option(40, help='Number of training points in the neural network training.'),
         grid_search_parameter_optimization: bool = Option(False, help='Perform a grid search in order to optimize the '
                                                                       'hyperparameters of the neural network and '
                                                                       'the optimization during training.'),
         num_points_plotting: int = Option(200, help='Number of points used for plotting '
                                                     'of the approximation result.')):
    """Approximates a function with 2d output from training data using a neural network."""
    m = 2
    if not config.HAVE_TORCH:
        raise TorchMissingError

    # training data
    assert training_points_sampling in ('uniform', 'random')
    if training_points_sampling == 'uniform':
        X = np.linspace(0, 1, num_training_points)[:, None]
    elif training_points_sampling == 'random':
        rng = new_rng(0)
        X = rng.uniform(0, 1, num_training_points)[:, None]

    F = np.column_stack([np.sin(2*np.pi*X).ravel(), np.cos(2*np.pi*X).ravel()])

    # set up regressor
    regressor = NeuralNetworkRegressor(restarts=1)
    if grid_search_parameter_optimization:
        print('Running grid search for best parameters in the neural network and its optimization:')
        import torch.optim as optim
        from sklearn.model_selection import GridSearchCV
        gs = GridSearchCV(regressor, {'validation_ratio': [0.1, 0.2],
                                      'optimizer': [optim.LBFGS, optim.Adam],
                                      'neural_network__hidden_layers': [[30], [30, 30], [30, 30, 30]]}).fit(X, F)
        regressor = gs.best_estimator_
        print('Best parameters:')
        print(gs.best_params_)
    else:
        regressor.fit(X, F)

    # visualization
    X_dense = np.linspace(X.min(), X.max(), num_points_plotting)[:, None]
    F_pred = regressor.predict(X_dense)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    titles = [r'$f_1(x)$', r'$f_2(x)$']
    for i in range(m):
        axs[i].plot(X_dense, F_pred[:, i], 'r-', lw=2, label='Neural network surrogate')
        axs[i].scatter(X, F[:, i], c='k', s=30, label='Training data', alpha=0.6)
        axs[i].set_ylabel(titles[i])
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc='best')

    axs[1].set_xlabel(r'$x$')
    plt.suptitle('Neural Network Surrogate vs Training Data')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run(main)
