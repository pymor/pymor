# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""VKOGA demo with 2d input."""

import matplotlib.pyplot as plt
import numpy as np
from typer import Option, run

from pymor.algorithms.vkoga import GaussianKernel, VKOGAEstimator
from pymor.tools.typer import Choices


def main(num_training_points: int = Option(100, help='Number of training points in the weak greedy algorithm.'),
         greedy_criterion: Choices('fp f p') = Option('fp', help='Selection criterion for the greedy algorithm.'),
         max_centers: int = Option(40, help='Maximum number of selected centers in the greedy algorithm.'),
         tol: float = Option(1e-6, help='Tolerance for the weak greedy algorithm.'),
         reg: float = Option(1e-12, help='Regularization parameter for the kernel interpolation.'),
         length_scale: float = Option(1.0, help='The length scale parameter of the kernel. '
                                                'Only used when `kernel = diagonal`.')):
    # define kernel
    kernel = GaussianKernel(length_scale)

    # function to be approximated
    def f(X):
        x1, x2 = X[:, 0], X[:, 1]
        return np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2)

    # training data
    grid_x1 = np.linspace(0, 1, 10)
    grid_x2 = np.linspace(0, 1, 10)
    X1, X2 = np.meshgrid(grid_x1, grid_x2)
    X_train = np.column_stack([X1.ravel(), X2.ravel()])
    F_train = f(X_train)

    # set up estimator
    estimator = VKOGAEstimator(kernel=kernel, criterion=greedy_criterion, max_centers=max_centers, tol=tol, reg=reg)
    estimator.fit(X_train, F_train)

    # evaluation grid
    grid_test = np.linspace(0, 1, 40)
    X1t, X2t = np.meshgrid(grid_test, grid_test)
    X_test = np.column_stack([X1t.ravel(), X2t.ravel()])
    F_pred = estimator.predict(X_test).squeeze()
    F_true = f(X_test)
    error = abs(F_true - F_pred)

    # visualization
    fig, axes = plt.subplots(3, figsize=(12, 10))

    # plot prediction
    axes[0].contourf(X1t, X2t, F_pred.reshape(X1t.shape), levels=20, cmap='coolwarm')
    axes[0].scatter(X_train[estimator._surrogate._centers_idx, 0], X_train[estimator._surrogate._centers_idx, 1],
                    c='k', s=40, label='Centers')
    axes[0].set_title('Prediction s_n')
    axes[0].set_xlabel('x₁')
    axes[0].set_ylabel('x₂')
    axes[0].legend()

    # plot reference
    axes[1].contourf(X1t, X2t, F_true.reshape(X1t.shape), levels=20, cmap='coolwarm')
    axes[1].scatter(X_train[estimator._surrogate._centers_idx, 0], X_train[estimator._surrogate._centers_idx, 1],
                    c='k', s=40, label='Centers')
    axes[1].set_title('Reference f')
    axes[1].set_xlabel('x₁')
    axes[1].set_ylabel('x₂')
    axes[1].legend()

    # plot error surface
    c = axes[2].contourf(X1t, X2t, error.reshape(X1t.shape), levels=20, cmap='viridis')
    axes[2].scatter(X_train[estimator._surrogate._centers_idx, 0], X_train[estimator._surrogate._centers_idx, 1],
                    c='r', s=30, label='Centers')
    axes[2].scatter(X_train[:, 0], X_train[:, 1], c='k', s=10, label='Training points')
    axes[2].set_title('|f - s_n| error surface')
    axes[2].set_xlabel('x₁')
    axes[2].set_ylabel('x₂')
    axes[2].legend()
    fig.colorbar(c, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run(main)
