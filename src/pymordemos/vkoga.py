# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""VKOGA demo."""

import matplotlib.pyplot as plt
import numpy as np
from typer import Option, run

from pymor.algorithms.vkoga import DiagonalVectorValuedKernel, GaussianKernel, VKOGAEstimator
from pymor.tools.random import new_rng
from pymor.tools.typer import Choices


def main(training_points_sampling: Choices('random uniform') = Option('random',
                                                                      help='Method for sampling the training points'),
         num_training_points: int = Option(40, help='Number of training points in the weak greedy algorithm.'),
         greedy_criterion: Choices('fp f p f/p') = Option('fp', help='Selection criterion for the greedy algorithm.'),
         max_centers: int = Option(20, help='Maximum number of selected centers in the greedy algorithm.'),
         tol: float = Option(1e-6, help='Tolerance for the weak greedy algorithm.'),
         reg: float = Option(1e-12, help='Regularization parameter for the kernel interpolation.'),
         length_scale: float = Option(1.0, help='The length scale parameter of the kernel. '
                                                'Only used when `kernel = diagonal`.')):
    m = 2
    kernel = DiagonalVectorValuedKernel(GaussianKernel(length_scale), m, diag_weights=[2, 1])

    assert training_points_sampling in ('uniform', 'random')
    if training_points_sampling == 'uniform':
        X = np.linspace(0, 1, num_training_points)[:, None]
    elif training_points_sampling == 'random':
        rng = new_rng(0)
        X = rng.uniform(0, 1, num_training_points)[:, None]

    F = np.column_stack([np.sin(2*np.pi*X).ravel(), np.cos(2*np.pi*X).ravel()])

    estimator = VKOGAEstimator(kernel=kernel, criterion=greedy_criterion, max_centers=max_centers, tol=tol, reg=reg)
    estimator.fit(X, F)

    weak_greedy_result = estimator._weak_greedy_result
    print('Result of the weak greedy algorithm:')
    print(weak_greedy_result)

    def plot_vkoga_results(vkoga, X_train, F_train, n_points=200):
        X_dense = np.linspace(X_train.min(), X_train.max(), n_points)[:, None]
        F_pred = vkoga.predict(X_dense)

        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        titles = [r'$f_1(x)$', r'$f_2(x)$']
        for i in range(m):
            axs[i].plot(X_dense, F_pred[:, i], 'r-', lw=2, label='VKOGA surrogate')
            axs[i].scatter(X_train, F_train[:, i], c='k', s=30, label='Training data', alpha=0.6)
            axs[i].scatter(vkoga._surrogate._centers[:, 0],
                           F_train[vkoga._surrogate._centers_idx, i],
                           c='b', s=60, label='Selected centers', zorder=5)
            axs[i].set_ylabel(titles[i])
            axs[i].grid(True, alpha=0.3)
            axs[i].legend(loc='best')

        axs[1].set_xlabel('x')
        plt.suptitle('VKOGA Surrogate vs Training Data')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    plot_vkoga_results(estimator, X, F)


if __name__ == '__main__':
    run(main)
