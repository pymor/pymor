# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from cyclopts import App

from pymor.algorithms.ml.vkoga import GaussianKernel, VKOGARegressor
from pymor.core.config import config
from pymor.core.exceptions import SklearnMissingError
from pymor.tools.random import new_rng

app = App(help_on_error=True)

@app.default
def main(training_points_sampling: Literal['random', 'uniform'] = 'random',
         kernel: Literal['Gaussian', 'Matern', 'RationalQuadratic'] = 'Gaussian',
         num_training_points: int = 40,
         greedy_criterion: Literal['fp', 'f', 'p'] = 'fp',
         max_centers: int = 20,
         tol: float = 1e-6,
         reg: float = 1e-12,
         length_scale: float = 1.0,
         test_extend: bool = True):
    """Approximates a function with 2d output from training data using VKOGA.

    Parameters
    ----------
    training_points_sampling
        Method for sampling the training points.
    kernel
        Kernel to use in VKOGA.
    num_training_points
        Number of training points in the weak greedy algorithm.
    greedy_criterion
        Selection criterion for the greedy algorithm.
    max_centers
        Maximum number of selected centers in the greedy algorithm.
    tol
        Tolerance for the weak greedy algorithm.
    reg
        Regularization parameter for the kernel interpolation.
    length_scale
        The length scale parameter of the kernel. Only used when `kernel = diagonal`.
    test_extend
        If True, also test the incremental `extend` method by fitting on 1/2 of
        the data and extending twice with 1/4 each, then comparing with the full fit.
    """
    m = 2
    if kernel in ('Matern', 'RationalQuadratic') and not config.HAVE_SKLEARN:
        raise SklearnMissingError

    if kernel == 'Gaussian':
        kernel = GaussianKernel(length_scale)
    elif kernel == 'Matern':
        from sklearn.gaussian_process.kernels import Matern
        kernel = Matern(length_scale)
    elif kernel == 'RationalQuadratic':
        from sklearn.gaussian_process.kernels import RationalQuadratic
        kernel = RationalQuadratic(length_scale)

    assert training_points_sampling in ('uniform', 'random')
    if training_points_sampling == 'uniform':
        X = np.linspace(0, 1, num_training_points)[:, None]
    elif training_points_sampling == 'random':
        rng = new_rng(0)
        X = rng.uniform(0, 1, num_training_points)[:, None]

    F = np.column_stack([np.sin(2*np.pi*X).ravel(), np.cos(2*np.pi*X).ravel()])

    regressor = VKOGARegressor(kernel=kernel, criterion=greedy_criterion, max_centers=max_centers, tol=tol, reg=reg)
    regressor.fit(X, F)

    weak_greedy_result = regressor._weak_greedy_result
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

    plot_vkoga_results(regressor, X, F)

    if test_extend:
        n = len(X)
        n_init = n // 2
        n_ext1 = (n - n_init) // 2
        n_ext2 = n - n_init - n_ext1

        X_init, X_ext1, X_ext2 = X[:n_init], X[n_init:n_init+n_ext1], X[n_init+n_ext1:]
        F_init, F_ext1, F_ext2 = F[:n_init], F[n_init:n_init+n_ext1], F[n_init+n_ext1:]

        regressor_ext = VKOGARegressor(kernel=kernel, criterion=greedy_criterion,
                                       max_centers=max_centers, tol=tol, reg=reg)
        regressor_ext.fit(X_init, F_init)
        n_centers_fit = len(regressor_ext._surrogate._centers)
        print(f'\nExtend test: fit on {n_init} points -> {n_centers_fit} centers')

        regressor_ext.extend(X_ext1, F_ext1)
        n_centers_ext1 = len(regressor_ext._surrogate._centers)
        print(f'Extended with {n_ext1} points -> {n_centers_ext1} centers')

        regressor_ext.extend(X_ext2, F_ext2)
        n_centers_ext2 = len(regressor_ext._surrogate._centers)
        print(f'Extended with {n_ext2} points -> {n_centers_ext2} centers')

        X_dense = np.linspace(X.min(), X.max(), 200)[:, None]
        err_full = np.max(np.abs(regressor.predict(X_dense) - regressor_ext.predict(X_dense)))
        print(f'Full fit: {len(regressor._surrogate._centers)} centers, '
              f'fit+extend: {n_centers_ext2} centers, '
              f'max prediction difference: {err_full:.3e}')

        plot_vkoga_results(regressor_ext, X, F)


if __name__ == '__main__':
    app()
