# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np


class GaussianKernel:
    r"""Standalone Gaussian (RBF) kernel with scikit-learn-like interface.

    .. math::
        k(x, y) = \exp(-||x - y||_2^2 / (2 l^2))

    where :math:`l` is the length scale parameter.

    Parameters
    ----------
    length_scale
        The length scale parameter of the kernel.
    """

    def __init__(self, length_scale=1.0):
        assert length_scale > 0.
        self.length_scale = length_scale

    def __call__(self, X, Y=None):
        """Evaluate the kernel matrix pairwise between `X` and `Y`.

        Parameters
        ----------
        X
            First inputs to the kernel.
        Y
            Second inputs to the kernel. If None, uses `Y = X`.

        Returns
        -------
        K
            Kernel matrix evaluated pairwise for `X` and `Y`.
        """
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)

        # compute squared Euclidean distances
        sqdist = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=-1)

        # Gaussian kernel
        return np.exp(-0.5 * sqdist / (self.length_scale ** 2))

    def diag(self, X):
        """Return the diagonal of the kernel matrix."""
        X = np.atleast_2d(X)
        return np.ones(X.shape[0])

    def get_params(self, deep=True):
        """Returns a dict of the init-parameters of the kernel, together with their values.

        The argument `deep=True` is required to match the scikit-learn interface.

        Parameters
        ----------
        deep
            Since the kernel has no subobjects with parameters, this parameter is only
            required to match the scikit-learn interface via duck-typing.

        Returns
        -------
        A dictionary of parameters and respective values of the kernel.
        """
        return {'length_scale': self.length_scale}

    def set_params(self, **params):
        """Set the parameters of the kernel.

        Parameters
        ----------
        Kernel parameters to set.

        Returns
        -------
        An instance of the kernel with the new parameters.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
