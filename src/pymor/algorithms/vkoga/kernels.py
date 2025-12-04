# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np


class GaussianKernel:
    r"""Standalone Gaussian (RBF) kernel with scikit-learn-like interface.

    .. math::
        k(x, y) = \exp(-||x - y||_2^2 / (2 \cdot \text{length_scale}^2))

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


class DiagonalVectorValuedKernel:
    r"""A simple vector-valued kernel built from a scalar base kernel.

    Produces a block-diagonal kernel matrix of shape :math:`(n, n', m, m)`,
    where :math:`m` is the number of outputs and :math:`n` and :math:`n'` are
    the numbers of inputs.

    Parameters
    ----------
    base_kernel
        A scalar kernel function :math:`k(x, x')`. Must return an array of shape :math:`(n, n')`.
    n_outputs
        Number of output components (:math:`m`).
    """

    def __init__(self, base_kernel, n_outputs, diag_weights=None):
        self.base_kernel = base_kernel
        self.n_outputs = n_outputs

        if diag_weights is None:
            self.diag_weights = np.ones(n_outputs)
        else:
            self.diag_weights = np.asarray(diag_weights, dtype=float)
            assert self.diag_weights.shape == (n_outputs,)

    def __call__(self, X, Y=None):
        r"""Compute blockified vector-valued kernel between `X` and `Y`.

        If `base_kernel(X, Y)` has shape :math:`(n, n')`, the returned matrix
        has shape :math:`(n, n', m, m)`, where `m = n_outputs`.
        """
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        K_scalar = self.base_kernel(X, Y)

        if self.n_outputs == 1:
            return K_scalar

        D = np.diag(self.diag_weights)
        K_block = K_scalar[..., None, None] * D[None, None, :, :]

        n, m = X.shape[0], Y.shape[0]
        axes_to_squeeze = []
        if n == 1:
            axes_to_squeeze.append(0)
        if m == 1:
            axes_to_squeeze.append(1)

        if axes_to_squeeze:
            K_block = np.squeeze(K_block, axis=tuple(axes_to_squeeze))

        return K_block

    def diag(self, X):
        """Return the diagonal of the kernel matrix."""
        X = np.atleast_2d(X)
        k_diag = self.base_kernel.diag(X)
        d = self.diag_weights
        return (k_diag[:, None] * d[None, :]).ravel()
