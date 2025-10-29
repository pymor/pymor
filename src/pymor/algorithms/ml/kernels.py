# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np


class GaussianKernel:
    r"""Standalone Gaussian (RBF) kernel with scikit-learn-like interface.

    .. math::
        k(x, y) = exp(-||x - y||^2 / (2 * length_scale^2))

    Parameters
    ----------
    length_scale : float, default=1.0
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

        # Compute squared Euclidean distances
        diff = X[:, None, :] - Y[None, :, :]
        sqdist = np.sum(diff**2, axis=-1)

        # Gaussian kernel
        K = np.exp(-0.5 * sqdist / (self.length_scale ** 2))
        return K


class DiagonalVectorValuedKernel:
    r"""A simple vector-valued kernel built from a scalar base kernel.

    Produces a block-diagonal kernel matrix of shape :math:`(n\cdot m, n'\cdot m)`,
    where `m` is the number of outputs and `n` and `n'` are the numbers of inputs.

    Parameters
    ----------
    base_kernel
        A scalar kernel function :math:`k(x, x')`. Must return an array of shape :math:`(n, n')`.
    n_outputs
        Number of output components (:math:`m`).
    """

    def __init__(self, base_kernel, n_outputs):
        self.base_kernel = base_kernel
        self.n_outputs = n_outputs

    def __call__(self, X, Y):
        r"""Compute blockified vector-valued kernel between `X` and `Y`.

        If `base_kernel(X, Y)` has shape :math:`(n, n')`, the returned matrix
        has shape :math:`(n\cdot m, n'\cdot m)`, where `m = n_outputs`.
        """
        K_scalar = self.base_kernel(X, Y)

        # efficient block-diagonal expansion: kron(I_m, K_scalar)
        # each block corresponds to one output dimension
        K_block = np.kron(np.eye(self.n_outputs), K_scalar)

        return K_block
