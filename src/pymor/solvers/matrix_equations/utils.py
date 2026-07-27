# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import numpy as np
import scipy.linalg as spla

from pymor.core.defaults import defaults


@defaults('value')
def mat_eqn_sparse_min_size(value=1000):
    """Returns minimal size for which a sparse solver will be used by default."""
    return value


def _chol(A):
    """Cholesky decomposition.

    This implementation uses SVD to compute the Cholesky factor (can be used for singular matrices).

    Parameters
    ----------
    A
        Symmetric positive semidefinite matrix as a |NumPy array|.

    Returns
    -------
    L
        Cholesky factor of A (in the sense that L * L^T approximates A).
    """
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    from pymor.bindings.scipy import svd_lapack_driver
    U, s, _ = spla.svd(A, lapack_driver=svd_lapack_driver())
    L = U * np.sqrt(s)
    return L
