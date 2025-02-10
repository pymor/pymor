# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

# This file was originally based upon the MORLAB toolbox
# Copyright (C) 2006-2023 Peter Benner, Jens Saak, and Steffen W. R. Werner
# All rights reserved.
# License: BSD 2-Clause License

import numpy as np
import scipy.linalg as spla


def compress_fac(Z, tol, column_compression=True):
    """Perform SVD-based column/row compression.

    Computes a column or row compression of the matrix Z using the SVD.
    Usually used to approximate the products Z^T*Z or Z*Z^T via a low-rank
    factor.

    Parameters
    ----------
    Z
        |NumPy array| of dimensions n x m.
    tol
        Nonnegative scalar, tolerance multiplied with the largest singular value
        to determine the rank of the approximation.
    column_compression
        Whether to do column compression.
        If `False`, do row compression.

    Returns
    -------
    W
        |NumPy array| of dimensions n x r in case of column compression and
        r x m in case of row compression.

    See Also
    --------
    compress_ldl
    """
    # check inputs
    assert isinstance(Z, np.ndarray)
    assert Z.ndim == 2

    assert tol >= 0

    # column/row compression
    U, s, Vh = spla.svd(Z, full_matrices=False, lapack_driver='gesvd')
    r = sum(s > s[0] * tol)
    W = U[:, :r] * s[:r] if column_compression else s[:r, np.newaxis] * Vh[:r]
    return W


def compress_ldl(Z, Y, tol, column_compression=True):
    """Compress an LDL^T factorization.

    Computes a column or row compression of the matrices Z and Y using an
    eigenvalue decomposition. Usually used to approximate the products
    Z'*Y*Z (row compression) or Z*Y*Z' (column compression) via low-rank
    factors.

    Parameters
    ----------
    Z
        |NumPy array| of dimensions n x m (column compression) or m x n
        (row compression).
    Y
        Symmetric |NumPy array| of dimensions m x m.
    tol
        Nonnegative scalar, tolerance multiplied with the largest singular value
        to determine the rank of the approximation.
    column_compression
        Whether to do column compression.
        If `False`, do row compression.

    Returns
    -------
    Z2
        |NumPy array| of dimensions n x r in case of column compression and
        r x n in case of row compression.
    Y2
        |NumPy array| of dimensions r x r.

    See Also
    --------
    compress_fac
    """
    # check inputs
    assert isinstance(Z, np.ndarray)
    assert Z.ndim == 2

    assert isinstance(Y, np.ndarray)
    assert Y.ndim == 2
    assert Y.shape[0] == Y.shape[1]
    if column_compression:
        assert Z.shape[1] == Y.shape[0]
    else:
        assert Z.shape[0] == Y.shape[0]

    assert tol >= 0

    # LDL^T column/row compression
    Q, R = spla.qr(Z if column_compression else Z.T, mode='economic')
    RYR = R @ Y @ R.T
    RYR = (RYR + RYR.T) / 2
    w, V = spla.eigh(RYR)
    r = sum(w > w[-1] * tol)
    Y2 = np.diag(w[-r:])
    Z2 = Q @ V[:, -r:]
    if not column_compression:
        Z2 = Z2.T

    return Z2, Y2
