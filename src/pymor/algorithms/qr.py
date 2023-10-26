# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.defaults import defaults
from pymor.core.exceptions import AccuracyError


@defaults('check', 'check_tol', 'method')
def qr(A, product=None, offset=0, check=True, check_tol=1e-3, copy=True, method='gram_schmidt', **kwargs):
    """Compute a QR decomposition.

    Parameters
    ----------
    A
        The |VectorArray| for which to compute the QR decomposition.
    product
        The inner product |Operator| w.r.t. which to orthonormalize.
        If `None`, the Euclidean product is used.
    offset
        Assume that the first `offset` vectors are already orthonormal.
        Those vectors will not be changed even when `copy` is `False`.
    check
        If `True`, check if the resulting |VectorArray| is really orthonormal.
    check_tol
        Tolerance for the check.
    copy
        If `True`, create a copy of `A` instead of modifying `A` in-place.
    method
        QR decomposition method (currently only `'gram_schmidt'`).
    kwargs
        Keyword arguments specific to the method.

    Returns
    -------
    Q
        The orthonormal |VectorArray| of the same length as `A`.
    R
        The upper-triangular/trapezoidal matrix.
    """
    assert method == 'gram_schmidt'
    Q, R = gram_schmidt(A, product=product, return_R=True, atol=0, rtol=0, offset=offset,
                        check=False, copy=copy, **kwargs)
    if check:
        _check_qr(A, product, offset, check_tol, Q, R)
    return Q, R


@defaults('check', 'check_tol', 'method')
def rrqr(A, product=None, offset=0, check=True, check_tol=1e-3, copy=True, method='gram_schmidt', **kwargs):
    """Compute a rank-revealing QR (RRQR) decomposition.

    Parameters
    ----------
    A
        The |VectorArray| for which to compute the QR decomposition.
    product
        The inner product |Operator| w.r.t. which to orthonormalize.
        If `None`, the Euclidean product is used.
    offset
        Assume that the first `offset` vectors are already orthonormal.
        Those vectors will not be changed even when `copy` is `False`.
    check
        If `True`, check if the resulting |VectorArray| is really orthonormal.
    check_tol
        Tolerance for the check.
    copy
        If `True`, create a copy of `A` instead of modifying `A` in-place.
    method
        QR decomposition method (currently only `'gram_schmidt'`).
    kwargs
        Keyword arguments specific to the method.

    Returns
    -------
    Q
        The orthonormal |VectorArray| of the length less or equal to the length of `A`.
    R
        The upper-triangular/trapezoidal matrix.
    """
    assert method == 'gram_schmidt'
    Q, R = gram_schmidt(A, product=product, return_R=True, offset=offset,
                        check=False, copy=copy, **kwargs)
    if check:
        _check_qr(A, product, offset, check_tol, Q, R)
    return Q, R


def _check_qr(A, product, offset, check_tol, Q, R):
    orth_error_matrix = A[offset:len(A)].inner(A, product)
    orth_error_matrix[:len(A) - offset, offset:len(A)] -= np.eye(len(A) - offset)
    if orth_error_matrix.size > 0:
        err = np.max(np.abs(orth_error_matrix))
        if err >= check_tol:
            raise AccuracyError(f'Q not orthonormal (max err={err})')
    A_norm = spla.norm(A.norm())
    qr_error = spla.norm((A - Q.lincomb(R.T)).norm())
    qr_error_rel = qr_error / A_norm
    if qr_error_rel >= check_tol:
        raise AccuracyError(f'QR not accurate (rel err={qr_error_rel})')
