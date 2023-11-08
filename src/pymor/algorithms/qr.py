# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.defaults import defaults
from pymor.core.exceptions import AccuracyError


@defaults('check', 'check_orth_tol', 'check_recon_tol', 'method')
def qr(A, product=None, offset=0,
       check=True, check_orth_tol=1e-3, check_recon_tol=1e-3,
       copy=True, method='gram_schmidt', **kwargs):
    r"""Compute a QR decomposition.

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
    check_orth_tol
        Tolerance for the orthogonality check:
        :math:`\lVert Q^H P Q - I \rVert_F \leqslant \mathtt{orth_tol}`.
    check_recon_tol
        Tolerance for the reconstruction check:
        :math:`\lVert A - Q R \rVert_F \leqslant \mathtt{recon_tol} \lVert A \rVert_F`.
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
    if check:
        A_orig = A if copy else A.copy()
    Q, R = gram_schmidt(A, product=product, return_R=True, atol=0, rtol=0, offset=offset,
                        check=False, copy=copy, **kwargs)
    if check:
        _check_qr(A_orig, Q, R, product, check_orth_tol, check_recon_tol)
    return Q, R


@defaults('check', 'check_orth_tol', 'check_recon_tol', 'method')
def rrqr(A, product=None, offset=0,
         check=True, check_orth_tol=1e-3, check_recon_tol=1e-3,
         copy=True, method='gram_schmidt', **kwargs):
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
    if check:
        A_orig = A if copy else A.copy()
    Q, R = gram_schmidt(A, product=product, return_R=True, offset=offset,
                        check=False, copy=copy, **kwargs)
    if check:
        _check_qr(A_orig, Q, R, product, check_orth_tol, check_recon_tol)
    return Q, R


def _check_qr(A, Q, R, product, check_orth_tol, check_recon_tol):
    orth_error_matrix = Q.gramian(product) - np.eye(len(Q))
    if orth_error_matrix.size > 0:
        err = spla.norm(orth_error_matrix)
        if err > check_orth_tol:
            raise AccuracyError(f'Q not orthonormal (orth err={err})')
    A_norm = spla.norm(A.norm())
    recon_err = spla.norm((A - Q.lincomb(R.T)).norm())
    recon_err_rel = recon_err / A_norm
    if recon_err_rel > check_recon_tol:
        raise AccuracyError(f'QR not accurate (rel recon err={recon_err_rel})')
