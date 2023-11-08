# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.defaults import defaults
from pymor.core.exceptions import AccuracyError


@defaults('check', 'check_orth_tol', 'check_recon_tol', 'method')
def qr(V, product=None, offset=0,
       check=True, check_orth_tol=1e-3, check_recon_tol=1e-3,
       copy=True, method='gram_schmidt', **kwargs):
    r"""Compute a thin QR decomposition.

    For a |VectorArray| `V` and inner product |Operator| `product`,
    the thin QR decomposition of `V` is given by
    orthonormal |VectorArray| `Q` and
    upper-triangular matrix `R` such that
    `V == Q.lincomb(R.T)` and `len(V) == len(Q)`.

    .. note::
        If the set of vectors in :math:`V` is linearly dependent,
        the implementation may raise an exception (see below).

    Parameters
    ----------
    V
        The |VectorArray| for which to compute the thin QR decomposition.
    product
        The inner product |Operator| w.r.t. which to orthonormalize.
        If `None`, the Euclidean product is used.
    offset
        Assume that the first `offset` vectors are already orthonormal.
        Those vectors will not be changed even when `copy` is `False`.
    check
        If `True`, check if the resulting `Q` is orthonormal and
        if `Q.lincomb(R.T)` approximates `V`.
    check_orth_tol
        Tolerance for the orthogonality check:
        :math:`\lVert Q^H P Q - I \rVert_F \leqslant \mathtt{orth\_tol}`.
    check_recon_tol
        Tolerance for the reconstruction check:
        :math:`\lVert V - Q R \rVert_F \leqslant \mathtt{recon\_tol} \lVert A \rVert_F`.
    copy
        If `True`, create a copy of `V` instead of modifying `V` in-place.
    method
        QR decomposition method (currently only `'gram_schmidt'`).
    kwargs
        Keyword arguments specific to the method.

    Returns
    -------
    Q
        The orthonormal |VectorArray| of the same length as `V`.
    R
        The upper-triangular/trapezoidal matrix.

    Raises
    ------
    RuntimeError
        If the vectors in `V` are detected to be linearly dependent.
    """
    assert method == 'gram_schmidt'
    len_V = len(V)
    if check:
        V_orig = V if copy else V.copy()
    Q, R = gram_schmidt(V, product=product, return_R=True, atol=0, rtol=0, offset=offset,
                        check=False, copy=copy, **kwargs)
    if len(Q) < len_V:
        raise RuntimeError('Could not construct an orthonormal basis of full length.')
    if check:
        _check_qr(V_orig, Q, R, product, offset, check_orth_tol, check_recon_tol)
    return Q, R


@defaults('atol', 'rtol', 'check', 'check_orth_tol', 'check_recon_tol', 'method')
def rrqr(V, product=None, atol=1e-13, rtol=1e-13, offset=0,
         check=True, check_orth_tol=1e-3, check_recon_tol=1e-3,
         copy=True, method='gram_schmidt', **kwargs):
    r"""Compute a rank-revealing QR (RRQR) decomposition.

    For a |VectorArray| `V` and inner product |Operator| `product`,
    a rank-revealing QR decomposition of `V` is given by any
    orthonormal |VectorArray| `Q` and
    upper-triangular/trapezoidal matrix `R` such that
    the norm of `V - Q.lincomb(R.T)` is "small" and
    `len(Q) <= len(V)`.

    Parameters
    ----------
    V
        The |VectorArray| for which to compute the rank-revealing QR decomposition.
    product
        The inner product |Operator| w.r.t. which to orthonormalize.
        If `None`, the Euclidean product is used.
    atol
        Absolute tolerance used to detect linearly dependent vectors
        (exact meaning depends on the `method`).
    rtol
        Relative tolerance used to detect linearly dependent vectors
        (exact meaning depends on the `method`).
    offset
        Assume that the first `offset` vectors are already orthonormal.
        Those vectors will not be changed even when `copy` is `False`.
    check
        If `True`, check if the resulting `Q` is orthonormal and if `Q*R` approximates `A`.
    check_orth_tol
        Tolerance for the orthogonality check:
        :math:`\lVert Q^H P Q - I \rVert_F \leqslant \mathtt{orth\_tol}`.
    check_recon_tol
        Tolerance for the reconstruction check:
        :math:`\lVert A - Q R \rVert_F \leqslant \mathtt{recon\_tol} \lVert A \rVert_F`.
    copy
        If `True`, create a copy of `A` instead of modifying `A` in-place.
    method
        QR decomposition method (currently only `'gram_schmidt'`).
    kwargs
        Keyword arguments specific to the method.

    Returns
    -------
    Q
        The orthonormal |VectorArray| such that `len(Q) <= len(V)`.
    R
        The upper-triangular/trapezoidal matrix.
    """
    assert method == 'gram_schmidt'
    if check:
        V_orig = V if copy else V.copy()
    Q, R = gram_schmidt(V, product=product, return_R=True, atol=atol, rtol=rtol, offset=offset,
                        check=False, copy=copy, **kwargs)
    if check:
        _check_qr(V_orig, Q, R, product, offset, check_orth_tol, check_recon_tol)
    return Q, R


def _check_qr(V, Q, R, product, offset, check_orth_tol, check_recon_tol):
    orth_error_matrix = Q[offset:].inner(Q, product)
    orth_error_matrix[:, offset:] -= np.eye(len(Q) - offset)
    if orth_error_matrix.size > 0:
        orth_err = spla.norm(orth_error_matrix)
        if orth_err > check_orth_tol:
            raise AccuracyError(f'Q not orthonormal (orth err={orth_err})')
    V_norm = spla.norm(V.norm())
    recon_err = spla.norm((V - Q.lincomb(R.T)).norm())
    recon_err_rel = recon_err / V_norm if V_norm > 0 else 0
    if recon_err > check_recon_tol * V_norm:
        raise AccuracyError(f'QR not accurate (rel recon err={recon_err_rel})')
