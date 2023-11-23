# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from numbers import Integral, Real

import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.qr import _check_qr
from pymor.algorithms.svd_va import method_of_snapshots
from pymor.core.defaults import defaults
from pymor.core.exceptions import LinAlgError
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray


@defaults('atol', 'rtol', 'check', 'check_orth_tol', 'check_recon_tol')
def orth(V, product=None,
         hierarchical=False, pad=False, allow_truncation=True,
         offset=0, copy=True,
         atol=1e-13, rtol=1e-13,
         check=True, check_orth_tol=1e-12, check_recon_tol=1e-12,
         method=None, **kwargs):
    """Compute an orthonormal basis.

    Given a |VectorArray| `V` and inner product |Operator| `product`,
    a `basis` for the span of the vectors in `V` is computed that is
    orthonormal w.r.t. `product`. The coefficients of the vectors in
    `V` w.r.t. the computed basis are returned as the second return
    value. In particular, we have::

        almost_equal(V, basis.lincomb(coeffs))

    This method can be used to compute a QR decomposition of the
    matrix of column vectors contained in `V` by setting
    `hierarchical=True`. In this case, `coeffs.T` will be
    upper-triangular (trapezoidal).

    If the vectors in `V` are linearly dependent, then
    `len(basis) < len(V)`. By setting `pad=True`, one can ensure that
    `len(basis) == len(V)`. Depending on the used orthogonalization
    algorithm, this is realized by padding `basis` with random vectors
    and orthonormalizing these again.

    Parameters
    ----------
    V
        The |VectorArray| for which an orthonoral basis is computed.
    product
        The inner product |Operator| w.r.t. which `Q` is orthonormal.
        If `None`, the Euclidean product is used.
    hierarchical
        If `True`, ensure that the span of the k vectors in `V` is
        contained in the span of the first k vectors in `basis`.
    pad
        If `False`, then span the vectors in `basis` agrees, up to
        nuerical accuracy, with the span of the vectors in `V`.
        Hence, if the vectors in `V` are (numerically) linearly
        dependent, `basis` will contain less vectors than `V`.
        By setting `pad` to `True` the span of `V` is exteneded
        to ensure `len(V) == len(basis)`.
    allow_truncation
        If `False`, raise an `LinAlgError` when the vectors in `V`
        are detected to be linearly dependent.
    offset
        Assume that the first `offset` vectors are already orthonormal.
        Those vectors will not be changed.
    copy
        If `True`, create a copy of `V` instead of modifying `V` in-place.
    check
        If `True`, check if the resulting `basis` is orthonormal and
        if `basis.lincomb(coeffs)` approximates `V`.
    atol
        Absolute tolerance used to detect linearly dependent vectors
        (exact meaning depends on the `method`).
    rtol
        Relative tolerance used to detect linearly dependent vectors
        (exact meaning depends on the `method`).
    check_orth_tol
        Tolerance for the orthogonality check:
        `frobenius_norm(basis.gramian(product) - np.eye(len(Q))) <= orth_tol`.
    check_recon_tol
        Tolerance for the reconstruction check:
        `norm((V - basis.lincomb(coeffs)).norm(product)) <= recon_tol * norm(V.norm(product))`.
    method
        Orthonormalization method (currently only `'gram_schmidt', 'method_of_snapshots'`).
    kwargs
        Keyword arguments specific to the used orthonormalization method.

    Returns
    -------
    basis
        The computed orthonormal basis.
    coefficients
        `(len(V), len(basis))`-shaped array containing the coefficients of the
        vectors in `V` w.r.t. the vectors in `basis`.

    Raises
    ------
    AccuracyError
        If the orthogonality or reconstruction check fail.
    LinAlgError
        If the vectors in `V` are detected to be linearly dependent.

    See Also
    --------
    :func:`~pymor.algorithms.qr.rrqr` : Rank-revealing QR decomposition with truncation.
    """
    assert isinstance(V, VectorArray)
    if product is not None:
        assert isinstance(product, Operator)
        assert product.source == product.range
        assert not product.parametric
        assert V in product.source
    assert isinstance(atol, Real)
    assert atol >= 0
    assert isinstance(rtol, Real)
    assert rtol >= 0
    assert isinstance(offset, Integral)
    assert offset >= 0
    assert isinstance(check_orth_tol, Real)
    assert check_orth_tol > 0
    assert isinstance(check_recon_tol, Real)
    assert check_recon_tol > 0
    assert allow_truncation or not pad
    if method is None:
        method = default_orth_method(hierarchical)
    if method not in {'gram_schmidt', 'method_of_snapshots'}:
        raise ValueError(f'Unknown QR method {method}')
    if hierarchical and method in {'method_of_snapshots'}:
        raise ValueError(f'{method} cannot yield hierarchical basis')

    if check and np.isfinite(check_recon_tol) and not copy:
        V_orig = V.copy()
    else:
        V_orig = V

    def _pad_vectors(basis, coeffs):
        missing_vectors = len(V_orig) - len(basis)
        basis.append(basis.random(missing_vectors))
        gram_schmidt(
            basis, product=product, atol=0, rtol=0, offset=len(V_orig)-missing_vectors,
            check=False, copy=False,
        )
        coeffs = np.hstack([coeffs, np.zeros((len(V_orig), missing_vectors))])
        return coeffs

    if method == 'gram_schmidt':
        basis = V.copy() if copy else V
        _, coeffs = gram_schmidt(
            basis, product=product, return_R=True, atol=atol, rtol=rtol, offset=offset,
            check=False, copy=False, **kwargs
        )
        coeffs = coeffs.T
        if len(basis) < len(V_orig) and pad:
            coeffs = _pad_vectors(basis, coeffs)
    elif method == 'method_of_snapshots':
        if offset > 0:
            coeffs_0 = V[offset:].inner(V[:offset], product=product)
            VV = V[offset:] - V[:offset].lincomb(coeffs_0)
        else:
            VV = V
        U, s, v = method_of_snapshots(
            VV, product=product, atol=atol, rtol=rtol
        )

        if copy:
            if offset > 0:
                basis = V[:offset].copy()
                basis.append(U)
            else:
                basis = U
        else:
            del V[offset + len(basis):]
            V[offset:].scal(0.)
            V[offset:] += basis
            basis = V
        coeffs = s * v.T

        if len(basis) < len(V_orig) and pad:
            coeffs = _pad_vectors(basis, coeffs)

        if offset > 0:
            coeffs = np.vstack([np.zeros((offset, len(V_orig))), np.hstack([coeffs_0, coeffs])])
            coeffs[:offset, :offset] = np.eye(offset)

    if not allow_truncation and len(basis) < len(V_orig):
        raise LinAlgError('The VectorArray has linearly dependent vectors.')

    if check:
        _check_qr(V_orig, basis, coeffs.T, product, offset, check_orth_tol, check_recon_tol)
    return basis, coeffs


@defaults('hierarchical', 'non_hierarchical')
def default_orth_method(is_hierarchical,
                        hierarchical='gram_schmidt',
                        non_hierarchical='gram_schmidt'):
    return hierarchical if is_hierarchical else non_hierarchical
