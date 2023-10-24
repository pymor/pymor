# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.gram_schmidt import gram_schmidt


def qr(A, product=None, offset=0, check=True, check_tol=1e-3, copy=True):
    """Compute a QR decomposition.

    Parameters
    ----------
    A
        The |VectorArray| for which to compute the QR decomposition.
    product
        The inner product |Operator| w.r.t. which to orthonormalize.
        If `None`, the Euclidean product is used.
    offset
        Assume that the first `offset` vectors are already orthonormal and start the
        algorithm at the `offset + 1`-th vector.
    check
        If `True`, check if the resulting |VectorArray| is really orthonormal.
    check_tol
        Tolerance for the check.
    copy
        If `True`, create a copy of `A` instead of modifying `A` in-place.

    Returns
    -------
    Q
        The orthonormal |VectorArray| of the same length as `A`.
    R
        The upper-triangular/trapezoidal matrix.
    """
    return gram_schmidt(A, product=product, return_R=True, atol=0, rtol=0, offset=offset,
                        check=check, check_tol=check_tol, copy=copy)


def rrqr(A, product=None, offset=0, check=True, check_tol=1e-3, copy=True):
    """Compute a rank-revealing QR (RRQR) decomposition.

    Parameters
    ----------
    A
        The |VectorArray| for which to compute the QR decomposition.
    product
        The inner product |Operator| w.r.t. which to orthonormalize.
        If `None`, the Euclidean product is used.
    offset
        Assume that the first `offset` vectors are already orthonormal and start the
        algorithm at the `offset + 1`-th vector.
    check
        If `True`, check if the resulting |VectorArray| is really orthonormal.
    check_tol
        Tolerance for the check.
    copy
        If `True`, create a copy of `A` instead of modifying `A` in-place.

    Returns
    -------
    Q
        The orthonormalized |VectorArray|.
    R
        The upper-triangular/trapezoidal matrix.
    """
    return gram_schmidt(A, product=product, return_R=True, offset=offset,
                        check=check, check_tol=check_tol, copy=copy)
