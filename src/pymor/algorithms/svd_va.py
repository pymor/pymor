# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Module for SVD method of operators represented by |VectorArrays|."""

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.scipy import svd_lapack_driver
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.constructions import IdentityOperator
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray


@defaults('rtol', 'atol', 'l2_err')
def method_of_snapshots(A, product=None, modes=None, rtol=1e-7, atol=0., l2_err=0.):
    """SVD of a |VectorArray| using the method of snapshots.

    Viewing the |VectorArray| `A` as a `A.dim` x `len(A)` matrix, the
    return value of this method is the singular value decomposition of
    `A`, where the inner product on R^(`dim(A)`) is given by `product`
    and the inner product on R^(`len(A)`) is the Euclidean inner
    product.

    .. warning::

        The left singular vectors may not be numerically orthonormal for
        ill-conditioned `A`.

    Parameters
    ----------
    A
        The |VectorArray| for which the SVD is to be computed.
    product
        Inner product |Operator| w.r.t. which the SVD is computed.
    modes
        If not `None`, at most the first `modes` singular values and
        vectors are returned.
    rtol
        Singular values smaller than this value multiplied by the
        largest singular value are ignored.
    atol
        Singular values smaller than this value are ignored.
    l2_err
        Do not return more modes than needed to bound the
        l2-approximation error by this value. I.e. the number of
        returned modes is at most ::

            argmin_N { sum_{n=N+1}^{infty} s_n^2 <= l2_err^2 }

        where `s_n` denotes the n-th singular value.

    Returns
    -------
    U
        |VectorArray| of left singular vectors.
    s
        One-dimensional |NumPy array| of singular values.
    Vh
        |NumPy array| of right singular vectors.
    """
    assert isinstance(A, VectorArray)
    assert product is None or isinstance(product, Operator)

    if A.dim == 0 or len(A) == 0:
        return A.space.empty(), np.array([]), np.zeros((0, len(A)))

    logger = getLogger('pymor.algorithms.svd_va.method_of_snapshots')

    with logger.block(f'Computing Gramian ({len(A)} vectors) ...'):
        B = A.gramian(product)

    with logger.block('Computing eigenvalue decomposition ...'):
        eigvals = (None
                   if modes is None or l2_err > 0.
                   else (max(len(B) - modes, 0), len(B) - 1))

        evals, V = spla.eigh(B, overwrite_a=True, subset_by_index=eigvals)
        evals = evals[::-1]
        V = V[:, ::-1]
        s = np.sqrt(np.clip(evals, a_min=0., a_max=None))

        selected_modes = _select_modes(s, modes, rtol, atol, l2_err)
        if selected_modes > A.dim:
            logger.warning('Number of computed singular vectors larger than array dimension! Truncating ...')
            selected_modes = A.dim

        s = s[:selected_modes]
        V = V[:, :selected_modes]
        Vh = V.conj().T

    with logger.block(f'Computing left-singular vectors ({len(V)} vectors) ...'):
        U = A.lincomb(V / s)

    return U, s, Vh


@defaults('rtol', 'atol', 'l2_err')
def qr_svd(A, product=None, modes=None, rtol=4e-8, atol=0., l2_err=0.):
    """SVD of a |VectorArray| using Gram-Schmidt orthogonalization.

    Viewing the |VectorArray| `A` as a `A.dim` x `len(A)` matrix, the
    return value of this method is the singular value decomposition of
    `A`, where the inner product on R^(`dim(A)`) is given by `product`
    and the inner product on R^(`len(A)`) is the Euclidean inner
    product.

    Parameters
    ----------
    A
        The |VectorArray| for which the SVD is to be computed.
    product
        Inner product |Operator| w.r.t. which the left singular vectors
        are computed.
    modes
        If not `None`, at most the first `modes` singular values and
        vectors are returned.
    rtol
        Singular values smaller than this value multiplied by the
        largest singular value are ignored.
    atol
        Singular values smaller than this value are ignored.
    l2_err
        Do not return more modes than needed to bound the
        l2-approximation error by this value. I.e. the number of
        returned modes is at most ::

            argmin_N { sum_{n=N+1}^{infty} s_n^2 <= l2_err^2 }

        where `s_n` denotes the n-th singular value.

    Returns
    -------
    U
        |VectorArray| of left singular vectors.
    s
        One-dimensional |NumPy array| of singular values.
    Vh
        |NumPy array| of right singular vectors.
    """
    assert isinstance(A, VectorArray)
    assert product is None or isinstance(product, Operator)

    if A.dim == 0 or len(A) == 0:
        return A.space.empty(), np.array([]), np.zeros((0, len(A)))

    logger = getLogger('pymor.algorithms.svd_va.qr_svd')

    with logger.block('Computing QR decomposition ...'):
        Q, R = gram_schmidt(A, product=product, return_R=True, check=False)

    with logger.block('Computing SVD of R ...'):
        U2, s, Vh = spla.svd(R, lapack_driver=svd_lapack_driver())

    with logger.block('Choosing the number of modes ...'):
        selected_modes = _select_modes(s, modes, rtol, atol, l2_err)
        U2 = U2[:, :selected_modes]
        s = s[:selected_modes]
        Vh = Vh[:selected_modes]

    with logger.block(f'Computing left singular vectors ({selected_modes} modes) ...'):
        U = Q.lincomb(U2)

    return U, s, Vh


@defaults('rtol', 'atol', 'l2_err')
def scipy_svd(A, product=None, modes=None, rtol=4e-8, atol=0., l2_err=0.):
    """SVD of a |VectorArray| using :func:`scipy.linalg.svd`.

    Viewing the |VectorArray| `A` as a `A.dim` x `len(A)` matrix, the
    return value of this method is the singular value decomposition of
    `A`, where the inner product on R^(`dim(A)`) is given by `product`
    and the inner product on R^(`len(A)`) is the Euclidean inner
    product.

    This functions uses :func:`scipy_linalg.svd` by converting the input
    |VectorArray| `A` to a |NumPy arry| by calling
    :meth:`~pymor.vectorarrays.interface.VectorArray.to_numpy`.

    Parameters
    ----------
    A
        The |VectorArray| for which the SVD is to be computed.
    product
        Not supported. Must be `None`.
        (Inner product |Operator| w.r.t. which the left singular vectors
        are computed.)
    modes
        If not `None`, at most the first `modes` singular values and
        vectors are returned.
    rtol
        Singular values smaller than this value multiplied by the
        largest singular value are ignored.
    atol
        Singular values smaller than this value are ignored.
    l2_err
        Do not return more modes than needed to bound the
        l2-approximation error by this value. I.e. the number of
        returned modes is at most ::

            argmin_N { sum_{n=N+1}^{infty} s_n^2 <= l2_err^2 }

        where `s_n` denotes the n-th singular value.

    Returns
    -------
    U
        |VectorArray| of left singular vectors.
    s
        One-dimensional |NumPy array| of singular values.
    Vh
        |NumPy array| of right singular vectors.
    """
    assert isinstance(A, VectorArray)
    if product is not None and not isinstance(product, IdentityOperator):
        raise NotImplementedError

    if A.dim == 0 or len(A) == 0:
        return A.space.empty(), np.array([]), np.zeros((0, len(A)))

    a = A.to_numpy(ensure_copy=True)
    U, s, Vh = spla.svd(a, full_matrices=False, compute_uv=True, overwrite_a=True, lapack_driver=svd_lapack_driver())

    selected_modes = _select_modes(s, modes, rtol, atol, l2_err)
    U = U[:, :selected_modes]
    s = s[:selected_modes]
    Vh = Vh[:selected_modes]

    U = A.space.from_numpy(U)
    return U, s, Vh


SVD_VA_METHODS = {
    'method_of_snapshots': method_of_snapshots,
    'qr_svd': qr_svd,
    'scipy_svd': scipy_svd
}


def _select_modes(s, modes, rtol, atol, l2_err):
    tol = max(rtol * s[0], atol)
    above_tol = np.where(s >= tol)[0]
    if len(above_tol) == 0:
        return 0
    last_above_tol = above_tol[-1]

    errs = np.concatenate((np.cumsum(s[::-1] ** 2)[::-1], [0.]))
    below_err = np.where(errs <= l2_err**2)[0]
    first_below_err = below_err[0]

    selected_modes = min(first_below_err, last_above_tol + 1)
    if modes is not None:
        selected_modes = min(selected_modes, modes)
    return selected_modes
