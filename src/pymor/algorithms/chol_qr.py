# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spsla

from pymor.core.exceptions import AccuracyError
from pymor.core.logger import getLogger
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.list import ListVectorArray


def shifted_chol_qr(A, product=None, return_R=False, maxiter=3, offset=0, orth_tol=None,
                    recompute_shift=False, check_finite=True, copy=True, product_norm=None):
    r"""Orthonormalize a |VectorArray| using the shifted CholeskyQR algorithm.

    This method computes a QR decomposition of a |VectorArray| via Cholesky factorizations of its
    Gramian matrix according to :cite:`FKNYY20`. For ill-conditioned matrices, the Cholesky
    factorization will break down. In this case a diagonal shift will be applied to the Gramian, see
    :cite:`BPS26` for details.

    - `shifted_chol_qr(A, maxiter=3, orth_tol=None)` is equivalent to the shifted CholeskyQR3
      algorithm (Algorithm 4.2 in :cite:`FKNYY20`).
    - `shifted_chol_qr(A, maxiter=np.inf, orth_tol=<some_number>)` is equivalent to the shifted
      CholeskyQR algorithm (Algorithm 4.1 in :cite:`FKNYY20`).
    - `shifted_chol_qr(A, product=<some_product>, maxiter=3, orth_tol=None)` is equivalent to the
      shifted CholeskyQR3 algorithm in an oblique inner product (Algorithm 5.1 in :cite:`FKNYY20`).

    .. note::
        Note that the unshifted single iteration CholeskyQR algorithm is unstable.
        Setting `maxiter=1` will perform the simple shifted CholeskyQR algorithm (Algorithm 2.1
        in :cite:`FKNYY20`) which is stable but leads to poor orthogonality in general, i.e.

        .. math::
            \lVert Q^TQ - I \rVert_2 < 2,

        (see Lemma 3.1 in :cite:`FKNYY20`).

    Parameters
    ----------
    A
        The |VectorArray| which is to be orthonormalized.
    product
        The inner product |Operator| w.r.t. which to orthonormalize.
        If `None`, the Euclidean product is used.
    return_R
        If `True`, the R matrix from QR decomposition is returned.
    maxiter
        Maximum number of iterations. Defaults to 3.
    offset
        Assume that the first `offset` vectors are already orthonormal and apply the
        algorithm for the vectors starting at `offset + 1`.
    orth_tol
        If not `None`, check if the resulting |VectorArray| is really orthornormal and
        repeat the algorithm until the check passes or `maxiter` is reached.
    recompute_shift
        If `False`, the shift is computed just once if the Cholesky decomposition fails
        and reused in possible further iterations. However, the initial shift might be too large
        for further iterations, which would lead to a non-orthonormal basis.
        If `True`, the shift is recomputed in iterations in which the Cholesky decomposition fails.
        Even for an ill-conditioned `A` (at least for matrix condition numbers up to 10^20)
        is it able to compute an orthonormal basis at the cost of higher runtimes.
    check_finite
        This argument is passed down to |SciPy linalg| functions. Disabling may give a
        performance gain, but may result in problems (crashes, non-termination) if the
        inputs do contain infinities or NaNs.
    copy
        If `True`, create a copy of `A` instead of modifying `A` in-place.
    product_norm
        The spectral norm of the inner product. If `None`, it will be computed with
        :func:`~pymor.algorithms.eigs.eigs`.

    Returns
    -------
    Q
        The orthonormalized |VectorArray|.
    R
        The upper-triangular/trapezoidal matrix (if `compute_R` is `True`).
    """
    params = _CholQRParameters(**locals())
    return _solve_chol_qr(params)


class _CholQRParameters:
    r"""Helper class for managing the parameters and options."""

    def __init__(self, A, product, return_R, maxiter, offset, orth_tol,
                 recompute_shift, check_finite, copy, product_norm):
        assert isinstance(A, VectorArray)
        assert 0 <= offset <= len(A)
        assert A.dim >= len(A)
        assert 0 < maxiter
        assert orth_tol is None or 0 < orth_tol

        # set input parameters
        self.A = A.copy() if copy else A
        self.product = product
        self.return_R = return_R
        self.maxiter = maxiter
        self.offset = offset
        self.orth_tol = orth_tol
        self.recompute_shift = recompute_shift
        self.check_finite = check_finite
        self.product_norm = product_norm

        self.dtype = None
        self.eps = None
        self.logger = getLogger('pymor.algorithms.chol_qr.shifted_chol_qr')


def _compute_gramian_and_offset_matrix(params):
    A = params.A
    offset = params.offset
    product = params.product

    if isinstance(A, ListVectorArray):
        # for a |ListVectorArray| it is slightly faster to compute `B` and `X` separately
        B = A[:offset].inner(A[offset:], product=product)
        X = A[offset:].gramian(product)
    else:
        B, X = np.split(A.inner(A[offset:], product=product), [offset], axis=0)

    if params.dtype is None:
        params.dtype = np.promote_types(X.dtype, np.promote_types(B.dtype, np.float32))
        params.eps = np.finfo(params.dtype).eps

    dtype = params.dtype
    B = B.astype(dtype=dtype, copy=False)
    X = X.astype(dtype=dtype, copy=False)

    return B, X


def _compute_shift(params: _CholQRParameters, X: np.ndarray):
    m = params.A.dim
    n = len(X)
    product_norm = params.product_norm

    shift = 11*params.eps
    if params.product is None:
        shift *= m*n+n*(n+1)
        XX = X
    else:
        if product_norm is None:
            from pymor.algorithms.eigs import eigs
            product_norm = np.sqrt(np.abs(eigs(params.product, k=1)[0][0]))
        shift *= (2*m*np.sqrt(m*n)+n*(n+1))*product_norm
        XX = params.A[params.offset:].gramian()
    try:
        shift *= spsla.eigsh(XX, k=1, tol=1e-2, return_eigenvectors=False, v0=np.ones([n]))[0]
    except spsla.ArpackNoConvergence as e:
        params.logger.warning(f'ARPACK failed with: {e}')
        params.logger.info('Proceeding with dense solver.')
        shift *= spla.eigh(XX, eigvals_only=True, subset_by_index=[n-1, n-1], driver='evr')[0]
    shift = max(shift, params.eps)  # ensure that shift is non-zero
    return shift


def _solve_chol_qr(params: _CholQRParameters):
    # unpack often used arguments
    A = params.A
    offset = params.offset

    if offset == len(A):
        return A, np.eye(len(A))

    if params.maxiter == 1:
        params.logger.warning('Single iteration shifted CholeskyQR can lead to poor orthogonality!')

    B, X = _compute_gramian_and_offset_matrix(params)

    trmm, trtri = spla.get_blas_funcs('trmm', dtype=params.dtype), spla.get_lapack_funcs('trtri', dtype=params.dtype)

    # compute shift
    shift = None

    iter = 1
    while iter <= params.maxiter:
        with params.logger.block(f'Iteration {iter}'):
            # This will compute the Cholesky factor of the lower right block
            # and keep applying shifts if it breaks down.
            X -= B.conj().T@B
            it = 0
            while True:
                try:
                    Rx = spla.cholesky(X, overwrite_a=False, check_finite=params.check_finite)
                    break
                except spla.LinAlgError:
                    it += 1
                    if it > 100:
                        assert False
                    if not shift or params.recompute_shift and it == 1:
                        shift = _compute_shift(params, X)
                    params.logger.warning('Cholesky factorization broke down! Matrix is ill-conditioned.')
                    params.logger.info(f'Applying shift: {shift}')
                    X[np.diag_indices_from(X)] += shift

            # orthogonalize
            Rinv = trtri(Rx)[0]
            A_todo = A[:offset].lincomb(-B@Rinv) + A[offset:].lincomb(Rinv)
            del A[offset:]
            A.append(A_todo)

            # update blocks of R
            if iter == 1:
                Bi = B
                Ri = Rx
            else:
                Bi += B @ Ri
                Ri = trmm(1, Rx, Ri, overwrite_b=True)

            # computation not needed in the last iteration
            if iter < params.maxiter:
                B, X = _compute_gramian_and_offset_matrix(params)
            elif params.orth_tol is not None:
                X = A[offset:].gramian(product=params.product)

            # check orthonormality (for an iterative algorithm)
            if params.orth_tol is not None:
                res = spla.norm(X - np.eye(len(A) - offset), ord='fro', check_finite=params.check_finite)
                params.logger.info(f'Residual = {res}')
                if res <= params.orth_tol*np.sqrt(len(A)):
                    break
                elif iter == params.maxiter:
                    raise AccuracyError('Orthonormality could not be achieved within the given tolerance. \
                    Consider increasing maxiter or enabling recompute_shift.')

            iter += 1

    # construct R from blocks
    R = np.eye(len(A), dtype=params.dtype)
    R[:offset, offset:] = Bi
    R[offset:, offset:] = Ri

    return (A, R) if params.return_R else A
