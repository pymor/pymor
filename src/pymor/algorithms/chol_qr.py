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

_INNER_ITERS = 10

def shifted_chol_qr(A, product=None, return_R=False, maxiter=3, offset=0, orth_tol=None,
                    recompute_shift=False, rtol=1e-13, check_finite=True, copy=True, product_norm=None):
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
    rtol
        If zero, it only removes zero vectors. Otherwise, if greater than zero, it removes too
        small vectors (relative to the longest vector). Furthermore, if offset is used,
        it might remove linearly dependent vectors. Decision is based on the diagonal of
        the initial Gramian. Additionally, if `return_R` is set, a potentially
        upper-trapezoidal factor `R` is returned. It holds `A \approx Q@R`.
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
                 recompute_shift, rtol, check_finite, copy, product_norm):
        assert isinstance(A, VectorArray)
        assert 0 <= offset <= len(A)
        assert A.dim >= len(A)
        assert 0 < maxiter
        assert orth_tol is None or 0 < orth_tol
        assert rtol >= 0

        # set input parameters
        self.A = A.copy() if copy else A
        self.product = product
        self.return_R = return_R
        self.maxiter = maxiter
        self.offset = offset
        self.orth_tol = orth_tol
        self.rtol = rtol
        self.check_finite = check_finite
        self.product_norm = product_norm

        self.chol_kernel = _recomputed_shifted_chol_kernel if recompute_shift else _basic_shifted_chol_kernel
        self.shift = None # used by `_basic_shifted_chol_kernel``
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

    X -= B.conj().T@B

    return B, X


def _compute_shift(params: _CholQRParameters, X: np.ndarray):
    m = params.A.dim
    n = len(X)

    shift = 11*params.eps
    if params.product is None:
        shift *= m*n+n*(n+1)
    else:
        if params.product_norm is None:
            from pymor.algorithms.eigs import eigs
            params.product_norm = np.sqrt(np.abs(eigs(params.product, k=1)[0][0]))
        shift *= (2*m*np.sqrt(m*n)+n*(n+1))*params.product_norm

    # eigsh outputs warnings, if n <= 2; it also throws an exception,
    # if X is a zero matrix (or is close to) or contains subnormal numbers
    use_eigh = n <= 2 or X.max() - X.min() < params.eps or np.any((X != 0) & (np.abs(X) < np.finfo(params.dtype).tiny))
    if not use_eigh:
        try:
            ew = spsla.eigsh(X, k=1, tol=1e-2, return_eigenvectors=False, v0=np.ones([n]))[0]
        except spsla.ArpackNoConvergence as e:
            params.logger.warning(f'ARPACK failed with: {e}')
            params.logger.info('Proceeding with dense solver.')
            use_eigh = True

    if use_eigh:
        ew = spla.eigh(X, eigvals_only=True, subset_by_index=[n-1, n-1], driver='evr')[0]

    shift = max(shift*ew, params.eps) # ensure that shift is non-zero
    return shift


def _basic_shifted_chol_kernel(params, X):
    """Kernel computes `R` and shift according to Algorithm 4.1 in :cite:`FKNYY20`."""
    global _INNER_ITERS
    for _ in range(_INNER_ITERS):
        try:
            R = spla.cholesky(X, overwrite_a=False, check_finite=params.check_finite)
            return R
        except spla.LinAlgError:
            params.logger.warning('Cholesky factorization broke down! Matrix is ill-conditioned.')

            if not params.shift:
                params.shift = _compute_shift(params, X)
            params.logger.info(f'Applying shift: {params.shift}')
            X[np.diag_indices_from(X)] += params.shift
    raise AccuracyError('Failed to compute a Cholesky factorization of X.')


def _recomputed_shifted_chol_kernel(params, X):
    """Kernel computes `R` and shift according to :cite:`BPS26`."""
    global _INNER_ITERS
    shift = 0 # does not use self.shift; recomputes it in every CholQR iteration
    for i in range(_INNER_ITERS):
        try:
            R = spla.cholesky(X + np.eye(len(X))*shift, overwrite_a=False, check_finite=params.check_finite)
            return R
        except spla.LinAlgError:
            params.logger.warning('Cholesky factorization broke down! Matrix is ill-conditioned.')

            # for really ill-conditioned matrices increasing the shift exponentially,
            # by multiplying it by 10 in each iteration
            shift = _compute_shift(params, X) if i == 0 else shift*10
            params.logger.info(f'Applying shift: {shift}')
    raise AccuracyError('Failed to compute a Cholesky factorization of X.')


def _solve_chol_qr(params: _CholQRParameters):
    # unpack often used arguments
    A = params.A
    offset = params.offset

    if offset == len(A):
        return (A, np.eye(len(A))) if params.return_R else A

    if params.maxiter == 1:
        params.logger.warning('Single iteration shifted CholeskyQR can lead to poor orthogonality!')

    B, X = _compute_gramian_and_offset_matrix(params)

    trmm, trtri = spla.get_blas_funcs('trmm', dtype=params.dtype), spla.get_lapack_funcs('trtri', dtype=params.dtype)

    # diagonal of X contains the pairwise result of inner-products of the vectors A[offset:]
    diag = np.abs(np.diag(X))
    M = np.max(diag)
    if params.offset > 0:
        M = max(M, 1) # length of orthonormal vectors is 1

    # find vectors that are to short relative to the longest vector in A
    # used squared relative tolerance, since diagonal contains squared norms of vectors
    A_rem = B_rem = None
    remove = np.where(M*params.rtol**2 >= diag)[0]
    if len(remove) > 0:
        params.logger.info(f'Removing linearly dependent vector {remove}')

    if len(remove) == len(A[offset:]):
        del A[offset:]
        return (A, np.hstack([np.eye(offset), B])) if params.return_R else A
    elif len(remove) > 0:
        B_rem = B[:,remove]
        B = np.delete(B, remove, axis=1)
        X = np.delete(np.delete(X, remove, axis=0), remove, axis=1)
        remove += offset
        A_rem = A[remove].copy()
        del A[remove]

    for iter in range(1,params.maxiter+1):
        with params.logger.block(f'Iteration {iter}'):
            Rx = params.chol_kernel(params, X)

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

    if not params.return_R:
        return A

    # construct R from blocks
    R = np.zeros([offset+Ri.shape[0], offset+Ri.shape[1]], dtype=params.dtype)
    R[:offset,:offset] = np.eye(offset)
    R[:offset, offset:] = Bi.astype(params.dtype, copy=False)
    R[offset:, offset:] = Ri.astype(params.dtype, copy=False)

    if B_rem is not None:
        # compute linear dependence of removed vectors to the orthonormal basis
        # and insert them back into R
        T = A[offset:].inner(A_rem, product=params.product)
        R = np.insert(R, [r-i for i,r in enumerate(remove)], values=np.vstack([B_rem, T]), axis=1)

    return (A, R)
