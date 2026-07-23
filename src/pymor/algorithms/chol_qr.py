# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spsla

from pymor.core.base import BasicObject, abstractmethod
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
    assert isinstance(A, VectorArray)
    assert 0 <= offset <= len(A)
    assert A.dim >= len(A)
    assert 0 < maxiter
    assert orth_tol is None or 0 < orth_tol
    logger = getLogger('pymor.algorithms.chol_qr.shifted_chol_qr')

    if copy:
        A = A.copy()

    chol_kernel = (RecomputedShiftedCholQRKernel if recompute_shift else BasicShiftedCholQRKernel)(
        A.dim, product=product, product_norm=product_norm, check_finite=check_finite
    )

    if offset == len(A):
        return (A, np.eye(len(A))) if return_R else A

    if maxiter == 1:
        logger.warning('Single iteration shifted CholeskyQR can lead to poor orthogonality!')

    B, X = _compute_gramian_and_offset_matrix(A, offset, product)

    trmm, trtri = spla.get_blas_funcs('trmm', dtype=X.dtype), spla.get_lapack_funcs('trtri', dtype=X.dtype)

    iter = 1
    while iter <= maxiter:
        with logger.block(f'Iteration {iter}'):
            X -= B.conj().T@B
            Rx = chol_kernel.apply(X)

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
            if iter < maxiter:
                B, X = _compute_gramian_and_offset_matrix(A, offset, product)
            elif orth_tol is not None:
                X = A[offset:].gramian(product=product)

            # check orthonormality (for an iterative algorithm)
            if orth_tol is not None:
                res = spla.norm(X - np.eye(len(A) - offset), ord='fro', check_finite=check_finite)
                logger.info(f'Residual = {res}')
                if res <= orth_tol*np.sqrt(len(A)):
                    break
                elif iter == maxiter:
                    raise AccuracyError('Orthonormality could not be achieved within the given tolerance. \
                    Consider increasing maxiter or enabling recompute_shift.')

            iter += 1

    if not return_R:
        return A

    # construct R from blocks
    R = np.eye(len(A), dtype=X.dtype)
    R[:offset, offset:] = Bi
    R[offset:, offset:] = Ri

    return (A, R)


def _compute_gramian_and_offset_matrix(A, offset, product):
    if isinstance(A, ListVectorArray):
        # for a |ListVectorArray| it is slightly faster to compute `B` and `X` separately
        B = A[:offset].inner(A[offset:], product=product)
        X = A[offset:].gramian(product)
    else:
        B, X = np.split(A.inner(A[offset:], product=product), [offset], axis=0)

    dtype = np.promote_types(X.dtype, np.promote_types(B.dtype, np.float32))
    B = B.astype(dtype=dtype, copy=False)
    X = X.astype(dtype=dtype, copy=False)

    return B, X


class ShiftedCholQRKernel(BasicObject):
    """Abstract base class for shifted_chol_qr kernels."""

    def __init__(self, dim, product=None, product_norm=None, check_finite=True):
        self.__auto_init(locals())

    @abstractmethod
    def apply(self, X): ...

    def _compute_shift(self, X):
        m = self.dim
        n = len(X)
        dtype = X.dtype
        eps = np.finfo(dtype).eps

        shift = 11*eps
        if self.product is None:
            shift *= m*n+n*(n+1)
        else:
            if self.product_norm is None:
                from pymor.algorithms.eigs import eigs
                self.product_norm = np.sqrt(np.abs(eigs(self.product, k=1)[0][0]))
            shift *= (2*m*np.sqrt(m*n)+n*(n+1))*self.product_norm

        # eigsh outputs warnings, if n <= 2; it also throws an exception,
        # if X is a zero matrix (or is close to) or contains subnormal numbers
        # see https://github.com/pymor/pymor/pull/2570#issuecomment-5045868061
        use_eigh = n <= 2 or X.max() - X.min() < eps or np.any((X != 0) & (np.abs(X) < np.finfo(dtype).tiny))
        if not use_eigh:
            try:
                ew = spsla.eigsh(X, k=1, tol=1e-2, return_eigenvectors=False, v0=np.ones([n]))[0]
            except spsla.ArpackNoConvergence as e:
                self.logger.warning(f'ARPACK failed with: {e}')
                self.logger.info('Proceeding with dense solver.')
                use_eigh = True

        if use_eigh:
            ew = spla.eigh(X, eigvals_only=True, subset_by_index=[n-1, n-1], driver='evr')[0]

        shift = max(shift*ew, eps) # ensure that shift is non-zero
        return shift


class BasicShiftedCholQRKernel(ShiftedCholQRKernel):
    """Kernel computes `R` and shift according to Algorithm 4.1 in :cite:`FKNYY20`."""

    INNER_ITERS = 10

    def apply(self, X):
        for _ in range(self.INNER_ITERS):
            try:
                R = spla.cholesky(X, overwrite_a=False, check_finite=self.check_finite)
                return R
            except spla.LinAlgError:
                self.logger.warning('Cholesky factorization broke down! Matrix is ill-conditioned.')

                if not self.shift:
                    self.shift = self._compute_shift(X)
                self.logger.info(f'Applying shift: {self.shift}')
                X[np.diag_indices_from(X)] += self.shift
        raise AccuracyError('Failed to compute a Cholesky factorization of X.')


class RecomputedShiftedCholQRKernel(ShiftedCholQRKernel):
    """Kernel computes `R` and shift according to :cite:`BPS26`."""

    INNER_ITERS = 10

    def apply(self, X):
        shift = 0 # does not use self.shift; recomputes it in every CholQR iteration
        for i in range(self.INNER_ITERS):
            try:
                R = spla.cholesky(X + np.eye(len(X))*shift, overwrite_a=False, check_finite=self.check_finite)
                return R
            except spla.LinAlgError:
                self.logger.warning('Cholesky factorization broke down! Matrix is ill-conditioned.')

                # for really ill-conditioned matrices increasing the shift exponentially,
                # by multiplying it by 10 in each iteration
                shift = self._compute_shift(X) if i == 0 else shift*10
                self.logger.info(f'Applying shift: {shift}')
        raise AccuracyError('Failed to compute a Cholesky factorization of X.')
