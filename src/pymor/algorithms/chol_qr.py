# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spsla

from pymor.core.defaults import defaults
from pymor.core.exceptions import AccuracyError
from pymor.core.logger import getLogger
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.list import ListVectorArray

_INNER_ITERS = 10

@defaults('maxiter', 'orth_tol', 'recompute_shift', 'rtol', 'check_finite')
def shifted_chol_qr(A, product=None, return_R=False, maxiter=3, offset=0, orth_tol=None,
                    recompute_shift=False, rtol=1e-13, check_finite=True, copy=True, product_norm=None):
    r"""Orthonormalize a |VectorArray| using the shifted CholeskyQR algorithm.

    This method computes a QR decomposition of a |VectorArray| via Cholesky factorizations
    of its Gramian matrix according to :cite:`FKNYY20`. For ill-conditioned matrices, the Cholesky
    factorization will break down. In this case a diagonal shift will be applied to the Gramian.

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
        If zero, it removes zero vectors. Otherwise, if greater than zero, it removes too
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
    return _CholQRStruct(**locals()).solve()


class _CholQRStruct:
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

        # define additional variables required inside subroutines
        self.m, self.n = A.dim, len(A[offset:])
        self.shift = 0
        self.dtype = None
        self.eps = None
        self.chol_kernel = self._recomputed_shifted_chol_kernel if recompute_shift else self._basic_shifted_chol_kernel
        self.logger = getLogger('pymor.algorithms.chol_qr.shifted_chol_qr')

        if self.maxiter == 1:
            self.logger.warning('Single iteration shifted CholeskyQR can lead to poor orthogonality!')


    def _compute_gramian_and_offset_matrix(self):
        r"""Helper routine for `shifted_chol_qr` perfoming multiple steps.

        1. Given an offset `A = [Q1, A2]`, the routine computes `B = Q1.H @ A2` and
           `X = A2.H @ A2`. If no offset is given, `B` is going to be dimensionless.
        2. Evaluates a unifiable datatype of `B` and `X` and casts both of them into that type.
           In particular for `ListVectorArray`s there might be a type mismatch.
           Sets the datatype and required LAPACK routine wrappers as class arguments.
        3. If a offset is given, a term `B.H@B` is subtracted from `X`.
        """
        A = self.A
        product = self.product
        offset = self.offset

        # Step 1
        if isinstance(A, ListVectorArray):
            # for a |ListVectorArray| it is slightly faster to compute `B` and `X` separately
            B = A[:offset].inner(A[offset:], product=product)
            X = A[offset:].gramian(product)
        else:
            B, X = np.split(A.inner(A[offset:], product=product), [offset], axis=0)

        # Step 2
        if self.dtype:
            dtype = self.dtype
        else:
            self.dtype = dtype = np.promote_types(X.dtype, np.promote_types(B.dtype, np.float32))
            self.eps = np.finfo(dtype).eps
            self.trmm = spla.get_blas_funcs('trmm', dtype=dtype)
            self.trtri = spla.get_lapack_funcs('trtri', dtype=dtype)

        B = B.astype(dtype=dtype, copy=False)
        X = X.astype(dtype=dtype, copy=False)

        # Step 3
        X -= B.conj().T@B

        return B, X


    def _compute_shift(self, X):
        m = self.m
        n = len(X)

        shift = 11*self.eps
        if self.product is None:
            shift *= m*n+n*(n+1)
        else:
            if self.product_norm is None:
                from pymor.algorithms.eigs import eigs
                self.product_norm = np.sqrt(np.abs(eigs(self.product, k=1)[0][0]))
            shift *= (2*m*np.sqrt(m*n)+n*(n+1))*self.product_norm

        # eigsh outputs warnings, if n <= 2; it also throws an exception,
        # if X is a zero matrix (or is close to) or contains subnormal numbers
        use_eigh = n <= 2 or X.max() - X.min() < self.eps or np.any((X != 0) & (np.abs(X) < np.finfo(self.dtype).tiny))
        if not use_eigh:
            try:
                ew = spsla.eigsh(X, k=1, tol=1e-2, return_eigenvectors=False, v0=np.ones([n]))[0]
            except spsla.ArpackNoConvergence as e:
                self.logger.warning(f'ARPACK failed with: {e}')
                self.logger.info('Proceeding with dense solver.')
                use_eigh = True

        if use_eigh:
            ew = spla.eigh(X, eigvals_only=True, subset_by_index=[n-1, n-1], driver='evr')[0]

        shift = max(shift*ew, self.eps) # ensure that shift is non-zero
        return shift


    def _basic_shifted_chol_kernel(self, X):
        global _INNER_ITERS
        for _ in range(_INNER_ITERS):
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


    def _recomputed_shifted_chol_kernel(self, X):
        global _INNER_ITERS
        shift = 0 # does not use self.shift; recomputes it in every CholQR iteration
        for i in range(_INNER_ITERS):
            try:
                R = spla.cholesky(X + np.eye(len(X))*shift, overwrite_a=False, check_finite=self.check_finite)
                return R
            except spla.LinAlgError:
                self.logger.warning('Cholesky factorization broke down! Matrix is ill-conditioned.')

                # for really ill-conditioned matrices increasing the shift exponentially,
                # by multiplying it by 10 in each iteration, might improve convergence
                # as shown in https://doi.org/10.48550/arXiv.2507.07788
                shift = self._compute_shift(X) if i == 0 else shift*10
                self.logger.info(f'Applying shift: {shift}')
        raise AccuracyError('Failed to compute a Cholesky factorization of X.')


    def solve(self):
        r"""Computes QR factorization of A in place.

        Given an input matrix `A = [Q1, A2]` with
        - `Q1` being orthogonal and of dimension `m x offset`
        - and `A2` of dimension `m x n`.

        Consider the Gramian `X = A.H * A = [Q1.H * Q1, Q1.H * A2] = [I,   B ]`,
                                           `[A2.H * Q1, A2.H * A2] = [B.H, X2]`

        its Cholesky factorization is given by `R = [I, B ]` with `R2 = X2 - B.H * B`.
                                                   `[0, R2]`

        Then, the the QR factorization of `A` is given by
        `A = [Q1, A2] = [Q1, Q2] * R` with `R = [I, B ]`. Hence, `Q2 = (A2 - Q1*B) * inv(R2)`.
                                               `[0, R2]`

        In case the offset is 0, i.e., the `Q1` has 0 vectors,
        the other matrices `I` and `B` are also dimensionless.

        One can perform multiple iterations in order to reduce errors. Here, we use the
        loss of orthogonality (||Q.H*Q - I||F = ||X - I||F) as a stopping criterium.

        Consider we have a QR decomposition `qr(A) = QR`. In a second iteration one would compute
        `qr(Q) * R = Q^*R^ * R`. `Q^` becomes the solution `Q$`
        and `R^ * R` the solution factor `R$`. Given the structure of the `R` factors,
        one only has to evaluate `B$ = B + B^*R2` and `R2$ = R2^.H *R2`.
        """
        A = self.A
        offset = self.offset

        if offset == len(A) or A.dim == 0:
            return (A, np.eye(len(A))) if self.return_R else A

        Ri = Bi = None # accumulated R and B over multiple iterations
        A_rem = B_rem = None

        B, X = self._compute_gramian_and_offset_matrix()

        # diagonal of X contains the pairwise result of inner-products
        # the vectors A[offset:]
        diag = np.abs(np.diag(X))
        M = np.max(diag)
        if self.offset > 0:
            M = max(M, 1) # length of orthogonal vectors is 1

        # find vectors that are to short relative to the longest vector in A
        # used squared relative tolerance, since diagonal contains squared norms of vectors
        remove = np.where(M*self.rtol**2 >= diag)[0]
        if len(remove) == self.n:
            del A[offset:]
            return (A, np.hstack([np.eye(offset), B])) if self.return_R else A
        elif len(remove) > 0:
            B_rem = B[:,remove]
            B = np.delete(B, remove, axis=1)
            X = np.delete(np.delete(X, remove, axis=0), remove, axis=1)
            remove += offset
            A_rem = A[remove].copy()
            del A[remove]
            self.n -= len(remove)

        for iter in range(1,self.maxiter+1):
            with self.logger.block(f'Iteration {iter}'):
                # This will compute the Cholesky factor of the upper right block
                # depending on the kernel it might apply shifts if it breaks down
                Rx = self.chol_kernel(X)

                # update blocks of full R
                if Ri is None:
                    Ri = Rx
                    Bi = B
                else:
                    Bi += B @ Ri
                    # does not properly overwrite Ri
                    Ri = self.trmm(1, Rx, Ri, overwrite_b=True)

                # orthogonalize
                Rinv = self.trtri(Rx)[0]
                A_todo = A[:offset].lincomb(-B@Rinv) + A[offset:].lincomb(Rinv)
                del A[offset:]
                A.append(A_todo)
                del A_todo

                # computation not needed in the last iteration
                if iter < self.maxiter:
                    B, X = self._compute_gramian_and_offset_matrix()
                elif self.orth_tol is not None:
                    X = A[offset:].gramian(product=self.product)

                # check orthonormality (for an iterative algorithm)
                if self.orth_tol is not None:
                    res = spla.norm(X - np.eye(len(A) - offset), ord='fro', check_finite=self.check_finite)
                    self.logger.info(f'Residual = {res}')
                    if res <= self.orth_tol*np.sqrt(len(A)):
                        break
                    elif iter == self.maxiter:
                        raise AccuracyError(
"""Orthonormality could not be achieved within the given tolerance.
Consider increasing maxiter or enabling recompute_shift or using gram_schmidt."""
                        )

        if not self.return_R:
            return A

        # construct R from blocks
        R = np.zeros([offset+Ri.shape[0], offset+self.n], dtype=self.dtype)
        R[:offset,:offset] = np.eye(offset)
        R[:offset, offset:] = Bi.astype(self.dtype, copy=False)
        R[offset:, offset:] = Ri.astype(self.dtype, copy=False)

        if B_rem is not None:
            # compute linear dependence of removed vectors to the orthnormal basis
            # and insert them back into R
            T = A[offset:].inner(A_rem, product=self.product)
            R = np.insert(R, [r-i for i,r in enumerate(remove)], values=np.vstack([B_rem, T]), axis=1)

        return (A, R)
