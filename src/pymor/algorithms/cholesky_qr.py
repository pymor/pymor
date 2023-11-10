#!/usr/bin/env python3

import numpy as np
import scipy.linalg as spla

from pymor.core.logger import getLogger

logger = getLogger('pymor.algorithms.cholesky_qr.shifted_cholqr')


def shifted_cholqr(A, product=None, return_R=True, maxiter=3, offset=0, orth_tol=None, check_finite=True, copy=True):
    r"""Orthonormalize a |VectorArray| using the shifted CholeskyQR algorithm.

    This method computes a QR decomposition of a |VectorArray| via Cholesky factorizations
    of its Gramian matrix according to :cite:`FKNYY20`. For ill-conditioned matrices, the Cholesky
    factorization will break down. In this case a diagonal shift will be applied to the Gramian.

    - `shifted_cholqr(A, maxiter=3, orth_tol=None)` is equivalent to the shifted CholeskyQR3
    algorithm (Algorithm 4.2 in :cite:`FKNYY20`).
    - `shifted_cholqr(A, maxiter=np.inf, orth_tol=<some_number>)` is equivalent to the shifted
    CholeskyQR algorithm (Algorithm 4.1 in :cite:`FKNYY20`).
    - `shifted_cholqr(A, product=<some_product>, maxiter=3, orth_tol=None)` is equivalent to the
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
    check_finite
        This argument is passed down to |SciPy linalg| functions. Disabling may give a
        performance gain, but may result in problems (crashes, non-termination) if the
        inputs do contain infinities or NaNs.
    copy
        If `True`, create a copy of `A` instead of modifying `A` in-place.

    Returns
    -------
    Q
        The orthonormalized |VectorArray|.
    R
        The upper-triangular/trapezoidal matrix (if `compute_R` is `True`).
    """
    assert 0 <= offset < len(A)
    assert 0 < maxiter
    assert orth_tol is None or 0 < orth_tol

    if maxiter == 1:
        logger.warning('Single iteration shifted CholeskyQR can lead to poor orthogonality!')

    if copy:
        A = A.copy()

    iter = 1
    shift = None
    while iter <= maxiter:
        with logger.block(f'Iteration {iter}'):
            if orth_tol is None or iter == 1:
                # compute only the necessary parts of the Gramian
                B, X = np.split(A[offset:].inner(A, product=product), [offset], axis=1)

            # This will compute the Cholesky factor of the lower right block
            # and keep applying shifts if it breaks down.
            while True:
                try:
                    Rx = spla.cholesky(X, overwrite_a=True, check_finite=check_finite)
                    break
                except spla.LinAlgError:
                    logger.warning('Cholesky factorization broke down! Matrix is ill-conditioned.')
                    if shift is None:
                        m, n = A.dim, len(A[offset:])
                        if product is None:
                            shift = m*n+n*(n+1)
                            XX = X
                        else:
                            from pymor.algorithms.eigs import eigs
                            shift = 2*m*np.sqrt(m*n)+n*(n+1)*np.sqrt(np.abs(eigs(product, k=1)[0][0]))
                            XX = A[offset:].gramian(product=product)
                        eps = np.finfo(XX.dtype).eps
                        shift *= 11*eps*spla.norm(XX, ord=2, check_finite=check_finite)

                    logger.info(f'Applying shift: {shift}')
                    idx = range(len(X))
                    X[idx, idx] += shift

            # orthogonalize
            Rinv = spla.lapack.dtrtri(Rx)[0].T
            A_todo = A[:offset].lincomb(-Rinv@B) + A[offset:].lincomb(Rinv)
            del A[offset:]
            A.append(A_todo)

            # update blocks of R
            if return_R:
                if iter == 1:
                    Bi = B.T
                    Ri = Rx
                else:
                    Bi += B.T @ Rx
                    spla.blas.dtrmm(1, Rx, Ri, overwrite_b=True)

            # check orthonormality (for an iterative algorithm)
            if orth_tol is not None:
                B, X = np.split(A[offset:].inner(A, product=product), [offset], axis=1)
                res = spla.norm(X - np.eye(len(A) - offset), ord='fro', check_finite=check_finite)
                logger.info(f'Residual = {res}')
                if res <= orth_tol*np.sqrt(len(A)):
                    break

            iter += 1

    return_args = [A]
    if return_R:
        # construct R from blocks
        R = np.eye(len(A))
        R[:offset, offset:] = Bi
        R[offset:, offset:] = Ri
        return_args.append(R)

    return tuple(return_args)
