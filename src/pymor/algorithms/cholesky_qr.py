#!/usr/bin/env python3

import numpy as np
import scipy.linalg as spla

from pymor.core.logger import getLogger

logger = getLogger('pymor.algorithms.cholesky_qr.cholesky_qr')


def _shifted_cholesky(X, m, n, product=None, check_finite=True):
    """This will return the Cholesky factor and keep applying shifts if it breaks down."""
    while True:
        try:
            R = spla.cholesky(X, overwrite_a=True, check_finite=check_finite)
            break
        except spla.LinAlgError:
            logger.warning('Cholesky factorization broke down! Matrix is ill-conditioned.')
            eps = np.finfo(X.dtype).eps
            if product is None:
                s = m*n+n*(n+1)
                s *= 11*eps*spla.norm(X, ord=2, check_finite=check_finite)
            else:
                from pymor.algorithms.eigs import eigs
                s = 2*m*np.sqrt(m*n)+n*(n+1)
                # TODO fix norm estimation!
                s *= 11*eps*spla.norm(X, ord=2, check_finite=check_finite)**2
                s *= np.sqrt(np.abs(eigs(product, k=1)[0][0]))
            logger.info(f'Applying shift: {s}')
            np.fill_diagonal(X, np.diag(X) + s)

    return R


def cholesky_qr(A, product=None, maxiter=5, tol=None, check_finite=True, return_R=True, offset=0):
    """No copy kwarg, because we cannot work in place anyway."""
    assert 0 <= offset < len(A)
    m, n = A.dim, len(A) - offset

    iter = 1
    while iter <= maxiter:
        with logger.block(f'Iteration {iter}'):
            if tol is None or iter == 1:
                # compute only the necessary parts of the Gramian
                A_orth, A_todo = A[:offset].copy(), A[offset:]
                B, X = np.split(A_todo.inner(A, product=product), [offset], axis=1)

            # compute Cholesky factor of lower right block
            Rx = _shifted_cholesky(X, m, n, product=product, check_finite=check_finite)
            Rinv = spla.lapack.dtrtri(Rx)[0].T
            if offset == 0:
                A = A.lincomb(Rinv)
            else:
                A_orth.append(A_orth.lincomb(-Rinv@B) + A_todo.lincomb(Rinv))
                A = A_orth

            # update blocks of R
            if return_R:
                if iter == 1:
                    Bi = B.T
                    Ri = Rx
                else:
                    Bi += B.T @ Rx
                    spla.blas.dtrmm(1, Rx, Ri, overwrite_b=True)

            # check orthogonality
            if tol is not None:
                A_orth, A_todo = A[:offset].copy(), A[offset:]
                B, X = np.split(A_todo.inner(A, product=product), [offset], axis=1)
                res = spla.norm(X - np.eye(n), ord='fro', check_finite=check_finite)
                logger.info(f'Residual = {res}')
                if res <= tol*np.sqrt(n):
                    break

            iter += 1

    if return_R:
        # construct R from blocks
        R = np.eye(len(A))
        R[:offset, offset:] = Bi
        R[offset:, offset:] = Ri
        return A, R
    else:
        return A
