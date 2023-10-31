#!/usr/bin/env python3

import numpy as np
import scipy.linalg as spla

from pymor.core.logger import getLogger

logger = getLogger('pymor.algorithms.cholesky_qr.cholesky_qr')


def _chol_qr(A, product=None, gramian=None, check_finite=True):
    X = A.gramian(product=product) if gramian is None else gramian

    while True:
        try:
            R = spla.cholesky(X, overwrite_a=True, check_finite=check_finite)
            break
        except spla.LinAlgError:
            logger.warning('Cholesky factorization broke down! Matrix is ill-conditioned.')
            m, n = A.dim, len(A)
            eps = np.finfo(X.dtype).eps
            if product is None:
                s = m*n+n*(n+1)
                s *= 11*eps*spla.norm(X, ord=2, check_finite=check_finite)
            else:
                from pymor.algorithms.eigs import eigs
                s = 2*m*np.sqrt(m*n)+n*(n+1)
                s *= 11*eps*spla.norm(A, ord=2, check_finite=check_finite)**2
                s *= np.sqrt(np.abs(eigs(product, k=1)[0][0]))
            logger.info(f'Applying shift: {s}')
            np.fill_diagonal(X, np.diag(X) + s)

    return A.lincomb(spla.lapack.dtrtri(R)[0].T), R


def cholesky_qr(A, product=None, maxiter=3, tol=None, check_finite=True, return_R=True, copy=True):
    if copy:
        A = A.copy()

    iter = 1
    with logger.block(f'Iteration {iter}'):
        A, R = _chol_qr(A, product=product, gramian=None, check_finite=check_finite)

    while iter < maxiter:
        iter += 1
        if tol is None:
            X = None
        else:
            X = A.gramian(product=product)
            res = spla.norm(X - np.eye(len(A)), ord='fro', check_finite=check_finite)
            logger.info(f'Residual = {res}')
            if res <= tol*np.sqrt(len(A)):
                break

        with logger.block(f'Iteration {iter}'):
            A, S = _chol_qr(A, product=product, gramian=X, check_finite=check_finite)

            if return_R:
                spla.blas.dtrmm(1, S, R, overwrite_b=True)

    if return_R:
        return A, R
    else:
        return A
