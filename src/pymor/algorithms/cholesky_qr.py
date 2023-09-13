#!/usr/bin/env python3

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.eigs import eigs
from pymor.core.logger import getLogger

logger = getLogger('pymor.algorithms.cholesky_qr.cholesky_qr')


def _chol_qr(V, product=None, gramian=None, check_finite=True):
    X = V.gramian(product=product) if gramian is None else gramian

    while True:
        try:
            R = spla.cholesky(X, overwrite_a=True, check_finite=check_finite)
            break
        except spla.LinAlgError:
            logger.warning('Cholesky factorization broke down! Matrix is ill-conditioned.')
            m, n = V.dim, len(V)
            eps = np.finfo(X.dtype).eps
            s = m*n+n*(n+1) if product is None else 2*m*np.sqrt(m*n)+n*(n+1)*np.sqrt(np.abs(eigs(product, k=1)[0][0]))
            s *= 11*eps*spla.norm(X, ord=2, check_finite=check_finite)
            logger.info(f'Applying shift: {s}')
            np.fill_diagonal(X, np.diag(X)+s)

    return V.lincomb(spla.lapack.dtrtri(R)[0].T), R


def cholesky_qr(V, product=None, maxiter=3, tol=None, check_finite=True, return_R=True, copy=True):
    if copy:
        V = V.copy()

    iter = 1
    with logger.block(f'Iteration {iter}'):
        Q, R = _chol_qr(V, product=product, gramian=None, check_finite=check_finite)

    gramian = None
    while iter < maxiter:
        iter += 1

        if tol is not None:
            gramian = Q.gramian(product=product)
            res = spla.norm(gramian - np.eye(len(V)), ord='fro', check_finite=check_finite)
            logger.info(f'Residual = {res}')
            if res <= tol*np.sqrt(len(V)):
                break

        with logger.block(f'Iteration {iter}'):
            Q, S = _chol_qr(Q, product=product, gramian=gramian, check_finite=check_finite)

            if return_R:
                spla.blas.dtrmm(1, S, R, overwrite_b=True)

    if return_R:
        return Q, R
    else:
        return Q
