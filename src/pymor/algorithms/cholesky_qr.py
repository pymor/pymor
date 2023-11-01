#!/usr/bin/env python3

import numpy as np
import scipy.linalg as spla

from pymor.core.base import BasicObject
from pymor.core.cache import CacheableObject, cached
from pymor.core.logger import getLogger

logger = getLogger('pymor.algorithms.cholesky_qr.cholesky_qr')


class _ProductNorm(CacheableObject):
    cache_region = 'memory'

    def __init__(self, product):
        self.__auto_init(locals())

    @cached
    def norm(self):
        if self.product is None:
            return 1
        else:
            from pymor.algorithms.eigs import eigs
            return np.sqrt(np.abs(eigs(self.product, k=1)[0][0]))


class _CholeskyShifter(BasicObject):
    def __init__(self, A, product, product_norm):
        self.__auto_init(locals())

    def compute_shift(self, X, check_finite=True):
        m, n = self.A.dim, len(self.A)
        if self.product is None:
            s = m*n+n*(n+1)
        else:
            s = 2*m*np.sqrt(m*n)+n*(n+1)*self.product_norm.norm()
            X = self.A.gramian(product=self.product)
        eps = np.finfo(X.dtype).eps
        s *= 11*eps*spla.norm(X, ord=2, check_finite=check_finite)
        return s


def _shifted_cholesky(X, shifter, check_finite=True):
    """This will return the Cholesky factor and keep applying shifts if it breaks down."""
    while True:
        try:
            R = spla.cholesky(X, overwrite_a=True, check_finite=check_finite)
            break
        except spla.LinAlgError:
            logger.warning('Cholesky factorization broke down! Matrix is ill-conditioned.')
            shift = shifter.compute_shift(X, check_finite=check_finite)
            logger.info(f'Applying shift: {shift}')
            np.fill_diagonal(X, np.diag(X) + shift)

    return R


def cholesky_qr(A, product=None, maxiter=5, tol=None, check_finite=True, return_R=True, offset=0):
    """No copy kwarg, because we cannot work in place anyway."""
    assert 0 <= offset < len(A)

    iter = 1
    product_norm = _ProductNorm(product)
    while iter <= maxiter:
        with logger.block(f'Iteration {iter}'):
            if tol is None or iter == 1:
                # compute only the necessary parts of the Gramian
                A_orth, A_todo = A[:offset].copy(), A[offset:]
                B, X = np.split(A_todo.inner(A, product=product), [offset], axis=1)

            # compute Cholesky factor of lower right block
            shifter = _CholeskyShifter(A_todo, product, product_norm)
            Rx = _shifted_cholesky(X, shifter, check_finite=check_finite)

            # orthogonalize
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
                res = spla.norm(X - np.eye(len(A_todo)), ord='fro', check_finite=check_finite)
                logger.info(f'Residual = {res}')
                if res <= tol*np.sqrt(len(A)):
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
