import numpy as np
import scipy.linalg as spla
from time import perf_counter
from pymor.algorithms.gram_schmidt import gram_schmidt

from pymor.algorithms.qr import qr
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.core.logger import set_log_levels

set_log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt': 'WARNING'})


def random_svd_va(m, n, cond):
    rng = np.random.default_rng(0)
    U = rng.normal(size=(n, m))
    U = spla.qr(U, mode='economic')[0]
    S = np.diag(np.geomspace(1/cond, 1, m))
    V = rng.normal(size=(m, m))
    V = spla.qr(V)[0]
    return NumpyVectorSpace.from_numpy(V @ S @ U.T)


if __name__ == '__main__':
    A = random_svd_va(100, 1000, 1e20)

    return_R = True

    cholesky_kwargs = {'tol': 1e-14, 'maxiter': 10}
    gram_schmidt_kwargs = {'rtol': 0}

    for solver in ('cholesky_qr', 'gram_schmidt'):
        print(f'Solver: \"{solver}\"')
        tic = perf_counter()
        kwargs = cholesky_kwargs if solver == 'cholesky_qr' else gram_schmidt_kwargs
        Q, R = qr(A, solver=solver, return_R=return_R, **kwargs)
        toc = perf_counter()
        print(f'len(Q): {len(Q)}')
        print(f'Time:\t\t\t{toc-tic} s')
        print(f'Orthogonality:\t{spla.norm(Q.gramian() - np.eye(len(A)))}')
        print(f'Reconstruction:\t{spla.norm((A - Q.lincomb(R.T)).norm())}')
