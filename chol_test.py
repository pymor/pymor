import numpy as np
import scipy.linalg as spla

from pymor.algorithms.cholesky_qr import cholesky_qr
from pymor.vectorarrays.numpy import NumpyVectorSpace

n = 1000
m = 100
cond = 1e-20

rng = np.random.default_rng(0)
U = rng.normal(size=(n, m))
U = spla.qr(U, mode='economic')[0]
S = np.diag(np.geomspace(cond, 1, m))
V = rng.normal(size=(m, m))
V = spla.qr(V)[0]
A = U @ S @ V.T
A = NumpyVectorSpace.from_numpy(A.T)

Q, R = cholesky_qr(A, return_R=True, tol=1e-14, maxiter=10)
print(len(Q))
print(spla.norm(Q.gramian() - np.eye(m)))
print(spla.norm((A - Q.lincomb(R.T)).norm()))
