import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import block_gram_schmidt
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

Q, R = block_gram_schmidt(A, return_R=True, atol=0, rtol=0)
print(len(Q))
print(spla.norm(Q.gramian() - np.eye(m)))
print(spla.norm((A - Q.lincomb(R.T)).norm()))
