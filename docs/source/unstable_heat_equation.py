import numpy as np
import scipy.sparse as sps

k = 50
n = 2 * k + 1
l = 50

E = sps.eye(n, format='lil')
E[0, 0] = E[-1, -1] = 0.5
E = E.tocsc()

d0 = n * [-2 * (n - 1)**2 + l]
d1 = (n - 1) * [(n - 1)**2]
A = sps.diags([d1, d0, d1], [-1, 0, 1], format='lil')
A[0, 0] = A[-1, -1] = -n * (n - 1) + l / 2
A = A.tocsc()

B = np.zeros((n, 1))
B[0, 0] = n - 1

C = np.zeros((1, n))
C[0, -1] = 1
