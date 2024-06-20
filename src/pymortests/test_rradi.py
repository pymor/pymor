from pymor.tools.io.matrices import load_matrix
from pymor.algorithms.lrradi import solve_ricc_lrcf
from pymor.basic import NumpyMatrixOperator, NumpyVectorSpace
import numpy as np

tA = load_matrix('./tA.mat', 'tA')
tA = NumpyMatrixOperator(tA)

E = None

B = load_matrix('./B.mat', 'B')
B = NumpyVectorSpace.from_numpy(B.T)

tC = load_matrix('./tC.mat', 'tC')
tC = NumpyVectorSpace.from_numpy(tC.T)

tQ = load_matrix('./tQ.mat', 'tQ')

R = np.eye(len(B))

Z_cf = solve_ricc_lrcf(tA, E, B, tC, R, tQ, None, True)
