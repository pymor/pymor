import numpy as np

from pymor.algorithms.pod import pod
from pymor.algorithms.gram_schmidt import gram_schmidt_biorth
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray
from pymor.vectorarrays.block import BlockVectorSpace


def POD_PH(X, F, modes):
    """given snapshot matrix X and force snapshot matrix F
    get a POD basis of X and F, Vtilde_r and Wtilde_r
    change the bases so that they are biorthogonal, V_r and W_r
    return the bases V_r and W_r
    
    based on algorithm 1 in Chaturantabut/Beattie/Gugercing
    """
    
    Vtilde_r, svals = pod(X, modes=modes)
    Wtilde_r, svals = pod(F, modes=modes)
    max_modes = min(len(Vtilde_r), len(Wtilde_r))
    Vtilde_r = Vtilde_r[:max_modes]
    Wtilde_r = Wtilde_r[:max_modes]

    M = Wtilde_r.inner(Vtilde_r)
    eigenvalues, eigenvectors = np.linalg.eig(M)
    M_inverse_sqrt = (eigenvectors.transpose() @ np.diag(np.sqrt(eigenvalues**(-1))) @ eigenvectors)
    V_r1 = Vtilde_r.lincomb(M_inverse_sqrt)
    W_r1 = Wtilde_r.lincomb(M_inverse_sqrt.transpose())
    # numpy_Vtilde_r = Vtilde_r.to_numpy().transpose()
    # numpy_Wtilde_r = Wtilde_r.to_numpy().transpose()
    # numpy_V_r = (numpy_Vtilde_r @ M_inverse_sqrt).transpose()
    # numpy_W_r = (numpy_Wtilde_r @ M_inverse_sqrt.transpose()).transpose()
    # half_dim = X.dim//2
    # space = BlockVectorSpace([NumpyVectorSpace(half_dim), NumpyVectorSpace(half_dim)])
    # V_r1 = space.from_numpy(np.array([numpy_V_r[:half_dim, :], numpy_V_r[half_dim:, :]]))
    # W_r1 = space.from_numpy(np.array([numpy_W_r[:half_dim, :], numpy_W_r[half_dim:, :]]))
    V_r2, W_r2 = gram_schmidt_biorth(Vtilde_r, Wtilde_r)
    # print("checking biorthogonality 1", np.linalg.norm((np.identity(len(V_r1)) - (W_r1.to_numpy() @ V_r1.to_numpy().transpose()))))
    # print("checking biorthogonality 2", np.linalg.norm((np.identity(len(V_r2)) - (W_r2.to_numpy() @ V_r2.to_numpy().transpose()))))
    return V_r2, W_r2

def check_POD(X, modes):
    """given snapshot matrix X and force snapshot matrix F
    get a POD basis of X and F, Vtilde_r and Wtilde_r
    change the bases so that they are biorthogonal, V_r and W_r
    return the bases V_r and W_r
    
    based on algorithm 1 in Chaturantabut/Beattie/Gugercing
    """
    
    Vtilde_r, svals = pod(X, modes=modes)

    return Vtilde_r


