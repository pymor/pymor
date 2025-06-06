import numpy as np

from pymor.algorithms.pod import pod
from pymor.algorithms.gram_schmidt import gram_schmidt_biorth
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray
from pymor.vectorarrays.block import BlockVectorSpace


def biorthogonalize(Vtilde_r, Wtilde_r, method):
    max_modes = min(len(Vtilde_r), len(Wtilde_r))
    if max_modes % 2 != 0:
        max_modes -= 1
    Vtilde_r = Vtilde_r[:max_modes]
    Wtilde_r = Wtilde_r[:max_modes]
    if method == 1:
        V_r, W_r = biorth_my_method(Vtilde_r, Wtilde_r, X.dim // 2)
    elif method == 2:
        V_r, W_r = gram_schmidt_biorth(Vtilde_r, Wtilde_r)
    return V_r, W_r


def POD_PH(X, F, modes, method):
    """given snapshot matrix X and force snapshot matrix F
    get a POD basis of X and F, Vtilde_r and Wtilde_r
    change the bases so that they are biorthogonal, V_r and W_r
    return the bases V_r and W_r
    
    based on algorithm 1 in Chaturantabut/Beattie/Gugercing
    """
    
    Vtilde_r, svals = pod(X, modes=modes)
    Wtilde_r, svals = pod(F, modes=modes)
    
    V_r, W_r = biorthogonalize(Vtilde_r, Wtilde_r, method)
    return V_r, W_r

def POD_new(X, modes, method):
    half_dim = X.dim//2
    X1, X2 = X.blocks[0], X.blocks[1]
    print("dimensions of X1", len(X1), X1.dim)

    Vtilde_r, _ = pod(X1, modes=modes)
    Wtilde_r, _ = pod(X2, modes=modes)
    
    V_r, W_r = biorthogonalize(Vtilde_r=Vtilde_r, Wtilde_r=Wtilde_r, method=method)
    return V_r, W_r


def biorth_my_method(Vtilde_r, Wtilde_r, half_dim):
    M = Wtilde_r.inner(Vtilde_r)
    eigenvalues, eigenvectors = np.linalg.eig(M)
    M_inverse_sqrt = (eigenvectors @ np.diag((np.sqrt(eigenvalues))**(-1)) @ np.linalg.inv(eigenvectors))

    numpy_V_r1 = Vtilde_r.to_numpy() @ M_inverse_sqrt
    numpy_W_r1 = Wtilde_r.to_numpy() @ M_inverse_sqrt.transpose()

    space = BlockVectorSpace([NumpyVectorSpace(half_dim), NumpyVectorSpace(half_dim)])
    V_r = space.from_numpy(numpy_V_r1)
    W_r = space.from_numpy(numpy_W_r1)

    return V_r, W_r

def check_POD(X, modes):
    """given snapshot matrix X and force snapshot matrix F
    get a POD basis of X and F, Vtilde_r and Wtilde_r
    change the bases so that they are biorthogonal, V_r and W_r
    return the bases V_r and W_r
    
    based on algorithm 1 in Chaturantabut/Beattie/Gugercing
    """
    
    Vtilde_r, svals = pod(X, modes=modes)

    return Vtilde_r


