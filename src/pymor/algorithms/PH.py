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
    if max_modes % 2 != 0:
        max_modes -= 1
    Vtilde_r = Vtilde_r[:max_modes]
    Wtilde_r = Wtilde_r[:max_modes]

    M = Wtilde_r.inner(Vtilde_r)
    eigenvalues, eigenvectors = np.linalg.eig(M)
    M_decomp = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
    print("checking M_decomp", np.linalg.norm(M - M_decomp))
    M_inverse_sqrt = (eigenvectors @ np.diag((np.sqrt(eigenvalues))**(-1)) @ np.linalg.inv(eigenvectors))
    M_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ np.linalg.inv(eigenvectors)
    print("checking M_sqrt", np.linalg.norm(M - M_sqrt @ M_sqrt))
    print("cheking M_inverse_sqrt", np.linalg.norm(np.linalg.inv(M) - M_inverse_sqrt @ M_inverse_sqrt))
    numpy_V_r1 = Vtilde_r.to_numpy() @ M_inverse_sqrt
    numpy_W_r1 = Wtilde_r.to_numpy() @ M_inverse_sqrt.transpose()
    numpy_Vtilde_r = Vtilde_r.to_numpy()
    numpy_Wtilde_r = Wtilde_r.to_numpy()
    # print(V_r1.shape, W_r1.shape)
    # print("checking biorthogonality of numpy", np.linalg.norm(np.identity(numpy_Vtilde_r.shape[1]) - W_r1.transpose() @ V_r1))
    # numpy_V_r = (numpy_Vtilde_r @ M_inverse_sqrt).transpose()
    # numpy_W_r = (numpy_Wtilde_r @ M_inverse_sqrt.transpose()).transpose()
    half_dim_V_r = numpy_V_r1.shape[0] // 2
    half_dim_X = X.dim // 2
    space = BlockVectorSpace([NumpyVectorSpace(half_dim_X), NumpyVectorSpace(half_dim_X)])
    V_r1 = space.from_numpy(numpy_V_r1)
    W_r1 = space.from_numpy(numpy_W_r1)
    print(V_r1)
    V_r2, W_r2 = gram_schmidt_biorth(Vtilde_r, Wtilde_r)
    # print("checking biorthogonality 1", np.linalg.norm((np.identity(len(V_r1)) - (W_r1.inner(V_r1)))))
    print("checking biorthogonality 2", np.linalg.norm((np.identity(len(V_r2)) - (W_r2.inner(V_r2)))))
    print("checking difference between V_r and W_r", np.sqrt((W_r2 - Wtilde_r).norm2().sum()))
    print("checking difference between V_r and W_r", np.sqrt((W_r1 - Wtilde_r).norm2().sum()))
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


