import numpy as np
import scipy.linalg as spla
import scipy.sparse as spsp

from pymor.basic import *
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.algorithms.svd_va import method_of_snapshots, qr_svd

def conjugate_transpose(A):
    """Performs conjugate transpose of A"""   
    if type(A) == np.ndarray:
        return A.conj().T

    if type(A) == VectorArray or NumpyVectorArray:
        A_numpy = A.conj().to_numpy().T
        return NumpyVectorSpace.from_numpy(A_numpy)
    
    return A.T

def dmd(A, rank=None, dt=1, modes='exact', order=True):
    """Dynamic Mode Decomposition.

    Dynamic Mode Decomposition (DMD) is a data processing algorithm which
    allows to decompose a matrix `A` in space and time. The matrix `A` is
    decomposed as `A = Wk * B * V`, where the columns of `Wk` contain the dynamic modes.
    The modes are ordered corresponding to the amplitudes in the diagonal
    matrix `B`. `V` is a Vandermonde matrix describing the temporal evolution.


    Parameters
    ----------
    A : VectorArray, shape `(m, n)`.
        Input array.

    rank : int
        If `rank < (m-1)` low-rank Dynamic Mode Decomposition is computed.

    dt : scalar, optional (default: 1)
        Factor specifying the time difference between the observations.

    modes : str `{'standard', 'exact', 'exact_scaled'}`
        - 'standard' : uses the standard definition to compute the dynamic modes, `Wk = U * EVECS`.
        - 'exact' : computes the exact dynamic modes, `Wk = XR * V * Sigma_inv * EVECS`.
        - 'exact_scaled' : computes the exact dynamic modes, `Wk = (1/EVALS) * XR * V * Sigma_inv * EVECS`.

    order :  bool `{True, False}`
        True: return modes sorted.


    Returns
    -------
    Wk : VectorArray
        |VectorArray| containing the dynamic modes of shape `(m, n-1)`  or `(m, k)`.

    omega : array_like
        Time scaled eigenvalues: `ln(l)/dt`.


    References
    ----------
    J. H. Tu, et al.
    "On Dynamic Mode Decomposition: Theory and Applications" (2013).
    (available at `arXiv <http://arxiv.org/abs/1312.0041>`_).

    N. B. Erichson, L. Mathelin, J. N. Kutz, S. L. Brunton.
    "Randomized Dynamic Mode Decomposition" (2019).
    (available at `SIAM https://epubs.siam.org/doi/pdf/10.1137/18M1215013`_).
    """

    assert modes in ('exact', 'standard', 'exact_scaled')
    assert isinstance(A, VectorArray)
    assert rank is None or rank < len(A)-1
    assert order in (True, False)

    # XL = x0, ..., xm-1  ; XR = x1, ..., xm
    XL = A[:-1]
    XR = A[1:]
    
    if rank is None: 
        k = len(XL) 
    else:
        k = rank

    print('SVD of XL...')
    U, SVALS, Vh = qr_svd(XL, product=None, modes=k)
    #Invert the Singular Values
    SVALS_inv = np.reciprocal(SVALS)
    Sigma_inv = spsp.diags(SVALS_inv).toarray()
    #Cut Vh to k modes
    Vh = Vh[:, :k]
    V = conjugate_transpose(Vh)

    #Solve the least Sqaures Problem 
    # real: A_tilde = U.T * XR * Vh.T * Sigma_inv
    # complex: A_tilde = U.H * XR * Vh.H * Sigma_inv
    A_tilde = U.conj().inner(XR[:k]).dot(V).dot(Sigma_inv)

    print('Calculating eigenvalue dec. ...')      
    EVALS, EVECS = spla.eig(A_tilde, b=None, left=True, right=False)
    omega = np.log(EVALS) / dt

    if order:
        # return ordered result
        sort_idx = np.argsort(np.abs(omega))
        EVECS = EVECS[:, sort_idx]
        EVALS = EVALS[sort_idx]
        omega = omega[sort_idx]

    print('Reconstructing Eigenvectors...')
    if modes == 'standard':
        Wk = U.lincomb(EVECS)
    if modes == 'exact' or 'exact_scaled':
        W_k = V.dot(Sigma_inv).dot(EVECS)
        Wk = XR[:k].lincomb(W_k)
        if modes == 'exact_scaled':
            EVALS_inv = np.reciprocal(EVALS)
            Wk = Wk*EVALS_inv

    return Wk, omega