import numpy as np
import scipy.linalg as spla

from pymor.core.defaults import defaults
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.algorithms.svd_va import method_of_snapshots, qr_svd
from pymor.algorithms.gram_schmidt import gram_schmidt


@defaults('svd_method')
def dmd(A=None, XL=None, XR=None, target_rank=None, dt=1, modes='exact', order=True, svd_method='qr_svd'):
    """Dynamic Mode Decomposition.

    See Algorithm 1 and Algorithm 2 in [TRLBK14]_.

    Parameters
    ----------
    A  :  optional
        The |VectorArray| for which the DMD Modes are to be computed.
        If A is used, XL and XR should not be used.

    XL  :  optional
        The |VectorArray| of the left Snapshot series A[:-1].
    XR  :  optional
        The |VectorArray| of the right Snapshot series A[1:].
        If XL and XR are given, A should not be given.

    target_rank : int
        Number of DMD Modes to be computed. If None target_rank = len(A).

    dt : scalar, optional (default: 1)
        Factor specifying the time difference between the observations.
        Used if the input data is a timeseries in continuous time.

    modes : str `{'standard', 'exact', 'exact_scaled'}`
        - 'standard' : uses the standard definition to compute the dynamic modes, `Wk = U * EVECS`.
        - 'exact' : computes the exact dynamic modes, `Wk = XR * V * Sigma_inv * EVECS`.
        - 'exact_scaled' : computes the exact dynamic modes,
                           `Wk = (1/EVALS) * XR * V * Sigma_inv * EVECS`.

    order : bool `{True, False}`
        True: return modes sorted.

    svd_method
        Which SVD method from :mod:`~pymor.algorithms.svd_va` to use
        (`'method_of_snapshots'` or `'qr_svd'`).


    Returns
    -------
    Wk : VectorArray
        |VectorArray| containing the dynamic modes.

    omega : array_like
        Time scaled eigenvalues: `ln(l)/dt`.
    """

    assert modes in ('exact', 'standard', 'exact_scaled')
    assert A is None or (XL is None and XR is None)
    assert not(A is None and (XL and XR is None))
    assert isinstance(A, VectorArray) or A is None
    assert isinstance(XL, VectorArray) or XL is None
    assert isinstance(XR, VectorArray) or XR is None
    assert XL is None and XR is None or len(XL) == len(XR)
    assert target_rank is None or target_rank <= len(A)
    assert order in (True, False)
    assert svd_method in ('qr_svd', 'method_of_snapshots')

    # XL = x0, ..., xm-1  ; XR = x1, ..., xm
    if A is not None:
        XL = A[:-1]
        XR = A[1:]

    rank = len(XL) if target_rank is None else target_rank
    svd = qr_svd if svd_method == 'qr_svd' else method_of_snapshots

    print('SVD of XL...')
    U, SVALS, Vh = svd(XL, product=None, modes=rank)

    if not len(U) == rank:
        rank = len(U)
        print('Cutting  to Dimension ', rank, ' - Not enought relevant Singularvectors.')

    # Invert the Singular Values
    SVALS_inv = np.reciprocal(SVALS)
    Sigma_inv = np.diag(SVALS_inv)
    # Cut Vh to relevant modes
    Vh = Vh[:, :rank]
    V = Vh.conj().T

    # Solve the least Sqaures Problem
    # real: A_tilde = U.T * XR * Vh.T * Sigma_inv
    # complex: A_tilde = U.H * XR * Vh.H * Sigma_inv
    # A = XR[:k] @ V @ Sigma_inv.inner(Uh)
    A_tilde = U.inner(XR[:rank]) @ V @ Sigma_inv

    print('Calculating eigenvalue dec. ...')
    EVALS, EVECS = spla.eig(A_tilde, b=None, left=True, right=False)
    # omega = np.log(EVALS) / dt
    omega = EVALS / dt

    if order:
        # return ordered result
        sort_idx = np.argsort(np.abs(omega))
        EVECS = EVECS[:, sort_idx]
        EVALS = EVALS[sort_idx]
        omega = omega[sort_idx]

    print('Reconstructing Eigenvectors...')
    if modes == 'standard':
        Wk = U.lincomb(EVECS)
    elif modes == 'exact' or 'exact_scaled':
        Wk = XR[:rank].lincomb(V @ Sigma_inv @ EVECS)
        if modes == 'exact_scaled':
            EVALS_inv = np.reciprocal(EVALS)
            Wk = Wk*EVALS_inv

    return Wk, omega


def rand_QB(A, target_rank=None, distribution='normal', oversampling=0, powerIterations=0):
    """
    randomisierte QB-Zerlegung

    See Algorithm 3.1 in [EMKB19]_.

    Parameters
    ----------
    A  :
        The |VectorArray| for which the randomized QB Decomposition is to be computed.

    target_rank  :  int
        The desired rank for the decomposition. If None rank = len(A).

    distribution : str
        Distribution used for the random projectionmatrix Omega. (`'normal'` or `'uniform'`)

    oversampling : int
        Oversamplingparameter. Number of extra columns of the projectionmatrix.

    powerIterations : int
        Number of power Iterations.


    Returns
    -------
    Q :
        |VectorArray| containig an approximate optimal Basis for the Image of the Inputmatrix A.
        len(Q) = target_rank
    B :
        Numpy Array. Projection of the Input Matrix into the lower dimensional subspace.
    """
    # TODO: weitere Assertions
    assert isinstance(A, VectorArray)
    assert target_rank is None or target_rank <= len(A)
    assert distribution in ('normal', 'uniform')

    if A.dim == 0 or len(A) == 0:
        return A.space.zeros(), np.zeros((target_rank, len(A)))

    rank = len(A) if target_rank is None else target_rank + oversampling

    Omega = np.random.normal(0, 1, (rank, len(A))) if distribution == 'normal' else np.random.rand(rank, len(A))

    Y = A.lincomb(Omega)

    # Power Iterations
    if(powerIterations > 0):
        for i in range(powerIterations):
            Q = gram_schmidt(Y)
            Z, _ = spla.qr(A.inner(Q))
            Y = A.lincomb(Z)

    Q = gram_schmidt(Y)
    B = Q.inner(A)

    if not len(Q) == rank:
        print('Cutting B to Dimension (', B.shape[0], ', ', len(Q), ').',
              'Not enought orthonormal Vectors in random Projection.')
        B = B[:, :len(Q)]

    return Q, B


@defaults('svd_method', 'distribution')
def rand_dmd(A, target_rank=None, dt=1, modes='exact', svd_method='qr_svd', distribution='normal',
             oversampling=0, powerIterations=0, order=True):
    """
    Ranzomized Dynamic Mode Decomposition

    See Algorithm 4.1 in [EMKB19]_.


    Parameters
    ----------
    A  :
        The |VectorArray| for which the DMD Modes are to be computed.

    target_rank : int
        Number of DMD Modes to be computed. If None target_rank = len(A).

    dt : scalar, optional (default: 1)
        Factor specifying the time difference between the observations.
        Used if the input data is a timeseries in continuous time.

    modes : str `{'standard', 'exact', 'exact_scaled'}`
        - 'standard' : uses the standard definition to compute the dynamic modes, `Wk = U * EVECS`.
        - 'exact' : computes the exact dynamic modes, `Wk = XR * V * Sigma_inv * EVECS`.
        - 'exact_scaled' : computes the exact dynamic modes,
                           `Wk = (1/EVALS) * XR * V * Sigma_inv * EVECS`.

    svd_method : str
        Which SVD method from :mod:`~pymor.algorithms.svd_va` to use
        (`'method_of_snapshots'` or `'qr_svd'`).

    distribution : str
        Distribution used for the randomized QB-Decomposition. (`'normal'` or `'uniform'`)

    oversampling : int
        Oversamplingparameter. Number of extra columns of the projectionmatrix.

    powerIterations : int
        Number of power Iterations.

    order : bool `{True, False}`
        True: return modes sorted.


    Returns
    -------
    Wk :
        |VectorArray| containing the dynamic modes.

    omega :
        Numpy Array containig time scaled eigenvalues: `ln(l)/dt`.

    """

    assert isinstance(A, VectorArray)
    assert target_rank is None or target_rank <= len(A)
    assert distribution in ('normal', 'uniform')
    assert modes in ('exact', 'standard', 'exact_scaled')
    assert order in (True, False)
    assert svd_method in ('qr_svd', 'method_of_snapshots')
    assert oversampling >= 0
    assert powerIterations >= 0

    rank = len(A) if target_rank is None else target_rank

    Q, B = rand_QB(A, target_rank=target_rank, distribution=distribution, oversampling=oversampling,
                   powerIterations=powerIterations)

    # transform B to VectorArray
    B = NumpyVectorSpace.from_numpy(B)

    Wk, omega = dmd(A=B, target_rank=None, dt=1, modes=modes, svd_method=svd_method, order=order)

    # transform Wk to numpy
    Wk = Wk.to_numpy()

    rank = min(rank, len(B))
    Wk = Q[:rank].lincomb(Wk[:, :rank])

    return Wk, omega
