import numpy as np
from scipy import linalg

from pymor.algorithms.svd_va import method_of_snapshots, qr_svd
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.constructions import LowRankOperator
from pymor.vectorarrays.interface import VectorArray


@defaults('svd_method')
def dmd(X, Y=None, target_rank=None, dt=1, modes='exact', svd_method='qr_svd', return_A_approx=False):
    """Dynamic Mode Decomposition.

    See Algorithm 1 and Algorithm 2 in :cite:`TRLBK14`.

    Parameters
    ----------
    X  :  |VectorArray|
        The |VectorArray| for which the DMD Modes are to be computed.
        If Y is given, X and Y are the left resp. right Snapshot series.
    Y  :  optional/|VectorArray|
        The |VectorArray| of the right Snapshot series.
    target_rank : int/optional
        Number of DMD Modes to be computed. If None target_rank = len(A).
    dt : scalar, optional (default: 1)
        Factor specifying the time difference between the observations.
        Used if the input data is a timeseries in continuous time.
    modes : str `{'standard', 'exact', 'exact_scaled'}`
        - 'standard' : uses the standard definition to compute the dynamic modes,
                    where U are the left singular vectors `Wk = U * evecs`.
        - 'exact' : computes the exact dynamic modes, `Wk = (1/evals) * Y * V * Sigma_inv * evecs`.
    svd_method
        Which SVD method from :mod:`~pymor.algorithms.svd_va` to use
        (`'method_of_snapshots'` or `'qr_svd'`).

    Returns
    -------
    Wk : |VectorArray|
        |VectorArray| containing the dynamic modes.
    omega : array_like
        Time scaled eigenvalues: `ln(l)/dt`.
    """
    assert modes in ('exact', 'standard')
    assert isinstance(X, VectorArray)
    assert target_rank is None or target_rank <= X.dim
    assert isinstance(Y, VectorArray) or Y is None
    assert svd_method in ('qr_svd', 'method_of_snapshots')

    logger = getLogger('pymor.algorithms.dmd.dmd')

    # X = z_0, ..., z_{m-1}; Y = z_1, ..., z_m
    if Y is None:
        assert target_rank is None or target_rank < len(X)

        Y = X[1:]
        X = X[:-1]
    else:
        assert target_rank is None or target_rank <= len(X)
        assert len(X) == len(Y)

    rank = len(X) if target_rank is None else target_rank
    svd = qr_svd if svd_method == 'qr_svd' else method_of_snapshots

    logger.info('SVD of X...')
    U, s, Vh = svd(X, product=None, modes=rank)

    V = Vh.conj().T

    # Solve the least Squares Problem
    A_tilde = U.inner(Y) @ V / s

    logger.info('Calculating eigenvalue dec. ...')
    evals, evecs = linalg.eig(A_tilde)

    omega = evals / dt

    # ordering
    sort_idx = np.argsort(np.abs(omega))[::-1]
    evecs = evecs[:, sort_idx]
    evals = evals[sort_idx]
    omega = omega[sort_idx]

    logger.info('Reconstructing Eigenvectors...')
    if modes == 'standard':
        Wk = U.lincomb(evecs.T)
    elif modes == 'exact' or 'exact_scaled':
        Wk = Y.lincomb((V @ evecs / s).T)
        evals_inv = np.reciprocal(evals)
        Wk = Wk * evals_inv
    else:
        assert False

    if return_A_approx:
        A_approx = LowRankOperator(U, A_tilde, U)
        return Wk, omega, A_approx

    return Wk, omega
