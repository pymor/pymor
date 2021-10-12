import numpy as np
from scipy import linalg

from pymor.algorithms.svd_va import method_of_snapshots, qr_svd
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.vectorarrays.interface import VectorArray


@defaults('svd_method')
def dmd(X, Y=None, svd_rank=None, dt=1, modes='exact', svd_method='qr_svd', return_A_tilde=False):
    """Dynamic Mode Decomposition.

    See Algorithm 1 and Algorithm 2 in :cite:`TRLBK14`.

    Parameters
    ----------
    X
        The |VectorArray| for which the DMD algorithm is performed.
        If `Y` is given, `X` and `Y` are the left resp. right snapshot series.
    Y
        The |VectorArray| of the right snapshot series.
    svd_rank
        Number of DMD Modes to be computed, if None `svd_rank = len(X)`.
    dt
        Factor specifying the time difference between the observations, default `dt = 1`.
    modes
        - 'standard': uses the standard definition to compute the dynamic modes
            `Wk = U * evecs`, where `U` are the left singular vectors of `X`.
        - 'exact' : computes the exact dynamic modes, `Wk = (1/evals) * Y * V * Sigma_inv * evecs`.
    svd_method
        Which SVD method from :mod:`~pymor.algorithms.svd_va` to use
        (`'method_of_snapshots'` or `'qr_svd'`).
    return_A_tilde
        If `True` the matrices `A_tilde` and `U`, which are necessary to reconstruct the operator
        `A` with `AX=Y` are returned s.t. `A_approx = U.conj().T @ A_tilde @ U`.

    Returns
    -------
    Wk
        |VectorArray| containing the dynamic modes.
    omega
        Time scaled eigenvalues: `ln(l)/dt`.
    """
    assert modes in ('exact', 'standard')
    assert isinstance(X, VectorArray)
    assert svd_rank is None or svd_rank <= X.dim
    assert isinstance(Y, VectorArray) or Y is None
    assert svd_method in ('qr_svd', 'method_of_snapshots')

    logger = getLogger('pymor.algorithms.dmd.dmd')

    # X = z_0, ..., z_{m-1}; Y = z_1, ..., z_m
    if Y is None:
        assert svd_rank is None or svd_rank < len(X)

        Y = X[1:]
        X = X[:-1]
    else:
        assert svd_rank is None or svd_rank <= len(X)
        assert len(X) == len(Y)

    assert len(X) >= X.dim

    rank = len(X) if svd_rank is None else svd_rank
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
    elif modes == 'exact':
        Wk = Y.lincomb((V @ evecs / s).T)
        evals_inv = np.reciprocal(evals)
        Wk = Wk * evals_inv
    else:
        assert False

    if return_A_tilde:
        return Wk, omega, A_tilde, U

    return Wk, omega
