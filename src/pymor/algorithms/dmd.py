import numpy as np
import scipy.linalg as spla

from pymor.algorithms.svd_va import method_of_snapshots, qr_svd
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.constructions import LowRankOperator
from pymor.vectorarrays.interface import VectorArray


@defaults('svd_method')
def dmd(X, Y=None, modes=None, atol=None, rtol=None, cont_time_dt=None, type='exact', order='magnitude',
        svd_method='method_of_snapshots', return_A_approx=False, return_A_tilde=False):
    """Dynamic Mode Decomposition.

    See Algorithm 1 and Algorithm 2 in :cite:`TRLBK14`.

    Parameters
    ----------
    X
        The |VectorArray| for which the DMD algorithm is performed.
        If `Y` is given, `X` and `Y` are the left resp. right snapshot series.
    Y
        The |VectorArray| of the right snapshot series.
    modes
        Maximum number of singular vectors of `X` to take into account.
    atol
        Absolute truncation tolerance for singular values of `X`.
    rtol
        Relative truncation tolerance for singular values of `X`.
    cont_time_dt
        If not `None`, return continuous-time DMD eigenvalues with scaling
        log(lambda) / dt.
    type
        - 'standard': uses the standard definition to compute the dynamic modes
            `Wk = U * evecs`, where `U` are the left singular vectors of `X`.
        - 'exact' : computes the exact dynamic modes, `Wk = (1/evals) * Y * V * Sigma_inv * evecs`.
    order
        Sort DMD eigenvalues either by `'magnitude'` or `'phase'`.
    svd_method
        Which SVD method from :mod:`~pymor.algorithms.svd_va` to use
        (`'method_of_snapshots'` or `'qr_svd'`).
    return_A_approx
        If `True`, the approximation of the operator `A` with `AX=Y` is returned as
        a :class:`~pymor.operators.constructions.LowRankOperator`.
    return_A_tilde
        If `True` the low-rank dynamics are returned.

    Returns
    -------
    Wk
        |VectorArray| containing the dynamic modes. The number of computed modes
        is given by the SVD truncation rank determined by the `modes`, `atol` and
        `rtol` arguments.
    evals
        Discrete or continuous time DMD eigenvalues.
    A_approx
        :class:`~pymor.operators.constructions.LowRankOperator` containing the approximation
        of the operator `A` with `AX=Y`. Only provided if `return_A_approx` is `True`.
    A_tilde
         Low-rank dynamics. Only provided if `return_A_tilde` is `True`.
    """
    assert isinstance(X, VectorArray)
    assert isinstance(Y, VectorArray) or Y is None
    assert Y is None or len(X) == len(Y)
    assert type in ('exact', 'standard')
    assert order in ('magnitude', 'phase')
    assert svd_method in ('qr_svd', 'method_of_snapshots')

    logger = getLogger('pymor.algorithms.dmd.dmd')

    if Y is None:
        Y = X[1:]
        X = X[:-1]

    svd = qr_svd if svd_method == 'qr_svd' else method_of_snapshots

    logger.info('SVD of X ...')
    U, s, Vh = svd(X, modes=modes, atol=atol, rtol=rtol)

    V = Vh.conj().T

    # compute low-rank dynamics
    A_tilde = U.inner(Y) @ V / s

    logger.info('Calculating DMD eigenvalues ...')
    evals, evecs = spla.eig(A_tilde)

    # ordering
    if order == 'magnitude':
        sort_idx = np.argsort(np.abs(evals))[::-1]
    elif order == 'phase':
        sort_idx = np.argsort(np.abs(np.angle(evals)))
    else:
        assert False
    evecs = evecs[:, sort_idx]
    evals = evals[sort_idx]

    logger.info('Computing DMD modes ...')
    if type == 'standard':
        Wk = U.lincomb(evecs.T)
    elif type == 'exact':
        Wk = Y.lincomb((((V / s) @ evecs) / evals).T)
    else:
        assert False

    retval = [Wk]

    if cont_time_dt is not None:
        retval.append(np.log(evals) / cont_time_dt)
    else:
        retval.append(evals)

    if return_A_approx:
        A_approx = LowRankOperator(Y.lincomb(V.T), np.diag(s), U, inverted=True)
        retval.append(A_approx)

    if return_A_tilde:
        retval.append(A_tilde)

    return tuple(retval)
