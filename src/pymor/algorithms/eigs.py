# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.logger import getLogger
from pymor.operators.constructions import IdentityOperator, InverseOperator
from pymor.operators.interface import Operator


def eigs(A, E=None, k=3, sigma=None, which='LM', b=None, l=None, maxiter=1000, tol=1e-13,
         imag_tol=1e-12, complex_pair_tol=1e-12, complex_evp=False, left_evp=False, seed=0):
    """Approximate a few eigenvalues of a linear |Operator|.

    Computes `k` eigenvalues `w` with corresponding eigenvectors `v` which solve
    the eigenvalue problem

    .. math::
        A v_i = w_i v_i

    or the generalized eigenvalue problem

    .. math::
        A v_i = w_i E v_i

    if `E` is not `None`.

    The implementation is based on Algorithm 4.2 in :cite:`RL95`.

    Parameters
    ----------
    A
        The linear |Operator| for which the eigenvalues are to be computed.
    E
        The linear |Operator| which defines the generalized eigenvalue problem.
    k
        The number of eigenvalues and eigenvectors which are to be computed.
    sigma
        If not `None` transforms the eigenvalue problem such that the k eigenvalues
        closest to sigma are computed.
    which
        A string specifying which `k` eigenvalues and eigenvectors to compute:

        - `'LM'`: select eigenvalues with largest magnitude
        - `'SM'`: select eigenvalues with smallest magnitude
        - `'LR'`: select eigenvalues with largest real part
        - `'SR'`: select eigenvalues with smallest real part
        - `'LI'`: select eigenvalues with largest imaginary part
        - `'SI'`: select eigenvalues with smallest imaginary part
    b
        Initial vector for Arnoldi iteration. Default is a random vector.
    l
        The size of the Arnoldi factorization. Default is `min(n - 1, max(2*k + 1, 20))`.
    maxiter
        The maximum number of iterations.
    tol
        The relative error tolerance for the Ritz estimates.
    imag_tol
        Relative imaginary parts below this tolerance are set to 0.
    complex_pair_tol
        Tolerance for detecting pairs of complex conjugate eigenvalues.
    complex_evp
        Wether to consider an eigenvalue problem with complex operators. When operators
        are real setting this argument to `False` will increase stability and performance.
    left_evp
        If set to `True` compute left eigenvectors else compute right eigenvectors.
    seed
        Random seed which is used for computing the initial vector for the Arnoldi
        iteration.

    Returns
    -------
    w
        A 1D |NumPy array| which contains the computed eigenvalues.
    v
        A |VectorArray| which contains the computed eigenvectors.
    """
    logger = getLogger('pymor.algorithms.eigs.eigs')

    assert isinstance(A, Operator) and A.linear
    assert not A.parametric
    assert A.source == A.range

    if E is None:
        E = IdentityOperator(A.source)
    else:
        assert isinstance(E, Operator) and E.linear
        assert not E.parametric
        assert E.source == E.range
        assert E.source == A.source

    if b is None:
        b = A.source.random(seed=seed)
    else:
        assert b in A.source

    n = A.source.dim
    l_min = 20

    if l is None:
        l = min(n - 1, max(2 * k + 1, l_min))

    assert k < n
    assert l > k

    if sigma is None:
        if left_evp:
            Aop = InverseOperator(E).H @ A.H
        else:
            Aop = InverseOperator(E) @ A
    else:
        if sigma.imag != 0:
            complex_evp = True
        else:
            sigma = sigma.real

        if left_evp:
            Aop = InverseOperator(A - sigma * E).H @ E.H
        else:
            Aop = InverseOperator(A - sigma * E) @ E

    V, H, f = _arnoldi(Aop, k, b, complex_evp)

    k0 = k
    i = 0

    while True:
        i += 1

        V, H, f = _extend_arnoldi(Aop, V, H, f, l - k)

        ew, ev = spla.eig(H)

        # truncate small imaginary parts
        ew.imag[np.abs(ew.imag) / np.abs(ew) < imag_tol] = 0

        if which == 'LM':
            idx = np.argsort(-np.abs(ew))
        elif which == 'SM':
            idx = np.argsort(np.abs(ew))
        elif which == 'LR':
            idx = np.argsort(-ew.real)
        elif which == 'SR':
            idx = np.argsort(ew.real)
        elif which == 'LI':
            idx = np.argsort(-np.abs(ew.imag))
        elif which == 'SI':
            idx = np.argsort(np.abs(ew.imag))

        k = k0
        ews = ew[idx]
        evs = ev[:, idx]

        rres = f.norm()[0] * np.abs(evs[l - 1]) / np.abs(ews)

        # increase k by one in order to keep complex conjugate pairs together
        if not complex_evp and ews[k - 1].imag != 0 and ews[k - 1].imag + ews[k].imag < complex_pair_tol:
            k += 1

        logger.info(f'Maximum of relative Ritz estimates at step {i}: {rres[:k].max():.5e}')

        if np.all(rres[:k] <= tol) or i >= maxiter:
            break

        # increase k in order to prevent stagnation
        k = min(l - 1, k + min(np.count_nonzero(rres[:k] <= tol), (l - k) // 2))

        # sort shifts for QR iteration based on their residual
        shifts = ews[k:l]
        srres = rres[k:l]
        idx = np.argsort(-srres)
        srres = srres[idx]
        shifts = shifts[idx]

        # don't use converged unwanted Ritz values as shifts
        shifts = shifts[srres != 0]
        k += np.count_nonzero(srres == 0)
        if not complex_evp and shifts[0].imag != 0 and shifts[0].imag + ews[1].imag >= complex_pair_tol:
            shifts = shifts[1:]
            k += 1

        H, Qs = _qr_iteration(H, shifts, complex_evp=complex_evp)

        V = V.lincomb(Qs.T)
        f = V[k] * H[k, k - 1] + f * Qs[l - 1, k - 1]
        V = V[:k]
        H = H[:k, :k]

    if sigma is not None:
        ews = 1 / ews + sigma

    return ews[:k0], V.lincomb(evs[:, :k0].T)


def _arnoldi(A, l, b, complex_evp):
    """Compute an Arnoldi factorization."""
    v = b * (1 / b.norm()[0])

    H = np.zeros((l, l), dtype=np.complex_ if complex_evp else np.float_)
    V = A.source.empty(reserve=l)

    V.append(v)

    for i in range(l):
        v = A.apply(v)
        V.append(v)

        _, R = gram_schmidt(V, return_R=True, atol=0, rtol=0, offset=len(V) - 1, copy=False)
        H[:i + 2, i] = R[:l, i + 1]
        v = V[-1]

    return V[:l], H, v * R[l, l]


def _extend_arnoldi(A, V, H, f, p):
    """Extend an existing Arnoldi factorization."""
    k = len(V)

    res = f.norm()[0]
    # the explicit "constant" mode is needed for numpy 1.16
    # mode only gained a default value with numpy 1.17
    H = np.pad(H, ((0, p), (0, p)), mode='constant')
    H[k, k - 1] = res
    v = f * (1 / res)
    V = V.copy()
    V.append(v)

    for i in range(k, k + p):
        v = A.apply(v)
        V.append(v)
        _, R = gram_schmidt(V, return_R=True, atol=0, rtol=0, offset=len(V) - 1, copy=False)
        H[:i + 2, i] = R[:k + p, i + 1]

        v = V[-1]

    return V[:k + p], H, v * R[k + p, k + p]


def _qr_iteration(H, shifts, complex_evp=False):
    """Perform the QR iteration."""
    Qs = np.eye(len(H))

    i = 0
    while i < len(shifts) - 1:
        s = shifts[i]
        if not complex_evp and shifts[i].imag != 0:
            Q, _ = np.linalg.qr(H @ H - 2 * s.real * H + np.abs(s)**2 * np.eye(len(H)))
            i += 2
        else:
            Q, _ = np.linalg.qr(H - s * np.eye(len(H)))
            i += 1
        Qs = Qs @ Q
        H = Q.T @ H @ Q

    return H, Qs
