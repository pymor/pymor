# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.operators.constructions import IdentityOperator


def eigs(A, E=None, k=3, which='LM', b=None, l=None, maxiter=1000, tol=1e-13):
    """Approximate a few eigenvalues of an |Operator|.

    Computes `k` eigenvalues `w[i]` with corresponding eigenvectors `v[i]` which solve
    the eigenvalue problem

    .. math::
        A v[i] = w[i] v[i]

    or the generalized eigenvalue problem

    .. math::
        A v[i] = w[i] E v[i]

    if `E` is not `None`.

    The implementation is based on Algorithm 4.2 in [RL95]_.

    Parameters
    ----------
    A
        The real |Operator| for which the eigenvalues are to be computed.
    E
        The |Operator| which defines the generalized eigenvalue problem.
    k
        The number of eigenvalues and eigenvectors which are to be computed.
    which
        A string specifying which `k` eigenvalues and eigenvectors to compute:
            - `'LM'`: select eigenvalues with largest |v[i]|
            - `'SM'`: select eigenvalues with smallest |v[i]|
            - `'LR'`: select eigenvalues with largest Re(v[i])
            - `'SR'`: select eigenvalues with smallest Re(v[i])
            - `'LI'`: select eigenvalues with largest Im(v[i])
            - `'SI'`: select eigenvalues with smallest Im(v[i])
    b
        Initial vector for Arnoldi iteration. Default is a random vector.
    l
        The size of the Arnoldi factorization. Default is `min(n - 1, max(2*k + 1, 20))`.
    maxiter
        The maximum number of iterations.
    tol
        The relative error tolerance for the ritz estimates.

    Returns
    -------
    w
        A |NumPy array| which contains the computed eigenvalues.
    v
        A |VectorArray| which contains the computed eigenvectors.
    """

    n = A.source.dim

    if l is None:
        l = np.min((n - 1, np.max((2 * k + 1, 20))))

    if E is None:
        E = IdentityOperator(A.source)

    assert A.source == A.range
    assert E.source == A.source
    assert E.range == A.source
    assert k < n
    assert l > k

    if b is None:
        b = A.source.random()

    V, H, f = arnoldi(A, E, k, b)
    k0 = k
    i = 0

    while True:
        i = i + 1

        V, H, f = extend_arnoldi(A, E, V, H, f, l - k)

        ew, ev = spla.eig(H)

        # truncate small imaginary parts
        ew.imag[np.abs(ew.imag) / np.abs(ew) < 1e-12] = 0

        if which == 'LM':
            idx = np.argsort(-np.abs(ew))
        elif which == 'SM':
            idx = np.argsort(np.abs(ew))
        elif which == 'LR':
            idx = np.argsort(-np.real(ew))
        elif which == 'SR':
            idx = np.argsort(np.real(ew))
        elif which == 'LI':
            idx = np.argsort(-np.abs(np.imag(ew)))
        elif which == 'SI':
            idx = np.argsort(np.abs(np.imag(ew)))

        k = k0
        ews = ew[idx]
        evs = ev[:, idx]

        rres = f.l2_norm()[0] * np.abs(evs[l - 1]) / np.abs(ews)

        # increase k by one in order to keep complex conjugate pairs together
        if ews[k - 1].imag != 0 and ews[k - 1].imag + ews[k].imag < 1e-12:
            k = k + 1

        if np.all(rres[:k] <= tol) or i >= maxiter:
            break

        # increase k in order to prevent stagnation
        k = np.min((l - 1, k + np.min((np.count_nonzero(rres[:k] <= tol), (l - k) // 2))))

        # sort shifts for QR iteration based on their residual
        shifts = ews[k:l]
        srres = rres[k:l]
        idx = np.argsort(-srres)
        srres = srres[idx]
        shifts = shifts[idx]

        # don't use converged unwanted ritzvalues as shifts
        shifts = np.delete(shifts, np.where(srres == 0))
        k = k + np.count_nonzero(srres == 0)
        if shifts[0].imag != 0 and shifts[0].imag + ews[1].imag >= 1e-12:
            shifts = shifts[1:]
            k = k + 1

        H, Qs = QR_iteration(H, shifts)

        V = V.lincomb(Qs.T)
        f = V[k] * H[k, k - 1] + f * Qs[l - 1, k - 1]
        V = V[:k]
        H = H[:k, :k]

    return ews[:k0], V.lincomb(evs[:, :k0].T)


def arnoldi(A, E, l, b):
    """Compute an Arnoldi factorization.

    Computes matrices :math:`V_l` and :math:`H_l` and a vector :math:`f_l` such that

    .. math::
        A V_l = V_l H_l + f_l e_l^T.

    Additionally it holds that :math:`V_l^T V_l` is the identity matrix and :math:`H_l`
    is an upper Hessenberg matrix. If `E` is not `None` it holds

    .. math::
        E^{-1} A V_l = V_l H_l + f_l e_l^T.

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E.
    l
        The length of the Arnoldi factorization.
    b
        A |VectorArray| which is used as the initial vector for the iteration.

    Returns
    -------
    V
        A |VectorArray| whose columns span an orthogonal basis for R^l.
    H
        A |NumPy array| which is an upper Hessenberg matrix.
    f
        A |VectorArray| which represents the residual vector of the Arnoldi factorzation.
    """

    v = b * (1 / b.l2_norm()[0])

    H = np.zeros((l, l))
    V = A.source.empty(reserve=l)

    V.append(v)

    for i in range(l):
        v = E.apply_inverse(A.apply(v))
        V.append(v)

        _, R = gram_schmidt(V, return_R=True, atol=0, rtol=0, offset=len(V) - 1, copy=False)
        H[:i + 2, i] = R[:l, i + 1]
        v = V[-1]

    return V[:l], H, v * R[l, l]


def extend_arnoldi(A, E, V, H, f, p):
    """Extend an existing Arnoldi factorization.

    Assuming that the inputs `V`, `H` and `f` define an Arnoldi factorization of length
    :math:`l` (see :func:`arnoldi`), computes matrices :math:`V_{l+p}` and :math:`H_{l+p}`
    and a vector :math:`f_{l+p}` which extend the factorization to a length `l+p` Arnoldi
    factorization.

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E.
    V
        The |VectorArray| V from the length :math:`l` Arnoldi factorization.
    H
        The |NumPy array| H from the length :math:`l` Arnoldi factorization.
    f
        The |VectorArray| f from the length :math:`l` Arnoldi factorization.
    p
        The number of addditional Arnoldi steps which are to be performed.

    Returns
    -------
    V
        A |VectorArray| whose columns span an orthogonal basis for R^(l+p).
    H
        A |NumPy array| which is an upper Hessenberg matrix.
    f
        A |VectorArray| which represents the residual vector of the Arnoldi factorzation.
    """

    k = len(V)

    res = f.l2_norm()[0]
    H = np.pad(H, ((0, p), (0, p)))
    H[k, k - 1] = res
    v = f * (1 / res)
    # since i cannot append to the VectorArrayView V I copy it before appending...
    # is there a better way to do this?
    V = V.copy()
    V.append(v)

    for i in range(k, k + p):
        v = E.apply_inverse(A.apply(v))
        V.append(v)

        _, R = gram_schmidt(V, return_R=True, atol=0, rtol=0, offset=len(V) - 1, copy=False)
        H[:i + 2, i] = R[:k + p, i + 1]

        v = V[-1]

    return V[:k + p], H, v * R[k + p, k + p]


def QR_iteration(H, shifts):
    """Perform the QR iteration.

    Performs a QR step for each shift provided in `shifts`. `H` is assumed to be an
    unreduced upper Hessenberg matrix. If a complex shift occurs a double step is
    peformed in order to avoid complex arithmetic.

    Parameters
    ----------
    H
        The |NumPy array| H which is an unreduced upper Hessenberg matrix.
    shifts
        A |NumPy array| which contains the shifts that are to be applied in the QR steps.

    Returns
    -------
    Hs
        A |NumPy array| in upper Hessenberg form such that it holds :math:`H Q_s = Q_s H_s`.
    Qs
        The product of the orthogonal matrices computed in each QR step.
    """

    Qs = np.eye(len(H))

    i = 0
    while i < len(shifts) - 1:
        s = shifts[i]
        if shifts[i].imag != 0:
            Q, R = np.linalg.qr(H @ H - 2 * np.real(s) * H + np.abs(s)**2 * np.eye(len(H)))
            i = i + 2
        else:
            Q, R = np.linalg.qr(H - s * np.eye(len(H)))
            i = i + 1
        Qs = Qs @ Q
        H = Q.T @ H @ Q

    return H, Qs
