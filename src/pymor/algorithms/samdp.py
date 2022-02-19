# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.constructions import IdentityOperator
from pymor.operators.interface import Operator


@defaults('which', 'tol', 'imagtol', 'conjtol', 'dorqitol', 'rqitol', 'maxrestart', 'krestart', 'init_shifts',
          'rqi_maxiter', 'seed')
def samdp(A, E, B, C, nwanted, init_shifts=None, which='NR', tol=1e-10, imagtol=1e-6, conjtol=1e-8,
          dorqitol=1e-4, rqitol=1e-10, maxrestart=100, krestart=20, rqi_maxiter=10, seed=0):
    """Compute the dominant pole triplets and residues of the transfer function of an LTI system.

    This function uses the subspace accelerated dominant pole (SAMDP) algorithm as described in
    :cite:`RM06` in Algorithm 2 in order to compute dominant pole triplets and residues of the
    transfer function

    .. math::
        H(s) = C (s E - A)^{-1} B

    of an LTI system. It is possible to take advantage of prior knowledge about the poles
    by specifying shift parameters, which are injected after a new pole has been found.

    .. note::
        Pairs of complex conjugate eigenvalues are always returned together. Accordingly, the
        number of returned poles can be equal to `nwanted + 1`.

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    C
        The operator C as a |VectorArray| from `A.source`.
    nwanted
        The number of dominant poles that should be computed.
    init_shifts
        A |NumPy array| containing shifts which are injected after a new pole has been found.
    which
        A string specifying the strategy by which the dominant poles and residues are selected.
        Possible values are:

        - `'NR'`: select poles with largest norm(residual) / abs(Re(pole))
        - `'NS'`: select poles with largest norm(residual) / abs(pole)
        - `'NM'`: select poles with largest norm(residual)
    tol
        Tolerance for the residual of the poles.
    imagtol
        Relative tolerance for imaginary parts of pairs of complex conjugate eigenvalues.
    conjtol
        Tolerance for the residual of the complex conjugate of a pole.
    dorqitol
        If the residual is smaller than dorqitol the two-sided Rayleigh quotient iteration
        is executed.
    rqitol
        Tolerance for the residual of a pole in the two-sided Rayleigh quotient iteration.
    maxrestart
        The maximum number of restarts.
    krestart
        Maximum dimension of search space before performing a restart.
    rqi_maxiter
        Maximum number of iterations for the two-sided Rayleigh quotient iteration.
    seed
        Random seed which is used for computing the initial shift and random restarts.

    Returns
    -------
    poles
        A 1D |NumPy array| containing the computed dominant poles.
    residues
        A 3D |NumPy array| of shape `(len(poles), len(C), len(B))` containing the computed residues.
    rightev
        A |VectorArray| containing the right eigenvectors of the computed poles.
    leftev
        A |VectorArray| containing the left eigenvectors of the computed poles.
    """
    logger = getLogger('pymor.algorithms.samdp.samdp')

    if E is None:
        E = IdentityOperator(A.source)

    assert isinstance(A, Operator) and A.linear
    assert not A.parametric
    assert A.source == A.range
    if E is not None:
        assert isinstance(E, Operator) and E.linear
        assert not E.parametric
        assert E.source == E.range
        assert E.source == A.source
    assert B in A.source
    assert C in A.source

    B_defl = B.copy()
    C_defl = C.copy()

    k = 0
    nrestart = 0
    nr_converged = 0
    np.random.seed(seed)

    X = A.source.empty()
    Q = A.source.empty()
    Qt = A.source.empty()
    Qs = A.source.empty()
    Qts = A.source.empty()
    AX = A.source.empty()
    V = A.source.empty()

    H = np.empty((0, 1))
    G = np.empty((0, 1))
    poles = np.empty(0)

    if init_shifts is None:
        st = np.random.uniform() * 10.j
        shift_nr = 0
        nr_shifts = 0
    else:
        st = init_shifts[0]
        shift_nr = 1
        nr_shifts = len(init_shifts)

    shifts = init_shifts

    while nrestart < maxrestart:
        k += 1

        sEmA = st * E - A
        sEmAB = sEmA.apply_inverse(B_defl)
        Hs = C_defl.inner(sEmAB)

        y_all, _, u_all = spla.svd(Hs)

        u = u_all.conj()[0]
        y = y_all[:, 0]

        x = sEmAB.lincomb(u)
        v = sEmA.apply_inverse_adjoint(C_defl.lincomb(y.T))

        X.append(x)
        V.append(v)
        gram_schmidt(V, atol=0, rtol=0, copy=False)
        gram_schmidt(X, atol=0, rtol=0, copy=False)

        AX.append(A.apply(X[k-1]))

        if k > 1:
            H = np.hstack((H, V[0:k-1].inner(AX[k-1])))
        H = np.vstack((H, V[k-1].inner(AX)))
        EX = E.apply(X)
        if k > 1:
            G = np.hstack((G, V[0:k-1].inner(EX[k-1])))
        G = np.vstack((G, V[k-1].inner(EX)))

        SH, UR, URt, res = _select_max_eig(H, G, X, V, B_defl, C_defl, which)

        if np.all(res < np.finfo(float).eps):
            st = np.random.uniform() * 10.j
            found = False
        else:
            found = True

        do_rqi = True
        while found:
            theta = SH[0, 0]
            schurvec = X.lincomb(UR[:, 0])
            schurvec.scal(1 / schurvec.norm())
            lschurvec = V.lincomb(URt[:, 0])
            lschurvec.scal(1 / lschurvec.norm())

            st = theta

            nres = (A.apply(schurvec) - (E.apply(schurvec) * theta)).norm()[0]

            logger.info(f'Step: {k}, Theta: {theta:.5e}, Residual: {nres:.5e}')

            if np.abs(np.imag(theta)) / np.abs(theta) < imagtol:
                rres = A.apply(schurvec.real) - E.apply(schurvec.real) * np.real(theta)
                nrr = rres.norm()
                if np.abs(nrr - nres) < np.finfo(float).eps:
                    schurvec = schurvec.real
                    lschurvec = lschurvec.real
                    theta = np.real(theta)
                    nres = nrr

            if nres < dorqitol and do_rqi and nres >= tol:
                schurvec, lschurvec, theta, nres = _twosided_rqi(A, E, schurvec, lschurvec, theta,
                                                                 nres, imagtol, rqitol, rqi_maxiter)
                do_rqi = False
                if np.abs(np.imag(theta)) / np.abs(theta) < imagtol:
                    rres = A.apply(schurvec.real) - E.apply(schurvec.real) * np.real(theta)
                    nrr = rres.norm()
                    if np.abs(nrr - nres) < np.finfo(float).eps:
                        schurvec = schurvec.real
                        lschurvec = lschurvec.real
                        theta = np.real(theta)
                        nres = nrr
                if nres >= tol:
                    logger.warning('Two-sided RQI did not reach desired tolerance.')

            found = nr_converged < nwanted and nres < tol

            if found:
                poles = np.append(poles, theta)
                logger.info(f'Pole: {theta:.5e}')

                Q.append(schurvec)
                Qt.append(lschurvec)
                Esch = E.apply(schurvec)
                Qs.append(Esch)
                Qts.append(E.apply_adjoint(lschurvec))

                nqqt = lschurvec.inner(Esch)[0][0]
                Q[-1].scal(1 / nqqt)
                Qs[-1].scal(1 / nqqt)

                nr_converged += 1

                if k > 1:
                    X = X.lincomb(UR[:, 1:k].T)
                    V = V.lincomb(URt[:, 1:k].T)
                else:
                    X = A.source.empty()
                    V = A.source.empty()

                if np.abs(np.imag(theta)) / np.abs(theta) < imagtol:
                    gram_schmidt(V, atol=0, rtol=0, copy=False)
                    gram_schmidt(X, atol=0, rtol=0, copy=False)

                B_defl -= E.apply(Q[-1].lincomb(Qt[-1].inner(B_defl).T))
                C_defl -= E.apply_adjoint(Qt[-1].lincomb(Q[-1].inner(C_defl).T))

                k -= 1

                cce = theta.conj()
                if np.abs(np.imag(cce)) / np.abs(cce) >= imagtol:

                    ccv = schurvec.conj()
                    ccv.scal(1 / ccv.norm())

                    r = A.apply(ccv) - E.apply(ccv) * cce

                    if r.norm() / np.abs(cce) < conjtol:
                        logger.info(f'Conjugate Pole: {cce:.5e}')
                        poles = np.append(poles, cce)
                        nr_converged += 1

                        Q.append(ccv)
                        ccvt = lschurvec.conj()
                        Qt.append(ccvt)

                        Esch = E.apply(ccv)
                        Qs.append(Esch)
                        Qts.append(E.apply_adjoint(ccvt))

                        nqqt = ccvt.inner(E.apply(ccv))[0][0]
                        Q[-1].scal(1 / nqqt)
                        Qs[-1].scal(1 / nqqt)

                        gram_schmidt(V, atol=0, rtol=0, copy=False)
                        gram_schmidt(X, atol=0, rtol=0, copy=False)

                        B_defl -= E.apply(Q[-1].lincomb(Qt[-1].inner(B_defl).T))
                        C_defl -= E.apply_adjoint(Qt[-1].lincomb(Q[-1].inner(C_defl).T))

                AX = A.apply(X)
                if k > 0:
                    G = V.inner(E.apply(X))
                    H = V.inner(AX)
                    SH, UR, URt, residues = _select_max_eig(H, G, X, V, B_defl, C_defl, which)
                    found = np.any(res >= np.finfo(float).eps)
                else:
                    G = np.empty((0, 1))
                    H = np.empty((0, 1))
                    found = False

                if nr_converged < nwanted:
                    if found:
                        st = SH[0, 0]
                    else:
                        st = np.random.uniform() * 10.j

                    if shift_nr < nr_shifts:
                        st = shifts[shift_nr]
                        shift_nr += 1
            elif k >= krestart:
                logger.info('Perform restart...')
                EX = E.apply(X)
                RR = AX.lincomb(UR.T) - EX.lincomb(UR.T).lincomb(SH.T)

                minidx = RR.norm().argmin()
                k = 1

                X = X.lincomb(UR[:, minidx])
                V = V.lincomb(URt[:, minidx])

                gram_schmidt(V, atol=0, rtol=0, copy=False)
                gram_schmidt(X, atol=0, rtol=0, copy=False)

                G = V.inner(E.apply(X))
                AX = A.apply(X)
                H = V.inner(AX)
                nrestart += 1

        if k >= krestart:
            logger.info('Perform restart...')
            EX = E.apply(X)
            RR = AX.lincomb(UR.T) - EX.lincomb(UR.T).lincomb(SH.T)

            minidx = RR.norm().argmin()
            k = 1

            X = X.lincomb(UR[:, minidx])
            V = V.lincomb(URt[:, minidx])

            gram_schmidt(V, atol=0, rtol=0, copy=False)
            gram_schmidt(X, atol=0, rtol=0, copy=False)

            G = V.inner(E.apply(X))
            AX = A.apply(X)
            H = V.inner(AX)
            nrestart += 1

        if nr_converged >= nwanted or nrestart == maxrestart:
            rightev = Q
            leftev = Qt
            absres = np.empty(len(poles))
            residues = []
            for i in range(len(poles)):
                leftev[i].scal(1 / leftev[i].inner(E.apply(rightev[i]))[0][0])
                residues.append(C.inner(rightev[i]) @ leftev[i].inner(B))
                absres[i] = spla.norm(residues[-1], ord=2)
            residues = np.array(residues)

            if which == 'NR':
                idx = np.argsort(-absres / np.abs(np.real(poles)))
            elif which == 'NS':
                idx = np.argsort(-absres / np.abs(poles))
            elif which == 'NM':
                idx = np.argsort(-absres)
            else:
                raise ValueError('Unknown SAMDP selection strategy.')

            residues = residues[idx]
            poles = poles[idx]
            rightev = rightev[idx]
            leftev = leftev[idx]
            if nr_converged < nwanted:
                logger.warning('The specified number of poles could not be computed.')
            break

    return poles, residues, rightev, leftev


def _twosided_rqi(A, E, x, y, theta, init_res, imagtol, rqitol, maxiter):
    """Refine an initial guess for an eigentriplet of the matrix pair (A, E).

    Parameters
    ----------
    A
        The |Operator| A from the LTI system.
    E
        The |Operator| E from the LTI system.
    x
        Initial guess for right eigenvector of matrix pair (A, E).
    y
        Initial guess for left eigenvector of matrix pair (A, E).
    theta
        Initial guess for eigenvalue of matrix pair (A, E).
    init_res
        Residual of initial guess.
    imagtol
        Relative tolerance for imaginary parts of pairs of complex conjugate eigenvalues.
    rqitol
        Convergence tolerance for the residual of the pole.
    maxiter
        Maximum number of iteration.

    Returns
    -------
    x
        Refined right eigenvector of matrix pair (A, E).
    y
        Refined left eigenvector of matrix pair (A, E).
    theta
        Refined eigenvalue of matrix pair (A, E).
    residual
        Residual of the computed triplet.
    """
    i = 0
    nrq = 1
    while i < maxiter:
        i += 1
        Ex = E.apply(x)
        Ey = E.apply_adjoint(y)
        tEmA = theta * E - A
        x_rqi = tEmA.apply_inverse(Ex)
        v_rqi = tEmA.apply_inverse_adjoint(Ey)

        x_rqi.scal(1 / x_rqi.norm())
        v_rqi.scal(1 / v_rqi.norm())

        Ax_rqi = A.apply(x_rqi)
        Ex_rqi = E.apply(x_rqi)

        x_rq = (v_rqi.inner(Ax_rqi) / v_rqi.inner(Ex_rqi))[0][0]
        if not np.isfinite(x_rq):
            x_rqi = x
            v_rqi = y
            x_rq = theta + 1e-10

        rqi_res = Ax_rqi - Ex_rqi * x_rq
        if np.abs(np.imag(x_rq)) / np.abs(x_rq) < imagtol:
            rx_rqi = np.real(x_rqi)
            rx_rqi.scal(1 / rx_rqi.norm())

            rres = A.apply(rx_rqi) - E.apply(rx_rqi) * np.real(x_rq)
            nrr = rres.norm()
            if nrr < rqi_res.norm():
                x_rqi = rx_rqi
                v_rqi = np.real(v_rqi)
                v_rqi.scal(1 / v_rqi.norm())
                x_rq = np.real(x_rq)
                rqi_res = rres

        x = x_rqi
        y = v_rqi
        nrq = rqi_res.norm()
        if nrq < rqitol:
            break
        theta = x_rq
        if not np.isfinite(nrq):
            nrq = 1
    if nrq < init_res:
        return x_rqi, v_rqi, x_rq, nrq
    else:
        return x, y, theta, init_res


def _select_max_eig(H, G, X, V, B, C, which):
    """Compute poles sorted from largest to smallest residual.

    Parameters
    ----------
    H
        The |Numpy array| H from the SAMDP algorithm.
    G
        The |Numpy array| G from the SAMDP algorithm.
    X
        A |VectorArray| describing the orthogonal search space used in the SAMDP algorithm.
    V
        A |VectorArray| describing the orthogonal search space used in the SAMDP algorithm.
    B
        The |VectorArray| B from the corresponding LTI system modified by deflation.
    C
        The |VectorArray| C from the corresponding LTI system modified by deflation.
    which
        A string that indicates which poles to select. See :func:`samdp`.

    Returns
    -------
    poles
        A |NumPy array| containing poles sorted according to the chosen strategy.
    rightevs
        A |NumPy array| containing the right eigenvectors of the computed poles.
    leftevs
        A |NumPy array| containing the left eigenvectors of the computed poles.
    residue
        A 1D |NumPy array| containing the norms of the residues.
    """
    D, Vt, Vs = spla.eig(H, G, left=True)
    idx = np.argsort(D)
    DP = D[idx]
    Vs = Vs[:, idx]
    Vt = Vt[:, idx]

    X = X.lincomb(Vs.T)
    V = V.lincomb(Vt.T)

    V.scal(1 / V.norm())
    X.scal(1 / X.norm())
    residue = spla.norm(C.inner(X), axis=0) * spla.norm(V.inner(B), axis=1)

    if which == 'NR':
        idx = np.argsort(-residue / np.abs(np.real(DP)))
    elif which == 'NS':
        idx = np.argsort(-residue / np.abs(DP))
    elif which == 'NM':
        idx = np.argsort(-residue)
    else:
        raise ValueError('Unknown SAMDP selection strategy.')

    return np.diag(DP[idx]), Vs[:, idx], Vt[:, idx], residue
