# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.operators.constructions import IdentityOperator
from pymor.core.defaults import defaults
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.logger import getLogger


@defaults('tol', 'imagtol', 'conjtol', 'rqitol', 'maxrestart', 'krestart', 'init_shifts',
          'rqi_maxiter')
def samdp(A, E, B, C, nwanted, init_shifts=None, tol=1e-10, imagtol=1e-8, conjtol=1e-8,
          rqitol=1e-4, maxrestart=100, krestart=10, rqi_maxiter=10):
    """Compute the dominant pole triplets and residues of the transfer function of an LTI system.

    This function uses the subspace accelerated dominant pole (samdp) algorithm as described in
    [RM06]_ in Algorithm 2 in order to compute dominant pole triplets and residues of the transfer
    function

    .. math::
        H(s) = C^T (s E - A)^{-1} B + D

    of an LTI system. It is possible to take advantage of prior knowledge about the poles
    by specifying shift parameters, which are injected after a new pole has been found.

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E or `None`.
    B
        The |Operator| B.
    C
        The |Operator| C.
    nwanted
        The number of dominant poles that should be computed.
    init_shifts
        A |NumPy array| containing shifts which are injected after a new pole has been found.
    tol
        Tolerance for the residual of the poles.
    imagtol
        Relative tolerance for imaginary parts of pairs of complex conjugate eigenvalues.
    conjtol
        Tolerance for the residual of the complex conjugate of a pole.
    rqitol
        If the residual is smaller than rqitol the two-sided Rayleigh quotient iteration
        is executed.
    maxrestart
        The maximum number of restarts.
    krestart
        Maximum dimension of search space before performing a restart.
    rqi_maxiter
        Maximum number of iterations for the two-sided Rayleigh quotient iteration.

    Returns
    -------
    poles
        A |NumPy array| containing the computed dominant poles.
    residues
        A |NumPy array| containing the computed residues.
    rightev
        A |VectorArray| containing the right eigenvectors of the computed poles.
    leftev
        A |VectorArray| containing the left eigenvectors of the computed poles.
    """

    logger = getLogger('pymor.algorithms.samdp.samdp')

    if E is None:
        E = IdentityOperator(A.source)

    B = B.as_range_array()
    C = C.as_source_array()

    B_defl = B.copy()
    C_defl = C.copy()

    k = 0
    nrestart = 0
    nr_converged = 0

    X = A.source.empty()
    Q = A.source.empty()
    Qt = A.source.empty()
    Qs = A.source.empty()
    Qts = A.source.empty()
    AX = A.source.empty()
    V = A.source.empty()

    H = np.empty((0, 1))
    G = np.empty((0, 1))
    poles = np.empty((1, 0))

    if init_shifts is None:
        st = 0.  # does it make sense to initlize the shift?
        shift_nr = 0
        nr_shifts = 0
    else:
        st = init_shifts[0]
        shift_nr = 1
        nr_shifts = len(init_shifts)

    shifts = init_shifts

    while nrestart < maxrestart:
        k = k + 1

        sEmA = st * E - A
        sEmAB = sEmA.apply_inverse(B_defl)
        Hs = C_defl.dot(sEmAB)

        y_all, _, u_all = spla.svd(Hs)  # use scipy.sparse.linalg.svds(Hs, k=1) instead?

        u = u_all.conj()[0]
        y = y_all[:, 0]

        x = sEmAB.lincomb(u)
        v = sEmA.apply_inverse_adjoint((C_defl.lincomb(y.T)))

        X.append(x)
        V.append(v)
        V = gram_schmidt(V, atol=0, rtol=0, offset=len(V) - 1, copy=False)
        X = gram_schmidt(X, atol=0, rtol=0, offset=len(X) - 1, copy=False)

        AX.append(A.apply(X[k-1]))

        if k > 1:
            H = np.append(H, V[0:k-1].dot(AX[k-1]), axis=1)
        H = np.append(H, V[k-1].dot(AX), axis=0)
        EX = E.apply(X)
        if k > 1:
            G = np.append(G, V[0:k-1].dot(EX[k-1]), axis=1)
        G = np.append(G, V[k-1].dot(EX), axis=0)

        SH, UR, URt = select_max_eig(H, G, X, V, B_defl, C_defl, E)

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

            if nres < rqitol and do_rqi:
                schurvec, lschurvec, theta, nres = twosided_rqi(A, E, schurvec, lschurvec, theta,
                                                                nres, tol, imagtol, rqi_maxiter)
                do_rqi = False
                if np.abs(np.imag(theta)) / np.abs(theta) < imagtol:
                    rres = A.apply(schurvec.real) - E.apply(schurvec.real) * np.real(theta)
                    nrr = rres.norm()
                    if nrr < nres:
                        schurvec = schurvec.real
                        lschurvec = lschurvec.real
                        theta = np.real(theta)
                        nres = nrr
            found = nres < tol and nr_converged < nwanted

            if found:
                poles = np.append(poles, theta)
                logger.info(f'Pole: {theta:.5e}')

                Q.append(schurvec)
                Qt.append(lschurvec)
                Esch = E.apply(schurvec)
                Qs.append(Esch)
                Qts.append(E.apply_adjoint(lschurvec))

                nqqt = lschurvec.dot(Esch)[0][0]
                Q[-1].scal(1 / nqqt)
                Qs[-1].scal(1 / nqqt)

                nr_converged = nr_converged + 1

                if k > 1:
                    X = X.lincomb(UR[:, 1:k].T)
                    V = V.lincomb(URt[:, 1:k].T)
                else:
                    X = A.source.empty()
                    V = A.source.empty()

                if np.abs(np.imag(theta)) / np.abs(theta) < imagtol:
                    V = gram_schmidt(V, atol=0, rtol=0, copy=False)
                    X = gram_schmidt(X, atol=0, rtol=0, copy=False)

                B_defl = B_defl - E.apply(Q[-1].lincomb(Qt[-1].dot(B_defl).T))
                C_defl = C_defl - E.apply_adjoint(Qt[-1].lincomb(Q[-1].dot(C_defl).T))

                k = k - 1

                cce = theta.conj()
                if np.abs(np.imag(cce)) / np.abs(cce) >= imagtol:

                    pairdifs = np.abs(poles - cce)

                    if np.min(pairdifs) > tol:
                        ccv = schurvec.conj()
                        ccv.scal(1 / ccv.norm())

                        r = A.apply(ccv) - E.apply(ccv) * cce

                        if r.norm() < conjtol:
                            logger.info(f'Conjugate Pole: {cce:.5e}')
                            poles = np.append(poles, cce)

                            Q.append(ccv)
                            ccvt = lschurvec.conj()
                            Qt.append(ccvt)

                            Esch = E.apply(ccv)
                            Qs.append(Esch)
                            Qts.append(E.apply_adjoint(ccvt))

                            nqqt = ccvt.dot(E.apply(ccv))[0][0]
                            Q[-1].scal(1 / nqqt)
                            Qs[-1].scal(1 / nqqt)

                            V = gram_schmidt(V, atol=0, rtol=0, copy=False)
                            X = gram_schmidt(X, atol=0, rtol=0, copy=False)

                            B_defl = B_defl - E.apply(Q[-1].lincomb(Qt[-1].dot(B_defl).T))
                            C_defl = C_defl - E.apply_adjoint(Qt[-1].lincomb(Q[-1].dot(C_defl).T))

                AX = A.apply(X)
                if k > 0:
                    G = V.dot(E.apply(X))
                    H = V.dot(AX)
                    SH, UR, URt = select_max_eig(H, G, X, V, B_defl, C_defl, E)
                    found = True
                else:
                    G = np.empty((0, 1))
                    H = np.empty((0, 1))
                    found = False

                if nr_converged < nwanted:
                    if found:
                        st = SH[0, 0]
                    else:
                        st = np.random.uniform() * 1.j

                    if shift_nr < nr_shifts:
                        st = shifts[shift_nr]
                        shift_nr = shift_nr + 1
            elif k >= krestart:
                logger.info(f'Perform restart...')
                EX = E.apply(X)
                RR = AX.lincomb(UR.T) - EX.lincomb(UR.T).lincomb(SH.T)
                minNorm = RR[0].norm()
                minidx = 0
                for i in range(1, len(RR)):
                    norm = RR[i].norm()
                    if norm < minNorm:
                        minidx = i
                        minNorm = norm

                k = 1

                X = X.lincomb(UR[:, minidx])
                V = V.lincomb(URt[:, minidx])

                V = gram_schmidt(V, atol=0, rtol=0, copy=False)
                X = gram_schmidt(X, atol=0, rtol=0, copy=False)

                G = V.dot(E.apply(X))
                AX = A.apply(X)
                H = V.dot(AX)
                nrestart = nrestart + 1

        if nr_converged == nwanted or nrestart == maxrestart:
            rightev = Q
            leftev = Qt
            absres = np.array([])
            residues = np.empty((len(C), len(B), 0))
            for i in range(len(poles)):
                leftev[i].scal(1 / (leftev[i].dot(E.apply(rightev[i].conj())))[0][0])
                residues = np.dstack((residues, C.dot(rightev[i]) @ (leftev[i].dot(B))))
                absres = np.append(absres, spla.norm(residues[:, :, i], 2))
            idx = np.argsort(-absres)
            residues = residues[:, :, idx]
            poles = poles[idx]
            rightev = rightev[idx]
            leftev = leftev[idx]
            if nr_converged < nwanted:
                logger.warning(f'The specified number of poles could not be computed.')
            break

    return poles, residues, rightev, leftev


def twosided_rqi(A, E, x, y, theta, init_res, tol, imagtol, maxiter):
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
    tol
        Convergence tolerance for the Rayleigh qotient iteration.
    imagtol
        Relative tolerance for imaginary parts of pairs of complex conjugate eigenvalues.
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

    iter = 0
    nrq = 1
    while nrq > tol and iter < maxiter:
        iter = iter + 1
        Ex = E.apply(x)
        Ey = E.apply_adjoint(y)
        tEmA = theta * E - A
        x_rqi = tEmA.apply_inverse(Ex)
        v_rqi = tEmA.apply_inverse_adjoint(Ey)

        x_rqi.scal(1 / x_rqi.norm())
        v_rqi.scal(1 / v_rqi.norm())

        Ax_rqi = A.apply(x_rqi)
        Ex_rqi = E.apply(x_rqi)

        x_rq = (v_rqi.dot(Ax_rqi) / v_rqi.dot(Ex_rqi))[0][0]
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
        theta = x_rq
        nrq = rqi_res.norm()

        if not np.isfinite(nrq):
            nrq = 1
    if nrq < init_res:
        return x_rqi, v_rqi, x_rq, nrq
    else:
        return x, y, theta, init_res


def select_max_eig(H, G, X, V, B, C, E):
    """Compute poles sorted from largest to smallest residual.

    Parameters
    ----------
    H
        The |Numpy array| H from the samdp algorithm.
    G
        The |Numpy array| G from the samdp algorithm.
    X
        A |VectorArray| describing the orthogonal search space used in the samdp algorithm.
    V
        A |VectorArray| describing the orthogonal search space used in the samdp algorithm.
    B
        The |VectorArray| B from the corresponding LTI system modified by deflation.
    C
        The |VectorArray| C from the corresponding LTI system modified by deflation.
    E
        The |Operator| E from the corresponding LTI system.

    Returns
    -------
    poles
        A |NumPy array| containing poles sorted from largest to smallest residual.
    rightevs
        A |NumPy array| containing the right eigenvectors of the computed poles.
    leftevs
        A |NumPy array| containing the left eigenvectors of the computed poles.
    """

    D, Vs = spla.eig(H, G)
    idx = np.argsort(D)
    DP = D[idx]
    Vs = Vs[:, idx]

    Dt, Vt = spla.eig(H.conj().T, G.conj().T)
    Vt = Vt[:, np.argsort(Dt)]

    X = X.lincomb(Vs.T)
    V = V.lincomb(Vt.T)

    residue = np.array([])

    for i in range(len(H)):
        V[i].scal(1 / V[i].norm())
        X[i].scal(1 / X[i].norm())
        residue = np.append(residue, spla.norm(C.dot(X[i]) @ V[i].dot(B), 2))

    idx = np.argsort(-residue)

    return np.diag(DP[idx]), Vs[:, idx], Vt[:, idx]
