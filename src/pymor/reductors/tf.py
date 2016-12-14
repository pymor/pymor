# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.to_matrix import to_matrix
from pymor.discretizations.iosys import LTISystem
from pymor.operators.constructions import IdentityOperator


def interpolation(discretization, sigma, b, c):
    """Realization-independent tangential Hermite interpolation.

    Parameters
    ----------
    sigma
        Interpolation points (closed under conjugation), list of length `r`.
    b
        Right tangential directions, |NumPy array| of shape
        `(discretization.m, r)`.
    c
        Left tangential directions, |NumPy array| of shape
        `(discretization.p, r)`.

    Returns
    -------
    lti
        |LTISystem| interpolating the transfer function of `discretization`.
    """
    r = len(sigma)
    assert isinstance(b, np.ndarray) and b.shape == (discretization.m, r)
    assert isinstance(c, np.ndarray) and c.shape == (discretization.p, r)

    # rescale tangential directions (could avoid overflow or underflow)
    for i in range(r):
        b[:, i] /= spla.norm(b[:, i])
        c[:, i] /= spla.norm(c[:, i])

    # matrices of the interpolatory LTI system
    Er = np.empty((r, r), dtype=complex)
    Ar = np.empty((r, r), dtype=complex)
    Br = np.empty((r, discretization.m), dtype=complex)
    Cr = np.empty((discretization.p, r), dtype=complex)

    Hs = [discretization.eval_tf(s) for s in sigma]
    dHs = [discretization.eval_dtf(s) for s in sigma]

    for i in range(r):
        for j in range(r):
            if i != j:
                Er[i, j] = -c[:, i].dot((Hs[i] - Hs[j]).dot(b[:, j])) / (sigma[i] - sigma[j])
                Ar[i, j] = -c[:, i].dot((sigma[i] * Hs[i] - sigma[j] * Hs[j])).dot(b[:, j]) / (sigma[i] - sigma[j])
            else:
                Er[i, i] = -c[:, i].dot(dHs[i].dot(b[:, i]))
                Ar[i, i] = -c[:, i].dot((Hs[i] + sigma[i] * dHs[i]).dot(b[:, i]))
        Br[i, :] = Hs[i].T.dot(c[:, i])
        Cr[:, i] = Hs[i].dot(b[:, i])

    # transform the system to have real matrices
    T = np.zeros((r, r), dtype=complex)
    for i in range(r):
        if sigma[i].imag == 0:
            T[i, i] = 1
        else:
            try:
                j = i + 1 + np.where(np.isclose(sigma[i + 1:], sigma[i].conjugate()))[0][0]
            except:
                j = None
            if j:
                T[i, i] = 1
                T[i, j] = 1
                T[j, i] = -1j
                T[j, j] = 1j
    Er = (T.dot(Er).dot(T.conj().T)).real
    Ar = (T.dot(Ar).dot(T.conj().T)).real
    Br = (T.dot(Br)).real
    Cr = (Cr.dot(T.conj().T)).real

    return LTISystem.from_matrices(Ar, Br, Cr, D=None, E=Er, cont_time=discretization.cont_time)


def tf_irka(discretization, r, sigma=None, b=None, c=None, tol=1e-4, maxit=100, verbose=False, force_sigma_in_rhp=False,
            conv_crit='rel_sigma_change'):
    """Reduce using TF-IRKA.

    .. [AG12] C. A. Beattie, S. Gugercin, Realization-independent
                H2-approximation,
                Proceedings of the 51st IEEE Conference on Decision and
                Control, 2012.

    Parameters
    ----------
    r
        Order of the reduced order model.
    sigma
        Initial interpolation points (closed under conjugation), list of
        length `r`.

        If `None`, interpolation points are log-spaced between 0.1 and 10.
    b
        Initial right tangential directions, |NumPy array| of shape
        `(discretization.m, r)`.

        If `None`, `b` is chosen with all ones.
    c
        Initial left tangential directions, |NumPy array| of shape
        `(discretization.p, r)`.

        If `None`, `c` is chosen with all ones.
    tol
        Tolerance for the largest change in interpolation points.
    maxit
        Maximum number of iterations.
    verbose
        Should consecutive distances be printed.
    force_sigma_in_rhp
        If 'False`, new interpolation are reflections of reduced order
        model's poles. Otherwise, they are always in the right
        half-plane.
    conv_crit
        Convergence criterion:
            - `'rel_sigma_change'`: relative change in interpolation points
            - `'rel_H2_dist'`: relative H_2 distance of reduced order models

    Returns
    -------
    rom
        Reduced |LTISystem| model.
    reduction_data
        Dictionary of additional data produced by the reduction process.
        Contains:

        - distances between interpolation points in subsequent iterations
          `dist`,
        - interpolation points from all iterations `Sigma`, and
        - right and left tangential directions `R` and `L`.
    """
    if not discretization.cont_time:
        raise NotImplementedError
    assert r > 0
    assert sigma is None or len(sigma) == r
    assert b is None or isinstance(b, np.ndarray) and b.shape == (discretization.m, r)
    assert c is None or isinstance(c, np.ndarray) and c.shape == (discretization.p, r)
    assert conv_crit in ('rel_sigma_change', 'rel_H2_dist')

    # basic choice for initial interpolation points and tangential
    # directions
    if sigma is None:
        sigma = np.logspace(-1, 1, r)
    if b is None:
        b = np.ones((discretization.m, r))
    if c is None:
        c = np.ones((discretization.p, r))

    if verbose:
        print('iter | conv. criterion')
        print('-----+----------------')

    dist = []
    Sigma = [np.array(sigma)]
    R = [b]
    L = [c]
    # main loop
    for it in range(maxit):
        # interpolatory reduced order model
        rom = interpolation(discretization, sigma, b, c)

        # new interpolation points
        if isinstance(rom.E, IdentityOperator):
            sigma, Y, X = spla.eig(to_matrix(rom.A), left=True, right=True)
        else:
            sigma, Y, X = spla.eig(to_matrix(rom.A), to_matrix(rom.E), left=True, right=True)
        if force_sigma_in_rhp:
            sigma = np.array([np.abs(s.real) + s.imag * 1j for s in sigma])
        else:
            sigma *= -1
        Sigma.append(sigma)

        # compute convergence criterion
        if conv_crit == 'rel_sigma_change':
            dist.append(np.max(np.abs((Sigma[-2] - Sigma[-1]) / Sigma[-2])))
        elif conv_crit == 'rel_H2_dist':
            if it == 0:
                rom_new = rom
                dist.append(np.inf)
            else:
                rom_old = rom_new
                rom_new = rom
                rom_diff = rom_old - rom_new
                try:
                    rel_H2_dist = rom_diff.norm() / rom_old.norm()
                except:
                    rel_H2_dist = np.inf
                dist.append(rel_H2_dist)

        if verbose:
            print('{:4d} | {:15.9e}'.format(it + 1, dist[-1]))

        # new tangential directions
        b = rom.B._matrix.T.dot(Y.conj())
        c = rom.C._matrix.dot(X)
        R.append(b)
        L.append(c)

        # check if convergence criterion is satisfied
        if dist[-1] < tol:
            break

    # final reduced order model
    rom = interpolation(discretization, sigma, b, c)
    reduction_data = {'dist': dist, 'Sigma': Sigma, 'R': R, 'L': L}

    return rom, reduction_data
