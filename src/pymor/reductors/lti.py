# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.arnoldi import arnoldi, arnoldi_tangential
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.operators.constructions import LincombOperator
from pymor.reductors.basic import reduce_generic_pg


def interpolation(discretization, sigma, b, c, use_arnoldi=False):
    """Tangential Hermite interpolation.

    Parameters
    ----------
    discretization
        |LTISystem|.
    sigma
        Interpolation points (closed under conjugation), list of length `r`.
    b
        Right tangential directions, |VectorArray| of length `r` from
        `discretization.B.source`.
    c
        Left tangential directions, |VectorArray| of length `r` from
        `discretization.C.range`.
    use_arnoldi
        Should the Arnoldi process be used for rational interpolation.

    Returns
    -------
    rom
        Reduced |LTISystem| model.
    rc
        Reconstructor of full state.
    reduction_data
        Dictionary of additional data produced by the reduction process.
        Contains projection matrices `V` and `W`,
    """
    r = len(sigma)
    assert b in discretization.B.source and len(b) == r
    assert c in discretization.C.range and len(c) == r

    if use_arnoldi:
        if discretization.m == 1:
            V = arnoldi(discretization.A, discretization.E, discretization.B, sigma)
        else:
            V = arnoldi_tangential(discretization.A, discretization.E, discretization.B, sigma, b)
        if discretization.p == 1:
            W = arnoldi(discretization.A, discretization.E, discretization.C, sigma, trans=True)
        else:
            W = arnoldi_tangential(discretization.A, discretization.E, discretization.C, sigma, c, trans=True)
    else:
        b.scal(1 / b.l2_norm())
        c.scal(1 / c.l2_norm())

        V = discretization.A.source.empty(reserve=r)
        W = discretization.A.source.empty(reserve=r)

        for i in range(r):
            if sigma[i].imag == 0:
                sEmA = LincombOperator((discretization.E, discretization.A), (sigma[i].real, -1))

                Bb = discretization.B.apply(b.real[i])
                V.append(sEmA.apply_inverse(Bb))

                CTc = discretization.C.apply_adjoint(c.real[i])
                W.append(sEmA.apply_inverse_adjoint(CTc))
            elif sigma[i].imag > 0:
                sEmA = LincombOperator((discretization.E, discretization.A), (sigma[i], -1))

                Bb = discretization.B.apply(b[i])
                v = sEmA.apply_inverse(Bb)
                V.append(v.real)
                V.append(v.imag)

                CTc = discretization.C.apply_adjoint(c[i])
                w = sEmA.apply_inverse_adjoint(CTc)
                W.append(w.real)
                W.append(w.imag)

        V = gram_schmidt(V, atol=0, rtol=0)
        W = gram_schmidt(W, atol=0, rtol=0)

    rom, rc, _ = reduce_generic_pg(discretization, V, W)
    reduction_data = {'V': V, 'W': W}

    return rom, rc, reduction_data


def irka(discretization, r, sigma=None, b=None, c=None, tol=1e-4, maxit=100, verbose=False, force_sigma_in_rhp=True,
         use_arnoldi=False, conv_crit='rel_sigma', compute_errors=False):
    r"""Reduce using IRKA.

    .. [GAB08] S. Gugercin, A. C. Antoulas, C. A. Beattie,
               :math:`\mathcal{H}_2` model reduction for large-scale linear
               dynamical systems,
               SIAM Journal on Matrix Analysis and Applications, 30(2),
               609-638, 2008.
    .. [ABG10] A. C. Antoulas, C. A. Beattie, S. Gugercin, Interpolatory
               model reduction of large-scale dynamical systems,
               Efficient Modeling and Control of Large-Scale Systems,
               Springer-Verlag, 2010.

    Parameters
    ----------
    discretization
        The |LTISystem| which is to be reduced.
    r
        Order of the reduced order model.
    sigma
        Initial interpolation points (closed under conjugation), list of
        length `r`.

        If `None`, interpolation points are log-spaced between 0.1 and 10.
    b
        Initial right tangential directions, |VectorArray| of length `r`
        from `discretization.B.source`.

        If `None`, `b` is chosen with all ones.
    c
        Initial left tangential directions, |VectorArray| of length `r` from
        `discretization.C.range`.

        If `None`, `c` is chosen with all ones.
    tol
        Tolerance for the largest change in interpolation points.
    maxit
        Maximum number of iterations.
    verbose
        Should consecutive distances be printed.
    force_sigma_in_rhp
        If `True`, new interpolation points are always in the right
        half-plane. Otherwise, they are reflections of reduced order model's
        poles.
    use_arnoldi
        Should the Arnoldi process be used for rational interpolation.
    conv_crit
        Convergence criterion:
            - `'rel_sigma'`: relative change in interpolation points
            - `'max_sin_PG'`: maximum of sines in Petrov-Galerkin subspaces
            - `'rel_H2'`: relative H_2 distance of reduced order models
    compute_errors
        Should the relative :math:`\mathcal{H}_2`-errors of intermediate
        reduced order models be computed.

        .. warning::
            Computing :math:`\mathcal{H}_2`-errors is expensive. Use this
            option only if necessary.

    Returns
    -------
    rom
        Reduced |LTISystem| model.
    rc
        Reconstructor of full state.
    reduction_data
        Dictionary of additional data produced by the reduction process.
        Contains:

        - projection matrices `V` and `W`,
        - distances between interpolation points in subsequent iterations
          `dist`,
        - interpolation points from all iterations `Sigma`,
        - right and left tangential directions `R` and `L`, and
        - relative :math:`\mathcal{H}_2`-errors `errors` (if
          `compute_errors` is `True`).
    """
    assert 0 < r < discretization.n
    assert sigma is None or len(sigma) == r
    assert b is None or b in discretization.B.source and len(b) == r
    assert c is None or c in discretization.C.range and len(c) == r
    assert conv_crit in ('rel_sigma', 'max_sin_PG', 'rel_H2')

    if sigma is None:
        sigma = np.logspace(-1, 1, r)
    if b is None:
        b = discretization.B.source.make_array(np.ones((r, discretization.m)))
    if c is None:
        c = discretization.C.range.make_array(np.ones((r, discretization.p)))

    if verbose:
        if compute_errors:
            print('iter | conv. criterion | rel. H_2-error')
            print('-----+-----------------+----------------')
        else:
            print('iter | conv. criterion')
            print('-----+----------------')

    dist = []
    Sigma = [np.array(sigma)]
    R = [b]
    L = [c]
    if compute_errors:
        errors = []
    for it in range(maxit):
        rom, rc, reduction_data = interpolation(discretization, sigma, b, c, use_arnoldi=use_arnoldi)

        if compute_errors:
            err = discretization - rom
            try:
                rel_H2_err = err.norm() / discretization.norm()
            except:
                rel_H2_err = np.inf
            errors.append(rel_H2_err)

        sigma, Y, X = spla.eig(rom.A._matrix, rom.E._matrix, left=True, right=True)
        if force_sigma_in_rhp:
            sigma = np.array([np.abs(s.real) + s.imag * 1j for s in sigma])
        else:
            sigma *= -1
        Sigma.append(sigma)

        if conv_crit == 'rel_sigma':
            dist.append(spla.norm((Sigma[-2] - Sigma[-1]) / Sigma[-2], ord=np.inf))
        elif conv_crit == 'max_sin_PG':
            if it == 0:
                V_new = reduction_data['V'].data.T
                W_new = reduction_data['W'].data.T
                dist.append(1)
            else:
                V_old = V_new
                W_old = W_new
                V_new = reduction_data['V'].data.T
                W_new = reduction_data['W'].data.T
                sinV = spla.norm(V_new - V_old.dot(V_old.T.dot(V_new)), ord=2)
                sinW = spla.norm(W_new - W_old.dot(W_old.T.dot(W_new)), ord=2)
                dist.append(np.max([sinV, sinW]))
        elif conv_crit == 'rel_H2':
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
            if compute_errors:
                print('{:4d} | {:15.9e} | {:15.9e}'.format(it + 1, dist[-1], rel_H2_err))
            else:
                print('{:4d} | {:15.9e}'.format(it + 1, dist[-1]))

        Y = rom.B.range.make_array(Y.conj().T)
        X = rom.C.source.make_array(X.T)
        b = rom.B.apply_adjoint(Y)
        c = rom.C.apply(X)
        R.append(b)
        L.append(c)

        if dist[-1] < tol:
            break

    rom, rc, reduction_data = interpolation(discretization, sigma, b, c, use_arnoldi=use_arnoldi)

    reduction_data.update({'dist': dist, 'Sigma': Sigma, 'R': R, 'L': L})
    if compute_errors:
        reduction_data['errors'] = errors

    return rom, rc, reduction_data
