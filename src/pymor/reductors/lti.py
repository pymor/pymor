# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.arnoldi import arnoldi
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.sylvester import solve_sylv_schur
from pymor.algorithms.to_matrix import to_matrix
from pymor.operators.constructions import IdentityOperator, LincombOperator
from pymor.reductors.basic import reduce_generic_pg


def interpolation(discretization, sigma, b, c, method='orth', use_arnoldi=False):
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
    method
        Method of projection the discretization:

        - `'orth'`: projection matrices are orthogonalized with respect
            to the Euclidean inner product
        - `'biorth'`: projection matrices are biorthogolized with
            respect to the E product
    use_arnoldi
        Should the Arnoldi process be used for rational interpolation.
        Available only for SISO systems. Otherwise, it is ignored.

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
    assert method in ('orth', 'biorth')

    if use_arnoldi and discretization.m == 1 and discretization.p == 1:
        V = arnoldi(discretization.A, discretization.E, discretization.B, sigma)
        W = arnoldi(discretization.A, discretization.E, discretization.C, sigma, trans=True)
        rom, rc, _ = reduce_generic_pg(discretization, V, W)
    else:
        # rescale tangential directions (could avoid overflow or underflow)
        b.scal(1 / b.l2_norm())
        c.scal(1 / c.l2_norm())

        # compute projection matrices
        V = discretization.A.source.empty(reserve=r)
        W = discretization.A.source.empty(reserve=r)
        for i in range(r):
            if sigma[i].imag == 0:
                sEmA = LincombOperator((discretization.E, discretization.A), (sigma[i].real, -1))

                Bb = discretization.B.apply(b.real[i])
                V.append(sEmA.apply_inverse(Bb))

                CTc = discretization.C.apply_transpose(c.real[i])
                W.append(sEmA.apply_inverse_transpose(CTc))
            elif sigma[i].imag > 0:
                sEmA = LincombOperator((discretization.E, discretization.A), (sigma[i], -1))

                Bb = discretization.B.apply(b[i])
                v = sEmA.apply_inverse(Bb)
                V.append(v.real)
                V.append(v.imag)

                CTc = discretization.C.apply_transpose(c[i])
                w = sEmA.apply_inverse_transpose(CTc)
                W.append(w.real)
                W.append(w.imag)

        if method == 'orth':
            V = gram_schmidt(V, atol=0, rtol=0)
            W = gram_schmidt(W, atol=0, rtol=0)
            rom, rc, _ = reduce_generic_pg(discretization, V, W)
        elif method == 'biorth':
            V, W = gram_schmidt_biorth(V, W, product=discretization.E)
            rom, rc, _ = reduce_generic_pg(discretization, V, W, use_default=['E'])

    reduction_data = {'V': V, 'W': W}

    return rom, rc, reduction_data


def irka(discretization, r, sigma=None, b=None, c=None, tol=1e-4, maxit=100, verbose=False, force_sigma_in_rhp=False,
         method='orth', use_arnoldi=False, conv_crit='rel_sigma_change', compute_errors=False):
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
        If 'False`, new interpolation are reflections of reduced order
        model's poles. Otherwise, they are always in the right
        half-plane.
    method
        Method of projection the discretization (see
        :func:`pymor.reductors.lti.interpolation`_).
    use_arnoldi
        Should the Arnoldi process be used for rational interpolation.
        Available only for SISO systems. Otherwise, it is ignored.
    conv_crit
        Convergence criterion:
            - `'rel_sigma_change'`: relative change in interpolation points
            - `'subspace_sin'`: maximum of sines of Petrov-Galerkin subspaces
            - `'rel_H2_dist'`: relative H_2 distance of reduced order models
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
    if not discretization.cont_time:
        raise NotImplementedError
    assert 0 < r < discretization.n
    assert sigma is None or len(sigma) == r
    assert b is None or b in discretization.B.source and len(b) == r
    assert c is None or c in discretization.C.range and len(c) == r
    assert method in ('orth', 'biorth')
    assert conv_crit in ('rel_sigma_change', 'subspace_sin', 'rel_H2_dist')

    # basic choice for initial interpolation points and tangential
    # directions
    if sigma is None:
        sigma = np.logspace(-1, 1, r)
    if b is None:
        # for the full order model we cannot assume that the source of B
        # is a NumpyVectorSpace, so we have to use 'from_data' here
        b = discretization.B.source.from_data(np.ones((r, discretization.m)))
    if c is None:
        # for the full order model we cannot assume that the range of C
        # is a NumpyVectorSpace, so we have to use 'from_data' here
        c = discretization.C.range.from_data(np.ones((r, discretization.p)))

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
    # main loop
    for it in range(maxit):
        # interpolatory reduced order model
        rom, rc, reduction_data = interpolation(discretization, sigma, b, c, method=method, use_arnoldi=use_arnoldi)

        if compute_errors:
            err = discretization - rom
            try:
                rel_H2_err = err.norm() / discretization.norm()
            except:
                rel_H2_err = np.inf
            errors.append(rel_H2_err)

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
            dist.append(spla.norm((Sigma[-2] - Sigma[-1]) / Sigma[-2], ord=np.inf))
        elif conv_crit == 'subspace_sin':
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
            if compute_errors:
                print('{:4d} | {:15.9e} | {:15.9e}'.format(it + 1, dist[-1], rel_H2_err))
            else:
                print('{:4d} | {:15.9e}'.format(it + 1, dist[-1]))

        # new tangential directions
        Y = rom.B.range.make_array(Y.conj().T)
        X = rom.C.source.make_array(X.T)
        b = rom.B.apply_transpose(Y)
        c = rom.C.apply(X)
        R.append(b)
        L.append(c)

        # check if convergence criterion is satisfied
        if dist[-1] < tol:
            break

    # final reduced order model
    rom, rc, reduction_data = interpolation(discretization, sigma, b, c, method=method, use_arnoldi=use_arnoldi)

    reduction_data.update({'dist': dist, 'Sigma': Sigma, 'R': R, 'L': L})
    if compute_errors:
        reduction_data['errors'] = errors

    return rom, rc, reduction_data


def tsia(discretization, rom0, tol=1e-4, maxit=100, verbose=False, method='orth', conv_crit='rel_sigma',
         compute_errors=False):
    """Reduce using TSIA (Two Sided Iteration Algorithm).

    In exact arithmetic, TSIA is equivalent to IRKA (under some
    assumptions on the poles of the reduced model). The main difference
    in implementation is that TSIA computes the Schur decomposition of
    the reduced matrices, while IRKA computes the eigenvalue
    decomposition. Therefore, TSIA might behave better for non-normal
    reduced matrices.

    .. [BKS11]  P. Benner, M. Köhler, J. Saak,
                Sparse-Dense Sylvester Equations in :math:`\mathcal{H}_2`-Model Order Reduction,
                Max Planck Institute Magdeburg Preprint, available from http://www.mpi-magdeburg.mpg.de/preprints/,
                2011.

    Parameters
    ----------
    discretization
        The |LTISystem| which is to be reduced.
    rom0
        Initial reduced order model.
    tol
        Tolerance for the convergence criterion.
    maxit
        Maximum number of iterations.
    verbose
        Should convergence criterion during iterations be printed.
    method
        Method of projection the discretization:

        - `'orth'`: projection matrices are orthogonalized with respect
            to the Euclidean inner product
        - `'biorth'`: projection matrices are biorthogolized with
            respect to the E product
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
        - convergence criterion in iterations `dist`,
        - relative :math:`\mathcal{H}_2`-errors `errors` (if
          `compute_errors` is `True`).
    """
    r = rom0.n
    assert 0 < r < discretization.n
    assert conv_crit in ('rel_sigma', 'max_sin_PG', 'rel_H2')

    if verbose:
        if compute_errors:
            print('iter | conv. criterion | rel. H_2-error')
            print('-----+-----------------+----------------')
        else:
            print('iter | conv. criterion')
            print('-----+----------------')

    V, W = solve_sylv_schur(discretization.A, rom0.A,
                            E=discretization.E, Er=rom0.E,
                            B=discretization.B, Br=rom0.B,
                            C=discretization.C, Cr=rom0.C)
    if method == 'orth':
        V = gram_schmidt(V, atol=0, rtol=0)
        W = gram_schmidt(W, atol=0, rtol=0)
    elif method == 'biorth':
        V, W = gram_schmidt_biorth(V, W, product=discretization.E)

    if conv_crit == 'rel_sigma':
        sigma = rom0.poles()
    dist = []
    if compute_errors:
        errors = []
    for it in range(maxit):
        if method == 'orth':
            rom, rc, _ = reduce_generic_pg(discretization, V, W)
        elif method == 'biorth':
            rom, rc, _ = reduce_generic_pg(discretization, V, W, use_default=['E'])

        if compute_errors:
            err = discretization - rom
            try:
                rel_H2_err = err.norm() / discretization.norm()
            except:
                rel_H2_err = np.inf
            errors.append(rel_H2_err)

        if conv_crit == 'rel_sigma':
            sigma_old, sigma = sigma, rom.poles()
            try:
                dist.append(spla.norm((sigma_old - sigma) / sigma_old, ord=np.inf))
            except:
                dist.append(np.nan)
        elif conv_crit == 'max_sin_PG':
            if it == 0:
                V_new = V
                W_new = W
                dist.append(1)
            if it > 0:
                V_old, V_new = V_new, V
                W_old, W_new = W_new, W
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

        V, W = solve_sylv_schur(discretization.A, rom.A,
                                E=discretization.E, Er=rom.E,
                                B=discretization.B, Br=rom.B,
                                C=discretization.C, Cr=rom.C)
        if method == 'orth':
            V = gram_schmidt(V, atol=0, rtol=0)
            W = gram_schmidt(W, atol=0, rtol=0)
        elif method == 'biorth':
            V, W = gram_schmidt_biorth(V, W, product=discretization.E)

        if dist[-1] < tol:
            break

    if method == 'orth':
        rom, rc, _ = reduce_generic_pg(discretization, V, W)
    elif method == 'biorth':
        rom, rc, _ = reduce_generic_pg(discretization, V, W, use_default=['E'])

    reduction_data = {'V': V, 'W': W, 'dist': dist}
    if compute_errors:
        reduction_data['errors'] = errors

    return rom, rc, reduction_data
