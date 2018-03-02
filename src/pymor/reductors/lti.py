# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.sylvester import solve_sylv_schur
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.logger import getLogger
from pymor.discretizations.iosys import LTISystem
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.basic import GenericPGReductor
from pymor.reductors.interpolation import LTI_BHIReductor


class IRKAReductor(GenericPGReductor):
    """Iterative Rational Krylov Algorithm reductor.

    Parameters
    ----------
    d
        |LTISystem|.
    """
    def __init__(self, d):
        assert isinstance(d, LTISystem)
        self.d = d

    def reduce(self, r, sigma=None, b=None, c=None, tol=1e-4, maxit=100, dist_num=1, force_sigma_in_rhp=False,
               projection='orth', use_arnoldi=False, conv_crit='rel_sigma_change', compute_errors=False):
        r"""Reduce using IRKA.

        .. [GAB08] S. Gugercin, A. C. Antoulas, C. A. Beattie,
                :math:`\mathcal{H}_2` model reduction for large-scale
                linear dynamical systems,
                SIAM Journal on Matrix Analysis and Applications, 30(2),
                609-638, 2008.
        .. [ABG10] A. C. Antoulas, C. A. Beattie, S. Gugercin,
                Interpolatory model reduction of large-scale dynamical
                systems,
                Efficient Modeling and Control of Large-Scale Systems,
                Springer-Verlag, 2010.

        Parameters
        ----------
        r
            Order of the reduced order model.
        sigma
            Initial interpolation points (closed under conjugation),
            list of length `r`.

            If `None`, interpolation points are log-spaced between 0.1
            and 10.
        b
            Initial right tangential directions, |VectorArray| of length
            `r` from `d.B.source`.

            If `None`, `b` is chosen with all ones.
        c
            Initial left tangential directions, |VectorArray| of length
            `r` from `d.C.range`.

            If `None`, `c` is chosen with all ones.
        tol
            Tolerance for the largest change in interpolation points.
        maxit
            Maximum number of iterations.
        dist_num
            Number of past iterations to compare the current iteration.
            Larger number can avoid occasional cyclic behaviour of IRKA.
        force_sigma_in_rhp
            If 'False`, new interpolation are reflections of reduced
            order model's poles. Otherwise, they are always in the right
            half-plane.
        projection
            Projection method:

                - `'orth'`: projection matrices are orthogonalized with
                    respect to the Euclidean inner product
                - `'biorth'`: projection matrices are biorthogolized
                    with respect to the E product
        use_arnoldi
            Should the Arnoldi process be used for rational
            interpolation. Available only for SISO systems. Otherwise,
            it is ignored.
        conv_crit
            Convergence criterion:

                - `'rel_sigma_change'`: relative change in interpolation
                  points
                - `'subspace_sin'`: maximum of sines of Petrov-Galerkin
                  subspaces
                - `'rel_H2_dist'`: relative :math:`\mathcal{H}_2`
                  distance of reduced order models
        compute_errors
            Should the relative :math:`\mathcal{H}_2`-errors of
            intermediate reduced order models be computed.

            .. warning::
                Computing :math:`\mathcal{H}_2`-errors is expensive. Use
                this option only if necessary.

        Returns
        -------
        rd
            Reduced |LTISystem| model.
        """
        d = self.d
        if not d.cont_time:
            raise NotImplementedError
        assert 0 < r < d.n
        assert sigma is None or len(sigma) == r
        assert b is None or b in d.B.source and len(b) == r
        assert c is None or c in d.C.range and len(c) == r
        assert dist_num >= 1
        assert projection in ('orth', 'biorth')
        assert conv_crit in ('rel_sigma_change', 'subspace_sin', 'rel_H2_dist')

        logger = getLogger('pymor.reductors.lti.IRKAReductor.reduce')
        logger.info('Starting IRKA')

        # basic choice for initial interpolation points and tangential
        # directions
        if sigma is None:
            sigma = np.logspace(-1, 1, r)
        if b is None:
            b = d.B.source.from_numpy(np.ones((r, d.m)))
        if c is None:
            c = d.C.range.from_numpy(np.ones((r, d.p)))

        if compute_errors:
            logger.info('iter | conv. criterion | rel. H_2-error')
            logger.info('-----+-----------------+----------------')
        else:
            logger.info('iter | conv. criterion')
            logger.info('-----+----------------')

        self.dist = []
        self.sigmas = [np.array(sigma)]
        self.R = [b]
        self.L = [c]
        self.errors = [] if compute_errors else None
        interp_reductor = LTI_BHIReductor(d)
        # main loop
        for it in range(maxit):
            # interpolatory reduced order model
            rd = interp_reductor.reduce(sigma, b, c, projection=projection, use_arnoldi=use_arnoldi)

            if compute_errors:
                err = d - rd
                try:
                    rel_H2_err = err.norm() / d.norm()
                except:
                    rel_H2_err = np.inf
                self.errors.append(rel_H2_err)

            # new interpolation points
            if isinstance(rd.E, IdentityOperator):
                sigma, Y, X = spla.eig(to_matrix(rd.A, format='dense'), left=True, right=True)
            else:
                sigma, Y, X = spla.eig(to_matrix(rd.A, format='dense'), to_matrix(rd.E, format='dense'),
                                       left=True, right=True)
            if force_sigma_in_rhp:
                sigma = np.array([np.abs(s.real) + s.imag * 1j for s in sigma])
            else:
                sigma *= -1
            self.sigmas.append(sigma)

            # new tangential directions
            Y = rd.B.range.make_array(Y.conj().T)
            X = rd.C.source.make_array(X.T)
            b = rd.B.apply_transpose(Y)
            c = rd.C.apply(X)
            self.R.append(b)
            self.L.append(c)

            # compute convergence criterion
            if conv_crit == 'rel_sigma_change':
                dist = spla.norm((self.sigmas[-2] - self.sigmas[-1]) / self.sigmas[-2], ord=np.inf)
                for i in range(2, min(dist_num + 1, len(self.sigmas))):
                    dist2 = spla.norm((self.sigmas[-i - 1] - self.sigmas[-1]) / self.sigmas[-i - 1], ord=np.inf)
                    dist = min(dist, dist2)
                self.dist.append(dist)
            elif conv_crit == 'subspace_sin':
                if it == 0:
                    V_list = (dist_num + 1) * [None]
                    W_list = (dist_num + 1) * [None]
                    V_list[0] = interp_reductor.V
                    W_list[0] = interp_reductor.W
                    self.dist.append(1)
                else:
                    for i in range(1, dist_num + 1):
                        V_list[-i] = V_list[-i - 1]
                        W_list[-i] = W_list[-i - 1]
                    V_list[0] = interp_reductor.V
                    W_list[0] = interp_reductor.W
                    # TODO: replace with SVD when it becomes possible
                    sinV = np.sqrt(np.max(spla.eigvalsh((V_list[0] -
                                                         V_list[1].lincomb(V_list[0].inner(V_list[1]))).gramian())))
                    sinW = np.sqrt(np.max(spla.eigvalsh((W_list[0] -
                                                         W_list[1].lincomb(W_list[0].inner(W_list[1]))).gramian())))
                    dist = max(sinV, sinW)
                    for i in range(2, dist_num + 1):
                        if V_list[i] is None:
                            break
                        sinV = np.sqrt(np.max(spla.eigvalsh((V_list[0] -
                                                             V_list[i].lincomb(V_list[0].inner(V_list[i]))).gramian())))
                        sinW = np.sqrt(np.max(spla.eigvalsh((W_list[0] -
                                                             W_list[i].lincomb(W_list[0].inner(W_list[i]))).gramian())))
                        dist = min(dist, max(sinV, sinW))
                    self.dist.append(dist)
            elif conv_crit == 'rel_H2_dist':
                if it == 0:
                    rd_list = (dist_num + 1) * [None]
                    rd_list[0] = rd
                    self.dist.append(np.inf)
                else:
                    for i in range(1, dist_num + 1):
                        rd_list[-i] = rd_list[-i - 1]
                    rd_list[0] = rd
                    rd_diff = rd_list[1] - rd_list[0]
                    try:
                        rel_H2_dist = rd_diff.norm() / rd_list[1].norm()
                    except:
                        rel_H2_dist = np.inf
                    for i in range(2, dist_num + 1):
                        if rd_list[i] is None:
                            break
                        rd_diff2 = rd_list[i] - rd_list[0]
                        try:
                            rel_H2_dist2 = rd_diff2.norm() / rd_list[i].norm()
                        except:
                            rel_H2_dist2 = np.inf
                        rel_H2_dist = min(rel_H2_dist, rel_H2_dist2)
                    self.dist.append(rel_H2_dist)

            if compute_errors:
                logger.info('{:4d} | {:15.9e} | {:15.9e}'.format(it + 1, self.dist[-1], rel_H2_err))
            else:
                logger.info('{:4d} | {:15.9e}'.format(it + 1, self.dist[-1]))

            # check if convergence criterion is satisfied
            if self.dist[-1] < tol:
                break

        # final reduced order model
        rd = interp_reductor.reduce(sigma, b, c, projection=projection, use_arnoldi=use_arnoldi)
        self.V = interp_reductor.V
        self.W = interp_reductor.W

        return rd

    extend_source_basis = None
    extend_range_basis = None


class TSIAReductor(GenericPGReductor):
    """Two-Sided Iteration Algorithm reductor.

    Parameters
    ----------
    d
        |LTISystem|.
    """
    def __init__(self, d):
        assert isinstance(d, LTISystem)
        self.d = d

    def reduce(self, rd0, tol=1e-4, maxit=100, dist_num=1, projection='orth', conv_crit='rel_sigma_change',
               compute_errors=False):
        """Reduce using TSIA.

        In exact arithmetic, TSIA is equivalent to IRKA (under some
        assumptions on the poles of the reduced model). The main
        difference in implementation is that TSIA computes the Schur
        decomposition of the reduced matrices, while IRKA computes the
        eigenvalue decomposition. Therefore, TSIA might behave better
        for non-normal reduced matrices.

        .. [BKS11] P. Benner, M. Köhler, J. Saak, Sparse-Dense Sylvester
                   Equations in :math:`\mathcal{H}_2`-Model Order
                   Reduction,
                   Max Planck Institute Magdeburg Preprint, available
                   from http://www.mpi-magdeburg.mpg.de/preprints/,
                   2011.

        Parameters
        ----------
        rd0
            Initial reduced order model.
        tol
            Tolerance for the convergence criterion.
        maxit
            Maximum number of iterations.
        dist_num
            Number of past iterations to compare the current iteration.
            Larger number can avoid occasional cyclic behaviour of TSIA.
        projection
            Projection method:

                - `'orth'`: projection matrices are orthogonalized with
                    respect to the Euclidean inner product
                - `'biorth'`: projection matrices are biorthogolized
                    with respect to the E product
        conv_crit
            Convergence criterion:

                - `'rel_sigma_change'`: relative change in interpolation
                  points
                - `'subspace_sin'`: maximum of sines of Petrov-Galerkin
                  subspaces
                - `'rel_H2_dist'`: relative :math:`\mathcal{H}_2`
                  distance of reduced order models
        compute_errors
            Should the relative :math:`\mathcal{H}_2`-errors of
            intermediate reduced order models be computed.

            .. warning::
                Computing :math:`\mathcal{H}_2`-errors is expensive. Use
                this option only if necessary.

        Returns
        -------
        rd
            Reduced |LTISystem|.
        """
        d = self.d
        r = rd0.n
        assert 0 < r < d.n
        assert dist_num >= 1
        assert projection in ('orth', 'biorth')
        assert conv_crit in ('rel_sigma_change', 'subspace_sin', 'rel_H2_dist')

        logger = getLogger('pymor.reductors.lti.TSIAReductor.reduce')
        logger.info('Starting TSIA')

        if compute_errors:
            logger.info('iter | conv. criterion | rel. H_2-error')
            logger.info('-----+-----------------+----------------')
        else:
            logger.info('iter | conv. criterion')
            logger.info('-----+----------------')

        # find initial projection matrices
        self.V, self.W = solve_sylv_schur(d.A, rd0.A,
                                          E=d.E, Er=rd0.E,
                                          B=d.B, Br=rd0.B,
                                          C=d.C, Cr=rd0.C)
        if projection == 'orth':
            self.V = gram_schmidt(self.V, atol=0, rtol=0)
            self.W = gram_schmidt(self.W, atol=0, rtol=0)
            self.biorthogonal_product = None
        elif projection == 'biorth':
            self.V, self.W = gram_schmidt_biorth(self.V, self.W, product=d.E)
            self.biorthogonal_product = 'E'

        if conv_crit == 'rel_sigma_change':
            sigma_list = (dist_num + 1) * [None]
            sigma_list[0] = rd0.poles(force_dense=True)
        self.dist = []
        self.errors = [] if compute_errors else None
        # main loop
        for it in range(maxit):
            # project the full order model
            rd = super().reduce()

            if compute_errors:
                err = d - rd
                try:
                    rel_H2_err = err.norm() / d.norm()
                except:
                    rel_H2_err = np.inf
                self.errors.append(rel_H2_err)

            # compute convergence criterion
            if conv_crit == 'rel_sigma_change':
                for i in range(1, dist_num + 1):
                    sigma_list[-i] = sigma_list[-i - 1]
                sigma_list[0] = rd.poles(force_dense=True)
                try:
                    dist = spla.norm((sigma_list[1] - sigma_list[0]) / sigma_list[1], ord=np.inf)
                except:
                    dist = np.inf
                for i in range(2, dist_num + 1):
                    if sigma_list[i] is None:
                        break
                    try:
                        dist2 = spla.norm((sigma_list[i] - sigma_list[0]) / sigma_list[i], ord=np.inf)
                    except:
                        dist2 = np.inf
                    dist = min(dist, dist2)
                self.dist.append(dist)
            elif conv_crit == 'subspace_sin':
                if it == 0:
                    V_list = (dist_num + 1) * [None]
                    W_list = (dist_num + 1) * [None]
                    V_list[0] = self.V
                    W_list[0] = self.W
                    self.dist.append(1)
                else:
                    for i in range(1, dist_num + 1):
                        V_list[-i] = V_list[-i - 1]
                        W_list[-i] = W_list[-i - 1]
                    V_list[0] = self.V
                    W_list[0] = self.W
                    # TODO: replace with SVD when it becomes possible
                    sinV = np.sqrt(np.max(spla.eigvalsh((V_list[0] -
                                                         V_list[1].lincomb(V_list[0].inner(V_list[1]))).gramian())))
                    sinW = np.sqrt(np.max(spla.eigvalsh((W_list[0] -
                                                         W_list[1].lincomb(W_list[0].inner(W_list[1]))).gramian())))
                    dist = max(sinV, sinW)
                    for i in range(2, dist_num + 1):
                        if V_list[i] is None:
                            break
                        sinV = np.sqrt(np.max(spla.eigvalsh((V_list[0] -
                                                             V_list[i].lincomb(V_list[0].inner(V_list[i]))).gramian())))
                        sinW = np.sqrt(np.max(spla.eigvalsh((W_list[0] -
                                                             W_list[i].lincomb(W_list[0].inner(W_list[i]))).gramian())))
                        dist = min(dist, max(sinV, sinW))
                    self.dist.append(dist)
            elif conv_crit == 'rel_H2_dist':
                if it == 0:
                    rd_list = (dist_num + 1) * [None]
                    rd_list[0] = rd
                    self.dist.append(np.inf)
                else:
                    for i in range(1, dist_num + 1):
                        rd_list[-i] = rd_list[-i - 1]
                    rd_list[0] = rd
                    rd_diff = rd_list[1] - rd_list[0]
                    try:
                        rel_H2_dist = rd_diff.norm() / rd_list[1].norm()
                    except:
                        rel_H2_dist = np.inf
                    for i in range(2, dist_num + 1):
                        if rd_list[i] is None:
                            break
                        rd_diff2 = rd_list[i] - rd_list[0]
                        try:
                            rel_H2_dist2 = rd_diff2.norm() / rd_list[i].norm()
                        except:
                            rel_H2_dist2 = np.inf
                        rel_H2_dist = min(rel_H2_dist, rel_H2_dist2)
                    self.dist.append(rel_H2_dist)

            if compute_errors:
                logger.info('{:4d} | {:15.9e} | {:15.9e}'.format(it + 1, self.dist[-1], rel_H2_err))
            else:
                logger.info('{:4d} | {:15.9e}'.format(it + 1, self.dist[-1]))

            # new projection matrices
            self.V, self.W = solve_sylv_schur(d.A, rd.A,
                                              E=d.E, Er=rd.E,
                                              B=d.B, Br=rd.B,
                                              C=d.C, Cr=rd.C)
            if projection == 'orth':
                self.V = gram_schmidt(self.V, atol=0, rtol=0)
                self.W = gram_schmidt(self.W, atol=0, rtol=0)
            elif projection == 'biorth':
                self.V, self.W = gram_schmidt_biorth(self.V, self.W, product=d.E)

            # check convergence criterion
            if self.dist[-1] < tol:
                break

        # final reduced order model
        rd = super().reduce()

        return rd

    extend_source_basis = None
    extend_range_basis = None
