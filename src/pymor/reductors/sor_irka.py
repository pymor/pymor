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
from pymor.discretizations.iosys import SecondOrderSystem
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.basic import GenericPGReductor
from pymor.reductors.interpolation import SO_BHIReductor
from pymor.reductors.lti import IRKAReductor


class SOR_IRKAReductor(GenericPGReductor):
    """SOR-IRKA reductor.

    Parameters
    ----------
    d
        SecondOrderSystem.
    """
    def __init__(self, d):
        assert isinstance(d, SecondOrderSystem)
        self.d = d

    def reduce(self, r, sigma=None, b=None, c=None, tol=1e-4, maxit=100, dist_num=1, force_sigma_in_rhp=False,
               projection='orth', use_arnoldi=False, conv_crit='rel_sigma_change', compute_errors=False,
               irka_options=None):
        """Reduce using SOR-IRKA.

        It uses IRKA as the intermediate reductor, to reduce from 2r poles to r.

        .. [W12] S. Wyatt,
                 Issues in Interpolatory Model Reduction: Inexact Solves,
                 Second Order Systems and DAEs,
                 PhD thesis, Virginia Tech, 2012

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
                    with respect to the M product
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
        irka_options
            Dict of options for IRKAReductor.reduce.

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
        assert irka_options is None or isinstance(irka_options, dict)
        if not irka_options:
            irka_options = {}

        logger = getLogger('pymor.reductors.sor_irka.SOR_IRKAReductor.reduce')
        logger.info('Starting SOR-IRKA')

        # basic choice for initial interpolation points and tangential
        # directions
        if sigma is None:
            sigma = np.logspace(-1, 1, r)
        if b is None:
            b = d.B.source.from_data(np.ones((r, d.m)))
        if c is None:
            c = d.Cp.range.from_data(np.ones((r, d.p)))

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
        interp_reductor = SO_BHIReductor(d)
        # main loop
        for it in range(maxit):
            # interpolatory reduced order model
            rd = interp_reductor.reduce(sigma, b, c, projection=projection)

            if compute_errors:
                err = d - rd
                try:
                    rel_H2_err = err.norm() / d.norm()
                except:
                    rel_H2_err = np.inf
                self.errors.append(rel_H2_err)

            # reduction to a system with r poles
            with logger.block('Intermediate reduction ...'):
                irka_reductor = IRKAReductor(rd.to_lti())
                rd_r = irka_reductor.reduce(r, **irka_options)

            # new interpolation points
            if isinstance(rd_r.E, IdentityOperator):
                sigma, Y, X = spla.eig(to_matrix(rd_r.A, format='dense'), left=True, right=True)
            else:
                sigma, Y, X = spla.eig(to_matrix(rd_r.A, format='dense'), to_matrix(rd_r.E, format='dense'),
                                       left=True, right=True)
            if force_sigma_in_rhp:
                sigma = np.array([np.abs(s.real) + s.imag * 1j for s in sigma])
            else:
                sigma *= -1
            self.sigmas.append(sigma)

            # new tangential directions
            Y = rd_r.B.range.make_array(Y.conj().T)
            X = rd_r.C.source.make_array(X.T)
            b = rd_r.B.apply_transpose(Y).block(0)
            c = rd_r.C.apply(X).block(0)
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
        rd = interp_reductor.reduce(sigma, b, c, projection=projection)
        self.V = interp_reductor.V
        self.W = interp_reductor.W

        return rd

    extend_source_basis = None
    extend_range_basis = None
