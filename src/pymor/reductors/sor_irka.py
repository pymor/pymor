# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.interfaces import BasicInterface
from pymor.discretizations.iosys import SecondOrderSystem
from pymor.reductors.interpolation import SO_BHIReductor
from pymor.reductors.h2 import IRKAReductor, _poles_and_tangential_directions, _convergence_criterion


class SOR_IRKAReductor(BasicInterface):
    """SOR-IRKA reductor.

    Parameters
    ----------
    d
        SecondOrderSystem.
    """
    def __init__(self, d):
        assert isinstance(d, SecondOrderSystem)
        self.d = d

    def reduce(self, r, sigma=None, b=None, c=None, rd0=None, tol=1e-4, maxit=100, num_prev=1, force_sigma_in_rhp=False,
               projection='orth', use_arnoldi=False, conv_crit='sigma', compute_errors=False,
               irka_options=None):
        r"""Reduce using SOR-IRKA.

        It uses IRKA as the intermediate reductor, to reduce from 2r to
        r poles.
        See Section 5.3.2 in [W12]_.

        Parameters
        ----------
        r
            Order of the reduced order model.
        sigma
            Initial interpolation points (closed under conjugation).

            If `None`, interpolation points are log-spaced between 0.1
            and 10. If `sigma` is an `int`, it is used as a seed to
            generate it randomly. Otherwise, it needs to be a
            one-dimensional array-like of length `r`.

            `sigma` and `rd0` cannot both be not `None`.
        b
            Initial right tangential directions.

            If `None`, if is chosen as all ones. If `b` is an `int`, it
            is used as a seed to generate it randomly. Otherwise, it
            needs to be a |VectorArray| of length `r` from `d.B.source`.

            `b` and `rd0` cannot both be not `None`.
        c
            Initial left tangential directions.

            If `None`, if is chosen as all ones. If `c` is an `int`, it
            is used as a seed to generate it randomly. Otherwise, it
            needs to be a |VectorArray| of length `r` from `d.Cp.range`.

            `c` and `rd0` cannot both be not `None`.
        rd0
            Initial reduced order model.

            If `None`, then `sigma`, `b`, and `c` are used. Otherwise,
            it needs to be an |LTISystem| of order `r` and it is used to
            construct `sigma`, `b`, and `c`.
        tol
            Tolerance for the convergence criterion.
        maxit
            Maximum number of iterations.
        num_prev
            Number of previous iterations to compare the current
            iteration to. Larger number can avoid occasional cyclic
            behavior of IRKA.
        force_sigma_in_rhp
            If `False`, new interpolation are reflections of the current
            reduced order model's poles. Otherwise, only the poles in
            the left half-plane are reflected.
        projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogolized with
              respect to the E product
        conv_crit
            Convergence criterion:

            - `'sigma'`: relative change in interpolation points
            - `'h2'`: relative :math:`\mathcal{H}_2` distance of
              reduced-order models
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
        assert isinstance(num_prev, int) and num_prev >= 1
        assert projection in ('orth', 'biorth')
        assert conv_crit in ('sigma', 'h2')
        assert irka_options is None or isinstance(irka_options, dict)
        if not irka_options:
            irka_options = {}

        # initial interpolation points and tangential directions
        assert sigma is None or isinstance(sigma, int) or len(sigma) == r
        assert b is None or isinstance(b, int) or b in d.B.source and len(b) == r
        assert c is None or isinstance(c, int) or c in d.Cp.range and len(c) == r
        assert (rd0 is None or
                isinstance(rd0, SecondOrderSystem) and
                rd0.n == r and rd0.B.source == d.B.source and rd0.Cp.range == d.Cp.range)
        assert sigma is None or rd0 is None
        assert b is None or rd0 is None
        assert c is None or rd0 is None
        if rd0 is not None:
            with self.logger.block('Intermediate reduction ...'):
                irka_reductor = IRKAReductor(rd0.to_lti())
                rd_r = irka_reductor.reduce(r, **irka_options)
            poles, b, c = _poles_and_tangential_directions(rd_r)
            sigma = np.abs(poles.real) + poles.imag * 1j if force_sigma_in_rhp else -poles
        else:
            if sigma is None:
                sigma = np.logspace(-1, 1, r)
            elif isinstance(sigma, int):
                np.random.seed(sigma)
                sigma = np.abs(np.random.randn(r))
            if b is None:
                b = d.B.source.from_numpy(np.ones((r, d.m)))
            elif isinstance(b, int):
                np.random.seed(b)
                b = d.B.source.from_numpy(np.random.randn(r, d.m))
            if c is None:
                c = d.Cp.range.from_numpy(np.ones((r, d.p)))
            elif isinstance(c, int):
                np.random.seed(c)
                c = d.Cp.range.from_numpy(np.random.randn(r, d.p))

        # begin logging
        self.logger.info('Starting SOR-IRKA')
        if not compute_errors:
            self.logger.info('iter | conv. criterion')
            self.logger.info('-----+----------------')
        else:
            self.logger.info('iter | conv. criterion | rel. H_2-error')
            self.logger.info('-----+-----------------+----------------')

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

            # reduction to a system with r poles
            with self.logger.block('Intermediate reduction ...'):
                irka_reductor = IRKAReductor(rd.to_lti())
                rd_r = irka_reductor.reduce(r, **irka_options)

            # new interpolation points and tangential directions
            poles, b, c = _poles_and_tangential_directions(rd_r)
            sigma = np.abs(poles.real) + poles.imag * 1j if force_sigma_in_rhp else -poles
            self.sigmas.append(sigma)
            self.R.append(b)
            self.L.append(c)

            # compute convergence criterion
            if conv_crit == 'sigma':
                dist = _convergence_criterion(self.sigmas[:-num_prev-2:-1], conv_crit)
                self.dist.append(dist)
            elif conv_crit == 'h2':
                if it == 0:
                    rd_list = (num_prev + 1) * [None]
                    rd_list[0] = rd
                    self.dist.append(np.inf)
                else:
                    rd_list[1:] = rd_list[:-1]
                    rd_list[0] = rd
                    dist = _convergence_criterion(rd_list, conv_crit)
                    self.dist.append(dist)

            # report convergence
            if not compute_errors:
                self.logger.info('{:4d} | {:15.9e}'.format(it + 1, self.dist[-1]))
            else:
                if np.max(rd.poles().real) < 0:
                    err = d - rd
                    rel_H2_err = err.h2_norm() / d.h2_norm()
                else:
                    rel_H2_err = np.inf
                self.errors.append(rel_H2_err)

                self.logger.info('{:4d} | {:15.9e} | {:15.9e}'.format(it + 1, self.dist[-1], rel_H2_err))

            # check if convergence criterion is satisfied
            if self.dist[-1] < tol:
                break

        # final reduced order model
        rd = interp_reductor.reduce(sigma, b, c, projection=projection)
        self.V = interp_reductor.V
        self.W = interp_reductor.W

        return rd

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self.V[:u.dim].lincomb(u.to_numpy())
