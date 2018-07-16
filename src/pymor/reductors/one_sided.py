# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.interfaces import BasicInterface
from pymor.models.iosys import LTIModel
from pymor.reductors.basic import GenericPGReductor
from pymor.reductors.h2 import _poles_and_tangential_directions, _convergence_criterion


class OneSidedIRKAReductor(BasicInterface):
    """One-Sided Iterative Rational Krylov Algorithm reductor.

    Parameters
    ----------
    fom
        |LTIModel|.
    version
        Version of the one-sided IRKA:

        - `'V'`: Galerkin projection using the input Krylov subspace,
        - `'W'`: Galerkin projection using the output Krylov subspace.
    """
    def __init__(self, fom, version):
        assert isinstance(fom, LTIModel)
        assert version in ('V', 'W')
        self.fom = fom
        self.version = version

    def reduce(self, r, sigma=None, b=None, c=None, rd0=None, tol=1e-4, maxit=100, num_prev=1,
               force_sigma_in_rhp=False, projection='orth', conv_crit='sigma',
               compute_errors=False):
        r"""Reduce using one-sided IRKA.

        Parameters
        ----------
        r
            Order of the reduced order model.
        sigma
            Initial interpolation points (closed under conjugation).

            If `None`, interpolation points are log-spaced between 0.1 and 10.
            If `sigma` is an `int`, it is used as a seed to generate it randomly.
            Otherwise, it needs to be a one-dimensional array-like of length `r`.

            `sigma` and `rd0` cannot both be not `None`.
        b
            Initial right tangential directions.

            If `None`, if is chosen as all ones.
            If `b` is an `int`, it is used as a seed to generate it randomly.
            Otherwise, it needs to be a |VectorArray| of length `r` from `fom.B.source`.

            `b` and `rd0` cannot both be not `None`.
        c
            Initial left tangential directions.

            If `None`, if is chosen as all ones.
            If `c` is an `int`, it is used as a seed to generate it randomly.
            Otherwise, it needs to be a |VectorArray| of length `r` from `fom.C.range`.

            `c` and `rd0` cannot both be not `None`.
        rd0
            Initial reduced order model.

            If `None`, then `sigma`, `b`, and `c` are used.
            Otherwise, it needs to be an |LTIModel| of order `r` and it is used to construct
            `sigma`, `b`, and `c`.
        tol
            Tolerance for the largest change in interpolation points.
        maxit
            Maximum number of iterations.
        num_prev
            Number of previous iterations to compare the current iteration to.
            A larger number can avoid occasional cyclic behavior.
        force_sigma_in_rhp
            If 'False`, new interpolation are reflections of reduced order model's poles.
            Otherwise, they are always in the right half-plane.
        projection
            Projection method:

            - `'orth'`: projection matrix is orthogonalized with respect to the Euclidean inner
              product,
            - `'Eorth'`: projection matrix is orthogonalized with respect to the E product.
        conv_crit
            Convergence criterion:

            - `'sigma'`: relative change in interpolation points,
            - `'h2'`: relative :math:`\mathcal{H}_2` distance of reduced order models.
        compute_errors
            Should the relative :math:`\mathcal{H}_2`-errors of intermediate reduced order models be
            computed.

            .. warning::
                Computing :math:`\mathcal{H}_2`-errors is expensive.
                Use this option only if necessary.

        Returns
        -------
        rom
            Reduced |LTIModel| model.
        """
        fom = self.fom
        if not fom.cont_time:
            raise NotImplementedError
        assert 0 < r < fom.order
        assert isinstance(num_prev, int) and num_prev >= 1
        assert projection in ('orth', 'Eorth')
        assert conv_crit in ('sigma', 'h2')

        # initial interpolation points and tangential directions
        assert sigma is None or isinstance(sigma, int) or len(sigma) == r
        assert b is None or isinstance(b, int) or b in fom.B.source and len(b) == r
        assert c is None or isinstance(c, int) or c in fom.C.range and len(c) == r
        assert (rd0 is None
                or isinstance(rd0, LTIModel)
                and rd0.order == r and rd0.input_space == fom.input_space and rd0.output_space == fom.output_space)
        assert sigma is None or rd0 is None
        assert b is None or rd0 is None
        assert c is None or rd0 is None
        if rd0 is not None:
            poles, b, c = _poles_and_tangential_directions(rd0)
            sigma = np.abs(poles.real) + poles.imag * 1j if force_sigma_in_rhp else -poles
        else:
            if sigma is None:
                sigma = np.logspace(-1, 1, r)
            elif isinstance(sigma, int):
                np.random.seed(sigma)
                sigma = np.abs(np.random.randn(r))
            if self.version == 'V':
                if b is None:
                    b = fom.B.source.from_numpy(np.ones((r, fom.input_dim)))
                elif isinstance(b, int):
                    np.random.seed(b)
                    b = fom.B.source.from_numpy(np.random.randn(r, fom.input_dim))
            else:
                if c is None:
                    c = fom.C.range.from_numpy(np.ones((r, fom.output_dim)))
                elif isinstance(c, int):
                    np.random.seed(c)
                    c = fom.C.range.from_numpy(np.random.randn(r, fom.output_dim))

        # begin logging
        self.logger.info('Starting one-sided IRKA')
        if not compute_errors:
            self.logger.info('iter | conv. criterion')
            self.logger.info('-----+----------------')
        else:
            self.logger.info('iter | conv. criterion | rel. H_2-error')
            self.logger.info('-----+-----------------+----------------')

        self.dist = []
        self.sigmas = [np.array(sigma)]
        if self.version == 'V':
            self.R = [b]
        else:
            self.L = [c]
        self.errors = [] if compute_errors else None
        # main loop
        for it in range(maxit):
            # interpolatory reduced order model
            self._projection_matrix(r, sigma, b, c, projection)
            rom = self.pg_reductor.reduce()

            # new interpolation points and tangential directions
            poles, b, c = _poles_and_tangential_directions(rom)
            sigma = np.abs(poles.real) + poles.imag * 1j if force_sigma_in_rhp else -poles
            self.sigmas.append(sigma)
            if self.version == 'V':
                self.R.append(b)
            else:
                self.L.append(c)

            # compute convergence criterion
            if conv_crit == 'sigma':
                dist = _convergence_criterion(self.sigmas[:-num_prev-2:-1], conv_crit)
                self.dist.append(dist)
            elif conv_crit == 'h2':
                if it == 0:
                    rom_list = (num_prev + 1) * [None]
                    rom_list[0] = rom
                    self.dist.append(np.inf)
                else:
                    rom_list[1:] = rom_list[:-1]
                    rom_list[0] = rom
                    dist = _convergence_criterion(rom_list, conv_crit)
                    self.dist.append(dist)

            # report convergence
            if not compute_errors:
                self.logger.info(f'{it+1:4d} | {self.dist[-1]:15.9e}')
            else:
                if np.max(rom.poles().real) < 0:
                    err = fom - rom
                    rel_H2_err = err.h2_norm() / fom.h2_norm()
                else:
                    rel_H2_err = np.inf
                self.errors.append(rel_H2_err)

                self.logger.info(f'{it+1:4d} | {self.dist[-1]:15.9e} | {rel_H2_err:15.9e}')

            # check if convergence criterion is satisfied
            if self.dist[-1] < tol:
                break

        # final reduced order model
        self._projection_matrix(r, sigma, b, c, projection)
        rom = self.pg_reductor.reduce()

        return rom

    def _projection_matrix(self, r, sigma, b, c, projection):
        fom = self.fom
        if self.version == 'V':
            V = fom.A.source.empty(reserve=r)
        else:
            W = fom.A.source.empty(reserve=r)
        for i in range(r):
            if sigma[i].imag == 0:
                sEmA = sigma[i].real * self.fom.E - self.fom.A
                if self.version == 'V':
                    Bb = fom.B.apply(b.real[i])
                    V.append(sEmA.apply_inverse(Bb))
                else:
                    CTc = fom.C.apply_adjoint(c.real[i])
                    W.append(sEmA.apply_inverse_adjoint(CTc))
            elif sigma[i].imag > 0:
                sEmA = sigma[i] * self.fom.E - self.fom.A
                if self.version == 'V':
                    Bb = fom.B.apply(b[i])
                    v = sEmA.apply_inverse(Bb)
                    V.append(v.real)
                    V.append(v.imag)
                else:
                    CTc = fom.C.apply_adjoint(c[i].conj())
                    w = sEmA.apply_inverse_adjoint(CTc)
                    W.append(w.real)
                    W.append(w.imag)

        if self.version == 'V':
            self.V = gram_schmidt(V, atol=0, rtol=0, product=None if projection == 'orth' else fom.E)
        else:
            self.V = gram_schmidt(W, atol=0, rtol=0, product=None if projection == 'orth' else fom.E)

        self.pg_reductor = GenericPGReductor(fom, self.V, self.V, projection == 'Eorth', product=fom.E)

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self.V[:u.dim].lincomb(u.to_numpy())
