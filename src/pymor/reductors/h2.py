# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.sylvester import solve_sylv_schur
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.interfaces import BasicInterface
from pymor.models.iosys import LTIModel
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.basic import LTIPGReductor
from pymor.reductors.interpolation import LTIBHIReductor, TFBHIReductor


class IRKAReductor(BasicInterface):
    """Iterative Rational Krylov Algorithm reductor.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    mu
        |Parameter|.
    """
    def __init__(self, fom, mu=None):
        assert isinstance(fom, LTIModel)
        self.fom = fom
        self.mu = fom.parse_parameter(mu)
        self.V = None
        self.W = None
        self._pg_reductor = None
        self.conv_crit = None
        self.sigmas = None
        self.R = None
        self.L = None
        self.errors = None

    def reduce(self, r, sigma=None, b=None, c=None, rom0=None, tol=1e-4, maxit=100, num_prev=1,
               force_sigma_in_rhp=False, projection='orth', conv_crit='sigma', compute_errors=False):
        r"""Reduce using IRKA.

        See [GAB08]_ (Algorithm 4.1) and [ABG10]_ (Algorithm 1).

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

            `sigma` and `rom0` cannot both be not `None`.
        b
            Initial right tangential directions.

            If `None`, if is chosen as all ones. If `b` is an `int`, it
            is used as a seed to generate it randomly. Otherwise, it
            needs to be a |VectorArray| of length `r` from `fom.B.source`.

            `b` and `rom0` cannot both be not `None`.
        c
            Initial left tangential directions.

            If `None`, if is chosen as all ones. If `c` is an `int`, it
            is used as a seed to generate it randomly. Otherwise, it
            needs to be a |VectorArray| of length `r` from `fom.C.range`.

            `c` and `rom0` cannot both be not `None`.
        rom0
            Initial reduced order model.

            If `None`, then `sigma`, `b`, and `c` are used. Otherwise,
            it needs to be an |LTIModel| of order `r` and it is used to
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
            reduced order model's poles. Otherwise, only poles in the
            left half-plane are reflected.
        projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogolized with
              respect to the E product
            - `'arnoldi'`: projection matrices are orthogonalized using
              the Arnoldi process (available only for SISO systems).
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
        assert projection in ('orth', 'biorth', 'arnoldi')
        if projection == 'arnoldi':
            assert fom.input_dim == fom.output_dim == 1
        assert conv_crit in ('sigma', 'h2')

        # initial interpolation points and tangential directions
        assert sigma is None or isinstance(sigma, int) or len(sigma) == r
        assert b is None or isinstance(b, int) or b in fom.B.source and len(b) == r
        assert c is None or isinstance(c, int) or c in fom.C.range and len(c) == r
        assert (rom0 is None
                or isinstance(rom0, LTIModel)
                and rom0.order == r and rom0.B.source == fom.B.source and rom0.C.range == fom.C.range)
        assert sigma is None or rom0 is None
        assert b is None or rom0 is None
        assert c is None or rom0 is None
        if rom0 is not None:
            poles, b, c = _poles_and_tangential_directions(rom0)
            sigma = np.abs(poles.real) + poles.imag * 1j if force_sigma_in_rhp else -poles
        else:
            if sigma is None:
                sigma = np.logspace(-1, 1, r)
            elif isinstance(sigma, int):
                np.random.seed(sigma)
                sigma = np.abs(np.random.randn(r))
            if b is None:
                b = fom.B.source.ones(r)
            elif isinstance(b, int):
                b = fom.B.source.random(r, distribution='normal', seed=b)
            if c is None:
                c = fom.C.range.ones(r)
            elif isinstance(c, int):
                c = fom.C.range.random(r, distribution='normal', seed=c)

        self.logger.info('Starting IRKA')
        self.conv_crit = []
        self.sigmas = [np.array(sigma)]
        self.R = [b]
        self.L = [c]
        self.errors = [] if compute_errors else None
        self._pg_reductor = LTIBHIReductor(fom, mu=self.mu)
        # main loop
        for it in range(maxit):
            # interpolatory reduced order model
            rom = self._pg_reductor.reduce(sigma, b, c, projection=projection)

            # new interpolation points and tangential directions
            poles, b, c = _poles_and_tangential_directions(rom)
            sigma = np.abs(poles.real) + poles.imag * 1j if force_sigma_in_rhp else -poles
            self.sigmas.append(sigma)
            self.R.append(b)
            self.L.append(c)

            # compute convergence criterion
            if conv_crit == 'sigma':
                dist = _convergence_criterion(self.sigmas[:-num_prev-2:-1], conv_crit)
                self.conv_crit.append(dist)
            elif conv_crit == 'h2':
                if it == 0:
                    rom_list = (num_prev + 1) * [None]
                    rom_list[0] = rom
                    self.conv_crit.append(np.inf)
                else:
                    rom_list[1:] = rom_list[:-1]
                    rom_list[0] = rom
                    dist = _convergence_criterion(rom_list, conv_crit)
                    self.conv_crit.append(dist)

            # report convergence
            self.logger.info(f'Convergence criterion in iteration {it + 1}: {self.conv_crit[-1]:e}')
            if compute_errors:
                if np.max(rom.poles().real) < 0:
                    err = fom - rom
                    rel_H2_err = err.h2_norm() / fom.h2_norm()
                else:
                    rel_H2_err = np.inf
                self.errors.append(rel_H2_err)

                self.logger.info(f'Relative H2-error in iteration {it + 1}: {rel_H2_err:e}')

            # check if convergence criterion is satisfied
            if self.conv_crit[-1] < tol:
                break

        # final reduced order model
        rom = self._pg_reductor.reduce(sigma, b, c, projection=projection)
        self.V = self._pg_reductor.V
        self.W = self._pg_reductor.W
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)


class OneSidedIRKAReductor(BasicInterface):
    """One-Sided Iterative Rational Krylov Algorithm reductor.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    version
        Version of the one-sided IRKA:

        - `'V'`: Galerkin projection using the input Krylov subspace,
        - `'W'`: Galerkin projection using the output Krylov subspace.
    mu
        |Parameter|.
    """
    def __init__(self, fom, version, mu=None):
        assert isinstance(fom, LTIModel)
        assert version in ('V', 'W')
        self.fom = fom
        self.version = version
        self.mu = fom.parse_parameter(mu)
        self.V = None
        self._pg_reductor = None
        self.conv_crit = None
        self.sigmas = None
        self.R = None
        self.L = None
        self.errors = None

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
                    b = fom.B.source.ones(r)
                elif isinstance(b, int):
                    b = fom.B.source.random(r, distribution='normal', seed=b)
            else:
                if c is None:
                    c = fom.C.range.ones(r)
                elif isinstance(c, int):
                    c = fom.C.range.random(r, distribution='normal', seed=c)

        self.logger.info('Starting one-sided IRKA')
        self.conv_crit = []
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
            rom = self._pg_reductor.reduce()

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
                self.conv_crit.append(dist)
            elif conv_crit == 'h2':
                if it == 0:
                    rom_list = (num_prev + 1) * [None]
                    rom_list[0] = rom
                    self.conv_crit.append(np.inf)
                else:
                    rom_list[1:] = rom_list[:-1]
                    rom_list[0] = rom
                    dist = _convergence_criterion(rom_list, conv_crit)
                    self.conv_crit.append(dist)

            # report convergence
            self.logger.info(f'Convergence criterion in iteration {it + 1}: {self.conv_crit[-1]:e}')
            if compute_errors:
                if np.max(rom.poles().real) < 0:
                    err = fom - rom
                    rel_H2_err = err.h2_norm() / fom.h2_norm()
                else:
                    rel_H2_err = np.inf
                self.errors.append(rel_H2_err)

                self.logger.info(f'Relative H2-error in iteration {it + 1}: {rel_H2_err:e}')

            # check if convergence criterion is satisfied
            if self.conv_crit[-1] < tol:
                break

        # final reduced order model
        self._projection_matrix(r, sigma, b, c, projection)
        rom = self._pg_reductor.reduce()
        return rom

    def _projection_matrix(self, r, sigma, b, c, projection):
        if self.fom.parametric:
            fom = self.fom.with_(**{op: getattr(self.fom, op).assemble(mu=self.mu)
                                    for op in ['A', 'B', 'C', 'D', 'E']},
                                 parameter_space=None)
        else:
            fom = self.fom
        if self.version == 'V':
            V = fom.A.source.empty(reserve=r)
        else:
            W = fom.A.source.empty(reserve=r)
        for i in range(r):
            if sigma[i].imag == 0:
                sEmA = sigma[i].real * fom.E - fom.A
                if self.version == 'V':
                    Bb = fom.B.apply(b.real[i])
                    V.append(sEmA.apply_inverse(Bb))
                else:
                    CTc = fom.C.apply_adjoint(c.real[i])
                    W.append(sEmA.apply_inverse_adjoint(CTc))
            elif sigma[i].imag > 0:
                sEmA = sigma[i] * fom.E - fom.A
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
        self._pg_reductor = LTIPGReductor(fom, self.V, self.V, projection == 'Eorth')

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)


class TSIAReductor(BasicInterface):
    """Two-Sided Iteration Algorithm reductor.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    mu
        |Parameter|.
    """
    def __init__(self, fom, mu=None):
        assert isinstance(fom, LTIModel)
        self.fom = fom
        self.mu = fom.parse_parameter(mu)
        self.V = None
        self.W = None
        self._pg_reductor = None
        self.conv_crit = None
        self.errors = None

    def reduce(self, rom0, tol=1e-4, maxit=100, num_prev=1, projection='orth', conv_crit='sigma',
               compute_errors=False):
        r"""Reduce using TSIA.

        See [XZ11]_ (Algorithm 1) and [BKS11]_.

        In exact arithmetic, TSIA is equivalent to IRKA (under some
        assumptions on the poles of the reduced model). The main
        difference in implementation is that TSIA computes the Schur
        decomposition of the reduced matrices, while IRKA computes the
        eigenvalue decomposition. Therefore, TSIA might behave better
        for non-normal reduced matrices.

        Parameters
        ----------
        rom0
            Initial reduced order model.
        tol
            Tolerance for the convergence criterion.
        maxit
            Maximum number of iterations.
        num_prev
            Number of previous iterations to compare the current
            iteration to. Larger number can avoid occasional cyclic
            behavior of TSIA.
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

        Returns
        -------
        rom
            Reduced |LTIModel|.
        """
        fom = self.fom
        assert isinstance(rom0, LTIModel) and rom0.B.source == fom.B.source and rom0.C.range == fom.C.range
        r = rom0.order
        assert 0 < r < fom.order
        assert isinstance(num_prev, int) and num_prev >= 1
        assert projection in ('orth', 'biorth')
        assert conv_crit in ('sigma', 'h2')

        # begin logging
        self.logger.info('Starting TSIA')

        # find initial projection matrices
        self._projection_matrices(rom0, projection)

        data = (num_prev + 1) * [None]
        data[0] = rom0.poles() if conv_crit == 'sigma' else rom0
        self.conv_crit = []
        self.errors = [] if compute_errors else None
        # main loop
        for it in range(maxit):
            # project the full order model
            rom = self._pg_reductor.reduce()

            # compute convergence criterion
            data[1:] = data[:-1]
            data[0] = rom.poles() if conv_crit == 'sigma' else rom
            dist = _convergence_criterion(data, conv_crit)
            self.conv_crit.append(dist)

            # report convergence
            self.logger.info(f'Convergence criterion in iteration {it + 1}: {self.conv_crit[-1]:e}')
            if compute_errors:
                if np.max(rom.poles().real) < 0:
                    err = fom - rom
                    rel_H2_err = err.h2_norm() / fom.h2_norm()
                else:
                    rel_H2_err = np.inf
                self.errors.append(rel_H2_err)

                self.logger.info(f'Relative H2-error in iteration {it + 1}: {rel_H2_err:e}')

            # new projection matrices
            self._projection_matrices(rom, projection)

            # check convergence criterion
            if self.conv_crit[-1] < tol:
                break

        # final reduced order model
        rom = self._pg_reductor.reduce()
        return rom

    def _projection_matrices(self, rom, projection):
        if self.fom.parametric:
            fom = self.fom.with_(**{op: getattr(self.fom, op).assemble(mu=self.mu)
                                    for op in ['A', 'B', 'C', 'D', 'E']},
                                 parameter_space=None)
        else:
            fom = self.fom

        self.V, self.W = solve_sylv_schur(fom.A, rom.A,
                                          E=fom.E, Er=rom.E,
                                          B=fom.B, Br=rom.B,
                                          C=fom.C, Cr=rom.C)

        if projection == 'orth':
            self.V = gram_schmidt(self.V, atol=0, rtol=0)
            self.W = gram_schmidt(self.W, atol=0, rtol=0)
        elif projection == 'biorth':
            self.V, self.W = gram_schmidt_biorth(self.V, self.W, product=fom.E)

        self._pg_reductor = LTIPGReductor(fom, self.W, self.V, projection == 'biorth')

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)


class TFIRKAReductor(BasicInterface):
    """Realization-independent IRKA reductor.

    See [BG12]_.

    Parameters
    ----------
    fom
        The full-order |Model| with `eval_tf` and `eval_dtf` methods.
    mu
        |Parameter|.
    """
    def __init__(self, fom, mu=None):
        self.fom = fom
        self.mu = fom.parse_parameter(mu)
        self.conv_crit = None
        self.sigmas = None
        self.R = None
        self.L = None

    def reduce(self, r, sigma=None, b=None, c=None, rom0=None, tol=1e-4, maxit=100, num_prev=1,
               force_sigma_in_rhp=False, conv_crit='sigma'):
        r"""Reduce using TF-IRKA.

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

            `sigma` and `rom0` cannot both be not `None`.
        b
            Initial right tangential directions.

            If `None`, if is chosen as all ones. If `b` is an `int`, it
            is used as a seed to generate it randomly. Otherwise, it
            needs to be a |NumPy array| of shape `(m, r)`.

            `b` and `rom0` cannot both be not `None`.
        c
            Initial left tangential directions.

            If `None`, if is chosen as all ones. If `c` is an `int`, it
            is used as a seed to generate it randomly. Otherwise, it
            needs to be a |NumPy array| of shape `(p, r)`.

            `c` and `rom0` cannot both be not `None`.
        rom0
            Initial reduced order model.

            If `None`, then `sigma`, `b`, and `c` are used. Otherwise,
            it needs to be an |LTIModel| of order `r` and it is used to
            construct `sigma`, `b`, and `c`.
        tol
            Tolerance for the convergence criterion.
        maxit
            Maximum number of iterations.
        num_prev
            Number of previous iterations to compare the current
            iteration to. Larger number can avoid occasional cyclic
            behavior of TF-IRKA.
        force_sigma_in_rhp
            If `False`, new interpolation are reflections of the current
            reduced order model's poles. Otherwise, only the poles in
            the left half-plane are reflected.
        conv_crit
            Convergence criterion:

            - `'sigma'`: relative change in interpolation points
            - `'h2'`: relative :math:`\mathcal{H}_2` distance of
              reduced-order models

        Returns
        -------
        rom
            Reduced |LTIModel| model.
        """
        fom = self.fom
        if not fom.cont_time:
            raise NotImplementedError
        assert r > 0
        assert isinstance(num_prev, int) and num_prev >= 1
        assert conv_crit in ('sigma', 'h2')

        # initial interpolation points and tangential directions
        assert sigma is None or isinstance(sigma, int) or len(sigma) == r
        assert b is None or isinstance(b, int) or isinstance(b, np.ndarray) and b.shape == (fom.input_dim, r)
        assert c is None or isinstance(c, int) or isinstance(c, np.ndarray) and c.shape == (fom.output_dim, r)
        assert rom0 is None or rom0.order == r and rom0.input_dim == fom.input_dim and rom0.output_dim == fom.output_dim
        assert sigma is None or rom0 is None
        assert b is None or rom0 is None
        assert c is None or rom0 is None
        if rom0 is not None:
            poles, b, c = _poles_and_tangential_directions(rom0)
            b = b.to_numpy().T
            c = c.to_numpy().T
            sigma = np.abs(poles.real) + poles.imag * 1j if force_sigma_in_rhp else -poles
        else:
            if sigma is None:
                sigma = np.logspace(-1, 1, r)
            elif isinstance(sigma, int):
                np.random.seed(sigma)
                sigma = np.abs(np.random.randn(r))
            if b is None:
                b = np.ones((fom.input_dim, r))
            elif isinstance(b, int):
                np.random.seed(b)
                b = np.random.randn(fom.input_dim, r)
            if c is None:
                c = np.ones((fom.output_dim, r))
            elif isinstance(c, int):
                np.random.seed(c)
                c = np.random.randn(fom.output_dim, r)

        self.logger.info('Starting TF-IRKA')
        self.conv_crit = []
        self.sigmas = [np.array(sigma)]
        self.R = [b]
        self.L = [c]
        interp_reductor = TFBHIReductor(fom, mu=self.mu)
        # main loop
        for it in range(maxit):
            # interpolatory reduced order model
            rom = interp_reductor.reduce(sigma, b, c)

            # new interpolation points and tangential directions
            poles, b, c = _poles_and_tangential_directions(rom)
            b = b.to_numpy().T
            c = c.to_numpy().T
            sigma = np.abs(poles.real) + poles.imag * 1j if force_sigma_in_rhp else -poles
            self.sigmas.append(sigma)
            self.R.append(b)
            self.L.append(c)

            # compute convergence criterion
            if conv_crit == 'sigma':
                dist = _convergence_criterion(self.sigmas[:-num_prev-2:-1], conv_crit)
                self.conv_crit.append(dist)
            elif conv_crit == 'h2':
                if it == 0:
                    rom_list = (num_prev + 1) * [None]
                    rom_list[0] = rom
                    self.conv_crit.append(np.inf)
                else:
                    rom_list[1:] = rom_list[:-1]
                    rom_list[0] = rom
                    dist = _convergence_criterion(rom_list, conv_crit)
                    self.conv_crit.append(dist)

            # report convergence
            self.logger.info(f'Convergence criterion in iteration {it + 1}: {self.conv_crit[-1]:e}')

            # check if convergence criterion is satisfied
            if self.conv_crit[-1] < tol:
                break

        # final reduced order model
        rom = interp_reductor.reduce(sigma, b, c)
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        raise TypeError(f'The reconstruct method is not available for {self.__class__.__name__}.')


def _poles_and_tangential_directions(rom):
    """Compute the poles and tangential directions of a reduced order model."""
    if isinstance(rom.E, IdentityOperator):
        poles, Y, X = spla.eig(to_matrix(rom.A, format='dense'),
                               left=True, right=True)
    else:
        poles, Y, X = spla.eig(to_matrix(rom.A, format='dense'), to_matrix(rom.E, format='dense'),
                               left=True, right=True)
    Y = rom.B.range.make_array(Y.conj().T)
    X = rom.C.source.make_array(X.T)
    b = rom.B.apply_adjoint(Y)
    c = rom.C.apply(X)
    return poles, b, c


def _convergence_criterion(data, conv_crit):
    """Compute the convergence criterion for given data."""
    if conv_crit == 'sigma':
        sigma = data[0]
        dist_list = [spla.norm((sigma_old - sigma) / sigma_old, ord=np.inf)
                     for sigma_old in data[1:] if sigma_old is not None]
        return min(dist_list)
    elif conv_crit == 'h2':
        rom = data[0]
        if np.max(rom.poles().real) >= 0:
            return np.inf
        dist_list = [np.inf]
        for rom_old in data[1:]:
            if rom_old is not None and np.max(rom_old.poles().real) < 0:
                rom_diff = rom_old - rom
                dist_list.append(rom_diff.h2_norm() / rom_old.h2_norm())
        return min(dist_list)
