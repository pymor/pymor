# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.sylvester import solve_sylv_schur
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.interfaces import BasicInterface
from pymor.discretizations.iosys import LTISystem
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.basic import GenericPGReductor
from pymor.reductors.interpolation import LTI_BHIReductor, TFInterpReductor


class IRKAReductor(BasicInterface):
    """Iterative Rational Krylov Algorithm reductor.

    Parameters
    ----------
    d
        |LTISystem|.
    """
    def __init__(self, d):
        assert isinstance(d, LTISystem)
        self.d = d

    def reduce(self, r, sigma=None, b=None, c=None, rd0=None, tol=1e-4, maxit=100, num_prev=1, force_sigma_in_rhp=False,
               projection='orth', use_arnoldi=False, conv_crit='sigma', compute_errors=False):
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
            needs to be a |VectorArray| of length `r` from `d.C.range`.

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
            reduced order model's poles. Otherwise, only poles in the
            left half-plane are reflected.
        projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogolized with
              respect to the E product
        use_arnoldi
            Should the Arnoldi process be used for rational
            interpolation. Available only for SISO systems. Otherwise,
            it is ignored.
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

        # initial interpolation points and tangential directions
        assert sigma is None or isinstance(sigma, int) or len(sigma) == r
        assert b is None or isinstance(b, int) or b in d.B.source and len(b) == r
        assert c is None or isinstance(c, int) or c in d.C.range and len(c) == r
        assert (rd0 is None or
                isinstance(rd0, LTISystem) and
                rd0.n == r and rd0.B.source == d.B.source and rd0.C.range == d.C.range)
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
            if b is None:
                b = d.B.source.from_numpy(np.ones((r, d.m)))
            elif isinstance(b, int):
                np.random.seed(b)
                b = d.B.source.from_numpy(np.random.randn(r, d.m))
            if c is None:
                c = d.C.range.from_numpy(np.ones((r, d.p)))
            elif isinstance(c, int):
                np.random.seed(c)
                c = d.C.range.from_numpy(np.random.randn(r, d.p))

        # being logging
        self.logger.info('Starting IRKA')
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
        interp_reductor = LTI_BHIReductor(d)
        # main loop
        for it in range(maxit):
            # interpolatory reduced order model
            rd = interp_reductor.reduce(sigma, b, c, projection=projection, use_arnoldi=use_arnoldi)

            # new interpolation points and tangential directions
            poles, b, c = _poles_and_tangential_directions(rd)
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
        rd = interp_reductor.reduce(sigma, b, c, projection=projection, use_arnoldi=use_arnoldi)
        self.V = interp_reductor.V
        self.W = interp_reductor.W

        return rd

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self.V[:u.dim].lincomb(u.to_numpy())


class TSIAReductor(BasicInterface):
    """Two-Sided Iteration Algorithm reductor.

    Parameters
    ----------
    d
        |LTISystem|.
    """
    def __init__(self, d):
        assert isinstance(d, LTISystem)
        self.d = d

    def reduce(self, rd0, tol=1e-4, maxit=100, num_prev=1, projection='orth', conv_crit='sigma',
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
        rd0
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
        rd
            Reduced |LTISystem|.
        """
        d = self.d
        assert isinstance(rd0, LTISystem) and rd0.B.source == d.B.source and rd0.C.range == d.C.range
        r = rd0.n
        assert 0 < r < d.n
        assert isinstance(num_prev, int) and num_prev >= 1
        assert projection in ('orth', 'biorth')
        assert conv_crit in ('sigma', 'h2')

        # begin logging
        self.logger.info('Starting TSIA')
        if not compute_errors:
            self.logger.info('iter | conv. criterion')
            self.logger.info('-----+----------------')
        else:
            self.logger.info('iter | conv. criterion | rel. H_2-error')
            self.logger.info('-----+-----------------+----------------')

        # find initial projection matrices
        self._projection_matrices(rd0, projection)

        data = (num_prev + 1) * [None]
        data[0] = rd0.poles() if conv_crit == 'sigma' else rd0
        self.dist = []
        self.errors = [] if compute_errors else None
        # main loop
        for it in range(maxit):
            # project the full order model
            rd = self.pg_reductor.reduce()

            # compute convergence criterion
            data[1:] = data[:-1]
            data[0] = rd.poles() if conv_crit == 'sigma' else rd
            dist = _convergence_criterion(data, conv_crit)
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

            # new projection matrices
            self._projection_matrices(rd, projection)

            # check convergence criterion
            if self.dist[-1] < tol:
                break

        # final reduced order model
        rd = self.pg_reductor.reduce()

        return rd

    def _projection_matrices(self, rd, projection):
        d = self.d
        self.V, self.W = solve_sylv_schur(d.A, rd.A,
                                          E=d.E, Er=rd.E,
                                          B=d.B, Br=rd.B,
                                          C=d.C, Cr=rd.C)
        if projection == 'orth':
            self.V = gram_schmidt(self.V, atol=0, rtol=0)
            self.W = gram_schmidt(self.W, atol=0, rtol=0)
        elif projection == 'biorth':
            self.V, self.W = gram_schmidt_biorth(self.V, self.W, product=d.E)

        self.pg_reductor = GenericPGReductor(d, self.W, self.V, projection == 'biorth', product=d.E)

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        self.pg_reductor.reconstruct(u)


class TF_IRKAReductor(BasicInterface):
    """Realization-independent IRKA reductor.

    See [BG12]_.

    Parameters
    ----------
    d
        Discretization with `eval_tf` and `eval_dtf` methods.
    """
    def __init__(self, d):
        self.d = d

    def reduce(self, r, sigma=None, b=None, c=None, rd0=None, tol=1e-4, maxit=100, num_prev=1, force_sigma_in_rhp=False,
               conv_crit='sigma'):
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

            `sigma` and `rd0` cannot both be not `None`.
        b
            Initial right tangential directions.

            If `None`, if is chosen as all ones. If `b` is an `int`, it
            is used as a seed to generate it randomly. Otherwise, it
            needs to be a |NumPy array| of shape `(m, r)`.

            `b` and `rd0` cannot both be not `None`.
        c
            Initial left tangential directions.

            If `None`, if is chosen as all ones. If `c` is an `int`, it
            is used as a seed to generate it randomly. Otherwise, it
            needs to be a |NumPy array| of shape `(p, r)`.

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
        rd
            Reduced |LTISystem| model.
        """
        d = self.d
        if not d.cont_time:
            raise NotImplementedError
        assert r > 0
        assert isinstance(num_prev, int) and num_prev >= 1
        assert conv_crit in ('sigma', 'h2')

        # initial interpolation points and tangential directions
        assert sigma is None or isinstance(sigma, int) or len(sigma) == r
        assert b is None or isinstance(b, int) or isinstance(b, np.ndarray) and b.shape == (d.m, r)
        assert c is None or isinstance(c, int) or isinstance(c, np.ndarray) and c.shape == (d.p, r)
        assert rd0 is None or rd0.n == r and rd0.B.source.dim == d.m and rd0.C.range.dim == d.p
        assert sigma is None or rd0 is None
        assert b is None or rd0 is None
        assert c is None or rd0 is None
        if rd0 is not None:
            poles, b, c = _poles_and_tangential_directions(rd0)
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
                b = np.ones((d.m, r))
            elif isinstance(b, int):
                np.random.seed(b)
                b = np.random.randn(d.m, r)
            if c is None:
                c = np.ones((d.p, r))
            elif isinstance(c, int):
                np.random.seed(c)
                c = np.random.randn(d.p, r)

        # begin logging
        self.logger.info('Starting TF-IRKA')
        self.logger.info('iter | conv. criterion')
        self.logger.info('-----+----------------')

        self.dist = []
        self.sigmas = [np.array(sigma)]
        self.R = [b]
        self.L = [c]
        interp_reductor = TFInterpReductor(d)
        # main loop
        for it in range(maxit):
            # interpolatory reduced order model
            rd = interp_reductor.reduce(sigma, b, c)

            # new interpolation points and tangential directions
            poles, b, c = _poles_and_tangential_directions(rd)
            b = b.to_numpy().T
            c = c.to_numpy().T
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
            self.logger.info('{:4d} | {:15.9e}'.format(it + 1, self.dist[-1]))

            # check if convergence criterion is satisfied
            if self.dist[-1] < tol:
                break

        # final reduced order model
        rd = interp_reductor.reduce(sigma, b, c)

        return rd


def _poles_and_tangential_directions(rd):
    """Compute the poles and tangential directions of a reduced order model."""
    if isinstance(rd.E, IdentityOperator):
        poles, Y, X = spla.eig(to_matrix(rd.A, format='dense'),
                               left=True, right=True)
    else:
        poles, Y, X = spla.eig(to_matrix(rd.A, format='dense'), to_matrix(rd.E, format='dense'),
                               left=True, right=True)
    Y = rd.B.range.make_array(Y.conj().T)
    X = rd.C.source.make_array(X.T)
    b = rd.B.apply_adjoint(Y)
    c = rd.C.apply(X)
    return poles, b, c


def _convergence_criterion(data, conv_crit):
    """Compute the convergence criterion for given data."""
    if conv_crit == 'sigma':
        sigma = data[0]
        dist_list = [spla.norm((sigma_old - sigma) / sigma_old, ord=np.inf)
                     for sigma_old in data[1:] if sigma_old is not None]
        return min(dist_list)
    elif conv_crit == 'h2':
        rd = data[0]
        if np.max(rd.poles().real) >= 0:
            return np.inf
        dist_list = [np.inf]
        for rd_old in data[1:]:
            if rd_old is not None and np.max(rd_old.poles().real) < 0:
                rd_diff = rd_old - rd
                dist_list.append(rd_diff.h2_norm() / rd_old.h2_norm())
        return min(dist_list)
