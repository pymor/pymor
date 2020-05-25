# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Reductors based on H2-norm."""

from numbers import Integral, Real

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.krylov import tangential_rational_krylov
from pymor.algorithms.sylvester import solve_sylv_schur
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.base import BasicObject
from pymor.models.iosys import InputOutputModel, LTIModel
from pymor.operators.constructions import IdentityOperator
from pymor.parameters.base import Mu
from pymor.reductors.basic import LTIPGReductor
from pymor.reductors.interpolation import LTIBHIReductor, TFBHIReductor


class GenericIRKAReductor(BasicObject):
    """Generic IRKA related reductor.

    Parameters
    ----------
    fom
        The full-order |Model| to reduce.
    mu
        |Parameter values|.
    """

    def _clear_lists(self):
        self.sigma_list = []
        self.b_list = []
        self.c_list = []
        self.conv_crit = []
        self._conv_data = []
        self.errors = []

    def __init__(self, fom, mu=None):
        if not isinstance(mu, Mu):
            mu = fom.parameters.parse(mu)
        assert fom.parameters.assert_compatible(mu)
        self.fom = fom
        self.mu = mu
        self.V = None
        self.W = None
        self._pg_reductor = None
        self._clear_lists()

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)

    def _check_rom0_params(self, rom0_params):
        if isinstance(rom0_params, Integral):
            assert rom0_params > 0
            if hasattr(self.fom, 'order'):  # self.fom can be a TransferFunction
                assert rom0_params < self.fom.order
        elif isinstance(rom0_params, np.ndarray):
            assert rom0_params.ndim == 1
        elif isinstance(rom0_params, dict):
            assert ('sigma', 'b', 'c') in rom0_params
            assert isinstance(rom0_params['sigma'], np.ndarray)
            assert rom0_params['sigma'].ndim == 1
            assert rom0_params['b'] in self.fom.input_space
            assert rom0_params['c'] in self.fom.output_space
            assert len(rom0_params['sigma']) == len(rom0_params['b'])
            assert len(rom0_params['sigma']) == len(rom0_params['c'])
        elif isinstance(rom0_params, LTIModel):
            assert rom0_params.order > 0
            if hasattr(self.fom, 'order'):  # self.fom can be a TransferFunction
                assert rom0_params < self.fom.order
            assert rom0_params.input_space == self.fom.input_space
            assert rom0_params.output_space == self.fom.output_space
        else:
            raise ValueError(f'rom0_params is of wrong type ({type(rom0_params)}).')

    @staticmethod
    def _check_common_args(tol, maxit, num_prev, conv_crit):
        assert isinstance(tol, Real) and tol > 0
        assert isinstance(maxit, Integral) and maxit >= 1
        assert isinstance(num_prev, Integral) and num_prev >= 1
        assert conv_crit in ('sigma', 'h2')

    def _order_to_sigma_b_c(self, r):
        sigma = np.logspace(-1, 1, r)
        b = (self.fom.input_space.ones(r)
             if self.fom.input_dim == 1
             else self.fom.input_space.random(r, distribution='normal', seed=0))
        c = (self.fom.output_space.ones(r)
             if self.fom.output_dim == 1
             else self.fom.output_space.random(r, distribution='normal', seed=0))
        return sigma, b, c

    @staticmethod
    def _rom_to_sigma_b_c(rom, force_sigma_in_rhp):
        poles, b, c = _lti_to_poles_b_c(rom)
        sigma = (np.abs(poles.real) + poles.imag * 1j
                 if force_sigma_in_rhp
                 else -poles)
        return sigma, b, c

    def _rom0_params_to_sigma_b_c(self, rom0_params, force_sigma_in_rhp):
        self.logger.info('Generating initial interpolation data')
        self._check_rom0_params(rom0_params)
        if isinstance(rom0_params, Integral):
            sigma, b, c = self._order_to_sigma_b_c(rom0_params)
        elif isinstance(rom0_params, np.ndarray):
            sigma = rom0_params
            _, b, c = self._order_to_sigma_b_c(len(rom0_params))
        elif isinstance(rom0_params, dict):
            sigma = rom0_params['sigma']
            b = rom0_params['b']
            c = rom0_params['c']
        else:
            sigma, b, c = self._rom_to_sigma_b_c(rom0_params, force_sigma_in_rhp)
        return sigma, b, c

    def _rom0_params_to_rom(self, rom0_params):
        self.logger.info('Generating initial reduced-order model')
        self._check_rom0_params(rom0_params)
        if isinstance(rom0_params, Integral):
            sigma, b, c = self._order_to_sigma_b_c(rom0_params)
            rom0 = _poles_b_c_to_lti(-sigma, b, c)
        elif isinstance(rom0_params, np.ndarray):
            sigma = rom0_params
            _, b, c = self._order_to_sigma_b_c(len(rom0_params))
            rom0 = _poles_b_c_to_lti(-sigma, b, c)
        elif isinstance(rom0_params, dict):
            sigma = rom0_params['sigma']
            b = rom0_params['b']
            c = rom0_params['c']
            rom0 = _poles_b_c_to_lti(-sigma, b, c)
        else:
            rom0 = rom0_params
        return rom0

    def _store_sigma_b_c(self, sigma, b, c):
        if sigma is not None:
            self.sigma_list.append(sigma)
        if b is not None:
            self.b_list.append(b)
        if c is not None:
            self.c_list.append(c)

    def _update_conv_data(self, sigma, rom, conv_crit):
        del self._conv_data[-1]
        self._conv_data.insert(0, sigma if conv_crit == 'sigma' else rom)

    def _compute_conv_crit(self, rom, conv_crit, it):
        if conv_crit == 'sigma':
            sigma = self._conv_data[0]
            dist = min(spla.norm((sigma_old - sigma) / sigma_old, ord=np.inf)
                       for sigma_old in self._conv_data[1:]
                       if sigma_old is not None)
        else:
            if rom.poles().real.max() >= 0:
                dist = np.inf
            else:
                dist = min((rom_old - rom).h2_norm() / rom_old.h2_norm()
                           if rom_old is not None and rom_old.poles().real.max() < 0
                           else np.inf
                           for rom_old in self._conv_data[1:])
        self.conv_crit.append(dist)
        self.logger.info(f'Convergence criterion in iteration {it + 1}: {dist:e}')

    def _compute_error(self, rom, it, compute_errors):
        if not compute_errors:
            return
        rel_h2_err = ((self.fom - rom).h2_norm() / self.fom.h2_norm()
                      if rom.poles().real.max() < 0
                      else np.inf)
        self.errors.append(rel_h2_err)
        self.logger.info(f'Relative H2-error in iteration {it + 1}: {rel_h2_err:e}')


class IRKAReductor(GenericIRKAReductor):
    """Iterative Rational Krylov Algorithm reductor.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    mu
        |Parameter values|.
    """
    def __init__(self, fom, mu=None):
        assert isinstance(fom, LTIModel)
        super().__init__(fom, mu=mu)

    def reduce(self, rom0_params, tol=1e-4, maxit=100, num_prev=1,
               force_sigma_in_rhp=False, projection='orth', conv_crit='sigma',
               compute_errors=False):
        r"""Reduce using IRKA.

        See [GAB08]_ (Algorithm 4.1) and [ABG10]_ (Algorithm 1).

        Parameters
        ----------
        rom0_params
            Can be:

            - order of the reduced model (a positive integer),
            - initial interpolation points (a 1D |NumPy array|),
            - dict with `'sigma'`, `'b'`, `'c'` as keys mapping to
              initial interpolation points (a 1D |NumPy array|), right
              tangential directions (|VectorArray| from
              `fom.input_space`), and left tangential directions
              (|VectorArray| from `fom.output_space`), all of the same
              length (the order of the reduced model),
            - initial reduced-order model (|LTIModel|).

            If the order of reduced model is given, initial
            interpolation data is generated randomly.
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
            reduced-order model's poles. Otherwise, only poles in the
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
            intermediate reduced-order models be computed.

            .. warning::
                Computing :math:`\mathcal{H}_2`-errors is expensive. Use
                this option only if necessary.

        Returns
        -------
        rom
            Reduced |LTIModel| model.
        """
        if not self.fom.cont_time:
            raise NotImplementedError

        self._clear_lists()
        sigma, b, c = self._rom0_params_to_sigma_b_c(rom0_params, force_sigma_in_rhp)
        self._store_sigma_b_c(sigma, b, c)
        self._check_common_args(tol, maxit, num_prev, conv_crit)
        assert projection in ('orth', 'biorth', 'arnoldi')
        if projection == 'arnoldi':
            assert self.fom.input_dim == self.fom.output_dim == 1

        self.logger.info('Starting IRKA')
        self._conv_data = (num_prev + 1) * [None]
        if conv_crit == 'sigma':
            self._conv_data[0] = sigma
        self._pg_reductor = LTIBHIReductor(self.fom, mu=self.mu)
        for it in range(maxit):
            rom = self._pg_reductor.reduce(sigma, b, c, projection=projection)
            sigma, b, c = self._rom_to_sigma_b_c(rom, force_sigma_in_rhp)
            self._store_sigma_b_c(sigma, b, c)
            self._update_conv_data(sigma, rom, conv_crit)
            self._compute_conv_crit(rom, conv_crit, it)
            self._compute_error(rom, it, compute_errors)
            if self.conv_crit[-1] < tol:
                break

        self.V = self._pg_reductor.V
        self.W = self._pg_reductor.W
        return rom


class OneSidedIRKAReductor(GenericIRKAReductor):
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
        |Parameter values|.
    """
    def __init__(self, fom, version, mu=None):
        assert isinstance(fom, LTIModel)
        assert version in ('V', 'W')
        super().__init__(fom, mu=mu)
        self.version = version

    def reduce(self, rom0_params, tol=1e-4, maxit=100, num_prev=1,
               force_sigma_in_rhp=False, projection='orth', conv_crit='sigma',
               compute_errors=False):
        r"""Reduce using one-sided IRKA.

        Parameters
        ----------
        rom0_params
            Can be:

            - order of the reduced model (a positive integer),
            - initial interpolation points (a 1D |NumPy array|),
            - dict with `'sigma'`, `'b'`, `'c'` as keys mapping to
              initial interpolation points (a 1D |NumPy array|), right
              tangential directions (|VectorArray| from
              `fom.input_space`), and left tangential directions
              (|VectorArray| from `fom.output_space`), all of the same
              length (the order of the reduced model),
            - initial reduced-order model (|LTIModel|).

            If the order of reduced model is given, initial
            interpolation data is generated randomly.
        tol
            Tolerance for the largest change in interpolation points.
        maxit
            Maximum number of iterations.
        num_prev
            Number of previous iterations to compare the current
            iteration to. A larger number can avoid occasional cyclic
            behavior.
        force_sigma_in_rhp
            If `False`, new interpolation are reflections of the current
            reduced-order model's poles. Otherwise, only poles in the
            left half-plane are reflected.
        projection
            Projection method:

            - `'orth'`: projection matrix is orthogonalized with respect
              to the Euclidean inner product,
            - `'Eorth'`: projection matrix is orthogonalized with
              respect to the E product.
        conv_crit
            Convergence criterion:

            - `'sigma'`: relative change in interpolation points,
            - `'h2'`: relative :math:`\mathcal{H}_2` distance of
              reduced-order models.
        compute_errors
            Should the relative :math:`\mathcal{H}_2`-errors of
            intermediate reduced-order models be computed.

            .. warning::
                Computing :math:`\mathcal{H}_2`-errors is expensive. Use
                this option only if necessary.

        Returns
        -------
        rom
            Reduced |LTIModel| model.
        """
        if not self.fom.cont_time:
            raise NotImplementedError

        self._clear_lists()
        sigma, b, c = self._rom0_params_to_sigma_b_c(rom0_params, force_sigma_in_rhp)
        self._store_sigma_b_c(sigma, b, c)
        self._check_common_args(tol, maxit, num_prev, conv_crit)
        assert projection in ('orth', 'Eorth')

        self.logger.info('Starting one-sided IRKA')
        self._conv_data = (num_prev + 1) * [None]
        if conv_crit == 'sigma':
            self._conv_data[0] = sigma
        for it in range(maxit):
            self._set_V_reductor(sigma, b, c, projection)
            rom = self._pg_reductor.reduce()
            sigma, b, c = self._rom_to_sigma_b_c(rom, force_sigma_in_rhp)
            self._store_sigma_b_c(sigma, b, c)
            self._update_conv_data(sigma, rom, conv_crit)
            self._compute_conv_crit(rom, conv_crit, it)
            self._compute_error(rom, it, compute_errors)
            if self.conv_crit[-1] < tol:
                break

        return rom

    def _set_V_reductor(self, sigma, b, c, projection):
        fom = (
            self.fom.with_(
                **{op: getattr(self.fom, op).assemble(mu=self.mu)
                   for op in ['A', 'B', 'C', 'D', 'E']}
            )
            if self.fom.parametric
            else self.fom
        )
        if self.version == 'V':
            self.V = tangential_rational_krylov(fom.A, fom.E, fom.B, b, sigma,
                                                orth=False)
            gram_schmidt(self.V, atol=0, rtol=0,
                         product=None if projection == 'orth' else fom.E,
                         copy=False)
        else:
            self.V = tangential_rational_krylov(fom.A, fom.E, fom.C, c, sigma, trans=True,
                                                orth=False)
            gram_schmidt(self.V, atol=0, rtol=0,
                         product=None if projection == 'orth' else fom.E,
                         copy=False)
        self.W = self.V
        self._pg_reductor = LTIPGReductor(fom, self.V, self.V,
                                          projection == 'Eorth')


class TSIAReductor(GenericIRKAReductor):
    """Two-Sided Iteration Algorithm reductor.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    mu
        |Parameter values|.
    """
    def __init__(self, fom, mu=None):
        assert isinstance(fom, LTIModel)
        super().__init__(fom, mu=mu)

    def reduce(self, rom0_params, tol=1e-4, maxit=100, num_prev=1, projection='orth',
               conv_crit='sigma', compute_errors=False):
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
        rom0_params
            Can be:

            - order of the reduced model (a positive integer),
            - initial interpolation points (a 1D |NumPy array|),
            - dict with `'sigma'`, `'b'`, `'c'` as keys mapping to
              initial interpolation points (a 1D |NumPy array|), right
              tangential directions (|VectorArray| from
              `fom.input_space`), and left tangential directions
              (|VectorArray| from `fom.output_space`), all of the same
              length (the order of the reduced model),
            - initial reduced-order model (|LTIModel|).

            If the order of reduced model is given, initial
            interpolation data is generated randomly.
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
            intermediate reduced-order models be computed.

            .. warning::
                Computing :math:`\mathcal{H}_2`-errors is expensive. Use
                this option only if necessary.

        Returns
        -------
        rom
            Reduced |LTIModel|.
        """
        if not self.fom.cont_time:
            raise NotImplementedError

        self._clear_lists()
        rom = self._rom0_params_to_rom(rom0_params)
        self._check_common_args(tol, maxit, num_prev, conv_crit)
        assert projection in ('orth', 'biorth')

        self.logger.info('Starting TSIA')
        self._conv_data = (num_prev + 1) * [None]
        self._conv_data[0] = -rom.poles() if conv_crit == 'sigma' else rom
        self._store_sigma_b_c(-rom.poles(), None, None)
        for it in range(maxit):
            self._set_V_W_reductor(rom, projection)
            rom = self._pg_reductor.reduce()
            self._store_sigma_b_c(-rom.poles(), None, None)
            self._update_conv_data(-rom.poles(), rom, conv_crit)
            self._compute_conv_crit(rom, conv_crit, it)
            self._compute_error(rom, it, compute_errors)
            if self.conv_crit[-1] < tol:
                break

        return rom

    def _set_V_W_reductor(self, rom, projection):
        fom = (
            self.fom.with_(
                **{op: getattr(self.fom, op).assemble(mu=self.mu)
                   for op in ['A', 'B', 'C', 'D', 'E']}
            )
            if self.fom.parametric
            else self.fom
        )
        self.V, self.W = solve_sylv_schur(fom.A, rom.A,
                                          E=fom.E, Er=rom.E,
                                          B=fom.B, Br=rom.B,
                                          C=fom.C, Cr=rom.C)
        if projection == 'orth':
            gram_schmidt(self.V, atol=0, rtol=0, copy=False)
            gram_schmidt(self.W, atol=0, rtol=0, copy=False)
        elif projection == 'biorth':
            gram_schmidt_biorth(self.V, self.W, product=fom.E, copy=False)
        self._pg_reductor = LTIPGReductor(fom, self.W, self.V,
                                          projection == 'biorth')


class TFIRKAReductor(GenericIRKAReductor):
    """Realization-independent IRKA reductor.

    See [BG12]_.

    Parameters
    ----------
    fom
        The full-order |Model| with `eval_tf` and `eval_dtf` methods.
    mu
        |Parameter values|.
    """
    def __init__(self, fom, mu=None):
        assert isinstance(fom, InputOutputModel)
        super().__init__(fom, mu=mu)

    def reduce(self, rom0_params, tol=1e-4, maxit=100, num_prev=1,
               force_sigma_in_rhp=False, conv_crit='sigma', compute_errors=False):
        r"""Reduce using TF-IRKA.

        Parameters
        ----------
        rom0_params
            Can be:

            - order of the reduced model (a positive integer),
            - initial interpolation points (a 1D |NumPy array|),
            - dict with `'sigma'`, `'b'`, `'c'` as keys mapping to
              initial interpolation points (a 1D |NumPy array|), right
              tangential directions (|VectorArray| from
              `fom.input_space`), and left tangential directions
              (|VectorArray| from `fom.output_space`), all of the same
              length (the order of the reduced model),
            - initial reduced-order model (|LTIModel|).

            If the order of reduced model is given, initial
            interpolation data is generated randomly.
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
            reduced-order model's poles. Otherwise, only poles in the
            left half-plane are reflected.
        conv_crit
            Convergence criterion:

            - `'sigma'`: relative change in interpolation points
            - `'h2'`: relative :math:`\mathcal{H}_2` distance of
              reduced-order models
        compute_errors
            Should the relative :math:`\mathcal{H}_2`-errors of
            intermediate reduced-order models be computed.

            .. warning::
                Computing :math:`\mathcal{H}_2`-errors is expensive. Use
                this option only if necessary.

        Returns
        -------
        rom
            Reduced |LTIModel| model.
        """
        if not self.fom.cont_time:
            raise NotImplementedError

        self._clear_lists()
        sigma, b, c = self._rom0_params_to_sigma_b_c(rom0_params, force_sigma_in_rhp)
        self._store_sigma_b_c(sigma, b, c)
        self._check_common_args(tol, maxit, num_prev, conv_crit)

        self.logger.info('Starting TF-IRKA')
        self._conv_data = (num_prev + 1) * [None]
        if conv_crit == 'sigma':
            self._conv_data[0] = sigma
        interp_reductor = TFBHIReductor(self.fom, mu=self.mu)
        for it in range(maxit):
            rom = interp_reductor.reduce(sigma, b, c)
            sigma, b, c = self._rom_to_sigma_b_c(rom, force_sigma_in_rhp)
            self._store_sigma_b_c(sigma, b, c)
            self._update_conv_data(sigma, rom, conv_crit)
            self._compute_conv_crit(rom, conv_crit, it)
            self._compute_error(rom, it, compute_errors)
            if self.conv_crit[-1] < tol:
                break

        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        raise TypeError(
            f'The reconstruct method is not available for {self.__class__.__name__}.'
        )


def _lti_to_poles_b_c(rom):
    """Compute poles and residues.

    Parameters
    ----------
    rom
        Reduced |LTIModel| (consisting of |NumpyMatrixOperators|).

    Returns
    -------
    poles
        1D |NumPy array| of poles.
    b
        |VectorArray| from `rom.B.source`.
    c
        |VectorArray| from `rom.C.range`.
    """
    A = to_matrix(rom.A, format='dense')
    B = to_matrix(rom.B, format='dense')
    C = to_matrix(rom.C, format='dense')
    if isinstance(rom.E, IdentityOperator):
        poles, X = spla.eig(A)
        EX = X
    else:
        E = to_matrix(rom.E, format='dense')
        poles, X = spla.eig(A, E)
        EX = E @ X
    b = rom.B.source.from_numpy(spla.solve(EX, B))
    c = rom.C.range.from_numpy((C @ X).T)
    return poles, b, c


def _poles_b_c_to_lti(poles, b, c):
    r"""Create an |LTIModel| from poles and residue rank-1 factors.

    Returns an |LTIModel| with real matrices such that its transfer
    function is

    .. math::
        \sum_{i = 1}^r \frac{c_i b_i^T}{s - \lambda_i}

    where :math:`\lambda_i, b_i, c_i` are the poles and residue rank-1
    factors.

    Parameters
    ----------
    poles
        Sequence of poles.
    b
        |VectorArray| of right residue rank-1 factors.
    c
        |VectorArray| of left residue rank-1 factors.

    Returns
    -------
    |LTIModel|.
    """
    A, B, C = [], [], []
    for i, pole in enumerate(poles):
        if pole.imag == 0:
            A.append(pole.real)
            B.append(b[i].to_numpy().real)
            C.append(c[i].to_numpy().real.T)
        elif pole.imag > 0:
            A.append([[pole.real, pole.imag],
                      [-pole.imag, pole.real]])
            bi = b[i].to_numpy()
            B.append(np.vstack([2 * bi.real, -2 * bi.imag]))
            ci = c[i].to_numpy()
            C.append(np.hstack([ci.real.T, ci.imag.T]))
    A = spla.block_diag(*A)
    B = np.vstack(B)
    C = np.hstack(C)
    return LTIModel.from_matrices(A, B, C)
