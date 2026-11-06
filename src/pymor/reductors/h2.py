# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Reductors based on H2-norm."""

from functools import partial
from numbers import Integral, Real

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.krylov import tangential_rational_krylov
from pymor.algorithms.loewner import complete_conjugate_pairs, sample_transfer_function
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel, _lti_to_poles_b_c, _poles_b_c_to_lti
from pymor.models.transfer_function import TransferFunction
from pymor.operators.constructions import IdentityOperator
from pymor.parameters.base import Mu
from pymor.reductors.basic import LTIPGReductor
from pymor.reductors.interpolation import LTIBHIReductor, TFBHIReductor, _assemble_fom
from pymor.solvers.matrix_equations.equations import RiccatiEquation, SylvesterEquation
from pymor.solvers.matrix_equations.sylvester import SylvesterSchurSolver
from pymor.tools.random import new_rng


class GenericIRKAReductor(BasicObject):
    """Generic IRKA related reductor.

    IRKA-type methods are fixed-point iterations which alternate between
    computing a reduced-order model from interpolation data and computing
    new interpolation data from that reduced-order model.
    Subclasses implement `reduce` by passing these two steps to
    :meth:`_iterate`.

    Parameters
    ----------
    fom
        The full-order |Model| to reduce.
    mu
        |Parameter values|.
    """

    _conv_crits = ('sigma', 'h2')

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
            assert {'sigma', 'b', 'c'} <= rom0_params.keys()
            assert isinstance(rom0_params['sigma'], np.ndarray)
            assert isinstance(rom0_params['b'], np.ndarray)
            assert isinstance(rom0_params['c'], np.ndarray)
            assert rom0_params['sigma'].ndim == 1
            assert rom0_params['b'].shape[1] == self.fom.dim_input
            assert rom0_params['c'].shape[1] == self.fom.dim_output
            assert len(rom0_params['sigma']) == rom0_params['b'].shape[0]
            assert len(rom0_params['sigma']) == rom0_params['c'].shape[0]
        elif isinstance(rom0_params, LTIModel):
            assert rom0_params.order > 0
            if hasattr(self.fom, 'order'):  # self.fom can be a TransferFunction
                assert rom0_params.order < self.fom.order
            assert rom0_params.dim_input == self.fom.dim_input
            assert rom0_params.dim_output == self.fom.dim_output
        else:
            raise ValueError(f'rom0_params is of wrong type ({type(rom0_params)}).')

    def _check_common_args(self, tol, maxit, num_prev, conv_crit):
        assert isinstance(tol, Real)
        assert tol > 0
        assert isinstance(maxit, Integral)
        assert maxit >= 1
        assert isinstance(num_prev, Integral)
        assert num_prev >= 1
        assert conv_crit in self._conv_crits

    def _order_to_sigma_b_c(self, r):
        sigma = np.logspace(-1, 1, r)
        b = (np.ones((r, 1))
             if self.fom.dim_input == 1
             else new_rng(0).normal(size=(r, self.fom.dim_input)))
        c = (np.ones((r, 1))
             if self.fom.dim_output == 1
             else new_rng(0).normal(size=(r, self.fom.dim_output)))
        return sigma, b, c

    @staticmethod
    def _rom_to_sigma_b_c(rom, force_sigma_in_rhp):
        poles, b, c = _lti_to_poles_b_c(rom)
        sigma = (np.abs(poles.real) + poles.imag * 1j
                 if force_sigma_in_rhp
                 else -poles)
        return sigma, b, c

    def _rom0_params_to_sigma_b_c(self, rom0_params, rom_to_sigma_b_c):
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
            sigma, b, c = rom_to_sigma_b_c(rom0_params)
        return sigma, b, c

    def _rom0_params_to_rom(self, rom0_params):
        if isinstance(rom0_params, LTIModel):
            self._check_rom0_params(rom0_params)
            return rom0_params
        sigma, b, c = self._rom0_params_to_sigma_b_c(rom0_params, None)
        return _poles_b_c_to_lti(-sigma, b, c)

    def _store_sigma_b_c(self, sigma, b, c):
        if sigma is not None:
            self.sigma_list.append(sigma)
        if b is not None:
            self.b_list.append(b)
        if c is not None:
            self.c_list.append(c)

    def _sigma_b_c(self, data):
        """Return interpolation points and tangential directions in interpolation data."""
        return data

    def _bhi_step(self, bhi_reductor, data, projection):
        """Interpolate using `bhi_reductor` and keep its projection."""
        rom = bhi_reductor.reduce(*data, projection=projection)
        self.V, self.W, self._pg_reductor = bhi_reductor.V, bhi_reductor.W, bhi_reductor
        return rom

    def _iterate(self, name, data, step, update, tol, maxit, num_prev, conv_crit, compute_errors):
        """Run the fixed-point iteration of an IRKA-type method.

        Parameters
        ----------
        name
            Name of the method used in log messages.
        data
            Initial interpolation data.
        step
            Function computing a reduced-order model from interpolation data.
            Projection-based methods also set `V`, `W` and `_pg_reductor`.
        update
            Function computing new interpolation data from a reduced-order
            model.
        tol, maxit, num_prev, conv_crit, compute_errors
            See :meth:`IRKAReductor.reduce`.

        Returns
        -------
        rom
            Reduced-order model from the last iteration.
        """
        if self.fom.sampling_time > 0:
            raise NotImplementedError
        self._check_common_args(tol, maxit, num_prev, conv_crit)

        self._clear_lists()
        sigma, b, c = self._sigma_b_c(data)
        self._store_sigma_b_c(sigma, b, c)
        self._conv_data = (num_prev + 1) * [None]
        if conv_crit == 'sigma':
            self._conv_data[0] = sigma

        self.logger.info(f'Starting {name}')
        for it in range(maxit):
            rom = step(data)
            data = update(rom)
            sigma, b, c = self._sigma_b_c(data)
            self._store_sigma_b_c(sigma, b, c)
            del self._conv_data[-1]
            self._conv_data.insert(0, self._conv_quantity(conv_crit, sigma, rom))
            self.conv_crit.append(self._conv_distance(conv_crit, rom))
            self.logger.info(f'Convergence criterion in iteration {it + 1}: {self.conv_crit[-1]:e}')
            self._compute_error(rom, it, compute_errors)
            if self.conv_crit[-1] < tol:
                break

        return rom

    def _conv_quantity(self, conv_crit, sigma, rom):
        """Return the quantity whose change is measured by `conv_crit`."""
        return sigma if conv_crit == 'sigma' else rom

    def _conv_distance(self, conv_crit, rom):
        """Return the distance of the newest to the previous convergence quantities."""
        if conv_crit == 'sigma':
            sigma = self._conv_data[0]
            return min(spla.norm((sigma_old - sigma) / sigma_old, ord=np.inf)
                       for sigma_old in self._conv_data[1:]
                       if sigma_old is not None)
        if rom.poles().real.max() >= 0:
            return np.inf
        return min((rom_old - rom).h2_norm() / rom_old.h2_norm()
                   if rom_old is not None and rom_old.poles().real.max() < 0
                   else np.inf
                   for rom_old in self._conv_data[1:])

    def _compute_error(self, rom, it, compute_errors):
        if not compute_errors:
            return
        rel_h2_err = ((self.fom - rom).h2_norm(mu=self.mu) / self.fom.h2_norm(mu=self.mu)
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

        See :cite:`GAB08` (Algorithm 4.1) and :cite:`ABG10` (Algorithm 1).

        Parameters
        ----------
        rom0_params
            Can be:

            - order of the reduced model (a positive integer),
            - initial interpolation points (a 1D |NumPy array|),
            - dict with `'sigma'`, `'b'`, `'c'` as keys mapping to
              initial interpolation points (a 1D |NumPy array|), right
              tangential directions (|NumPy array| of shape
              `(len(sigma), fom.dim_input)`), and left tangential directions
              (|NumPy array| of shape `(len(sigma), fom.dim_output)`),
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
            If `False`, new interpolation points are reflections of the current
            reduced-order model's poles. Otherwise, only poles in the
            left half-plane are reflected.
        projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogonalized with
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
        assert projection in ('orth', 'biorth', 'arnoldi')
        if projection == 'arnoldi':
            assert self.fom.dim_input == self.fom.dim_output == 1

        rom_to_sigma_b_c = partial(self._rom_to_sigma_b_c, force_sigma_in_rhp=force_sigma_in_rhp)
        data = self._rom0_params_to_sigma_b_c(rom0_params, rom_to_sigma_b_c)
        step = partial(self._bhi_step, LTIBHIReductor(self.fom, mu=self.mu), projection=projection)
        return self._iterate('IRKA', data, step, rom_to_sigma_b_c,
                             tol, maxit, num_prev, conv_crit, compute_errors)


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
              tangential directions (|NumPy array| of shape
              `(len(sigma), fom.dim_input)`), and left tangential directions
              (|NumPy array| of shape `(len(sigma), fom.dim_output)`),
            - initial reduced-order model (|LTIModel|).

            If the order of reduced model is given, initial
            interpolation data is generated randomly.
        tol
            Tolerance for the convergence criterion.
        maxit
            Maximum number of iterations.
        num_prev
            Number of previous iterations to compare the current
            iteration to. A larger number can avoid occasional cyclic
            behavior.
        force_sigma_in_rhp
            If `False`, new interpolation points are reflections of the current
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
        assert projection in ('orth', 'Eorth')

        rom_to_sigma_b_c = partial(self._rom_to_sigma_b_c, force_sigma_in_rhp=force_sigma_in_rhp)
        data = self._rom0_params_to_sigma_b_c(rom0_params, rom_to_sigma_b_c)
        fom = _assemble_fom(self.fom, self.mu, ('A', 'B', 'C', 'D', 'E'))
        step = partial(self._krylov_step, fom, projection=projection)
        return self._iterate('one-sided IRKA', data, step, rom_to_sigma_b_c,
                             tol, maxit, num_prev, conv_crit, compute_errors)

    def _krylov_step(self, fom, data, projection):
        sigma, b, c = data
        if self.version == 'V':
            self.V = tangential_rational_krylov(fom.A, fom.E, fom.B, fom.B.source.from_numpy(b.T), sigma,
                                                orth=False, shifted_system_solver=fom.shifted_system_solver)
        else:
            self.V = tangential_rational_krylov(fom.A, fom.E, fom.C, fom.C.range.from_numpy(c.T), sigma, trans=True,
                                                orth=False, shifted_system_solver=fom.shifted_system_solver)
        gram_schmidt(self.V, atol=0, rtol=0,
                     product=None if projection == 'orth' else fom.E,
                     copy=False)
        self.W = self.V
        self._pg_reductor = LTIPGReductor(fom, self.V, self.V,
                                          projection == 'Eorth')
        return self._pg_reductor.reduce()


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

        See :cite:`XZ11` (Algorithm 1) and :cite:`BKS11`.

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
              tangential directions (|NumPy array| of shape
              `(len(sigma), fom.dim_input)`), and left tangential directions
              (|NumPy array| of shape `(len(sigma), fom.dim_output)`),
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
            - `'biorth'`: projection matrices are biorthogonalized with
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
        assert projection in ('orth', 'biorth')

        # the interpolation data of TSIA is the reduced-order model itself
        data = self._rom0_params_to_rom(rom0_params)
        fom = _assemble_fom(self.fom, self.mu, ('A', 'B', 'C', 'D', 'E'))
        step = partial(self._sylvester_step, fom, projection=projection)
        return self._iterate('TSIA', data, step, lambda rom: rom,
                             tol, maxit, num_prev, conv_crit, compute_errors)

    def _sigma_b_c(self, data):
        return -data.poles(), None, None

    def _sylvester_step(self, fom, rom, projection):
        equation = SylvesterEquation(fom.A, rom.A, E=fom.E, Er=rom.E, B=fom.B, Br=rom.B, C=fom.C, Cr=rom.C)
        self.V, self.W = equation.solve(solver=SylvesterSchurSolver(shifted_system_solver=fom.shifted_system_solver))

        if projection == 'orth':
            gram_schmidt(self.V, atol=0, rtol=0, copy=False)
            gram_schmidt(self.W, atol=0, rtol=0, copy=False)
        elif projection == 'biorth':
            gram_schmidt_biorth(self.V, self.W, product=fom.E, copy=False)
        self._pg_reductor = LTIPGReductor(fom, self.W, self.V,
                                          projection == 'biorth')
        return self._pg_reductor.reduce()


class TFIRKAReductor(GenericIRKAReductor):
    """Realization-independent IRKA reductor.

    See :cite:`BG12`.

    Parameters
    ----------
    fom
        |TransferFunction| or |Model| with a `transfer_function` attribute,
        with `eval_tf` and `eval_dtf` methods that should be defined at least
        over the open right half of the complex plane.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        assert isinstance(fom, TransferFunction) or hasattr(fom, 'transfer_function')
        if not isinstance(fom, TransferFunction):
            fom = fom.transfer_function
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
              tangential directions (|NumPy array| of shape
              `(len(sigma), fom.dim_input)`), and left tangential directions
              (|NumPy array| of shape `(len(sigma), fom.dim_output)`),
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
            If `False`, new interpolation points are reflections of the current
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
        rom_to_sigma_b_c = partial(self._rom_to_sigma_b_c, force_sigma_in_rhp=force_sigma_in_rhp)
        data = self._rom0_params_to_sigma_b_c(rom0_params, rom_to_sigma_b_c)
        interp_reductor = TFBHIReductor(self.fom, mu=self.mu)
        return self._iterate('TF-IRKA', data, lambda data: interp_reductor.reduce(*data), rom_to_sigma_b_c,
                             tol, maxit, num_prev, conv_crit, compute_errors)

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        raise TypeError(
            f'The reconstruct method is not available for {self.__class__.__name__}.'
        )


class GapIRKAReductor(GenericIRKAReductor):
    """Gap-IRKA reductor.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    mu
        |Parameter values|.
    riccati_solver
        |RiccatiSolver| for computing the normalized coprime factorizations
        of intermediate reduced-order models (if `None`, the default solver
        is used).
    """

    _conv_crits = ('sigma', 'htwogap', 'ltwo')

    def __init__(self, fom, mu=None, riccati_solver=None):
        assert isinstance(fom, LTIModel)
        super().__init__(fom, mu=mu)
        self.riccati_solver = riccati_solver

    def reduce(self, rom0_params, tol=1e-4, maxit=100, num_prev=1, conv_crit='sigma',
               projection='orth'):
        r"""Reduce using gap-IRKA.

        See :cite:`BBG19` Algorithm 1.

        Parameters
        ----------
        rom0_params
            Can be:

            - order of the reduced model (a positive integer),
            - initial interpolation points (a 1D |NumPy array|),
            - dict with `'sigma'`, `'b'`, `'c'` as keys mapping to
              initial interpolation points (a 1D |NumPy array|), right
              tangential directions (|NumPy array| of shape
              `(len(sigma), fom.dim_input)`), and left tangential directions
              (|NumPy array| of shape `(len(sigma), fom.dim_output)`),
            - initial reduced-order model (|LTIModel|).

            If the order of reduced model is given, initial
            interpolation data is generated randomly.
        tol
            Tolerance for the convergence criterion.
        maxit
            Maximum number of iterations.
        num_prev
            Number of previous iterations to compare the current
            iteration to. A larger number can avoid occasional cyclic
            behavior.
        conv_crit
            Convergence criterion:

            - `'sigma'`: relative change in interpolation points
            - `'htwogap'`: :math:`\mathcal{H}_2-gap` distance of
              reduced-order models divided by :math:`\mathcal{L}_2`
              norm of new reduced-order model
            - `'ltwo'`: relative :math:`\mathcal{L}_2` distance of
              reduced-order models
        projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogonalized with
              respect to the E product

        Returns
        -------
        rom
            Reduced |LTIModel| model.
        """
        assert projection in ('orth', 'biorth')

        data = self._rom0_params_to_sigma_b_c(rom0_params, self._unstable_rom_to_sigma_b_c)
        step = partial(self._bhi_step, LTIBHIReductor(self.fom, mu=self.mu), projection=projection)
        return self._iterate('gap IRKA', data, step, self._unstable_rom_to_sigma_b_c,
                             tol, maxit, num_prev, conv_crit, False)

    def _unstable_rom_to_sigma_b_c(self, rom):
        poles, b, c, _ = self._unstable_lti_to_poles_b_c(rom)
        return -poles, b, c

    def _unstable_lti_to_poles_b_c(self, rom):
        A = to_matrix(rom.A, format='dense')
        B = to_matrix(rom.B, format='dense')
        C = to_matrix(rom.C, format='dense')

        if isinstance(rom.E, IdentityOperator):
            P = RiccatiEquation(rom.A, None, rom.B.as_range_array(),
                                rom.C.as_source_array()).solve(solver=self.riccati_solver)

            F = P @ C.T
            AF = A - F @ C
            poles, X = spla.eig(AF)
            EX = X
        else:
            P = RiccatiEquation(rom.A, rom.E, rom.B.as_range_array(),
                                rom.C.as_source_array()).solve(solver=self.riccati_solver)
            E = to_matrix(rom.E, format='dense')
            F = E @ P @ C.T
            AF = A - F @ C
            poles, X = spla.eig(AF, E)
            EX = E @ X
        b = spla.solve(EX, B)
        c = (C @ X).T
        mFB = np.concatenate((-F, B), axis=1)
        gap_rom = LTIModel.from_matrices(AF, mFB, C, E=None if isinstance(rom.E, IdentityOperator) else E)
        return poles, b, c, gap_rom

    def _conv_quantity(self, conv_crit, sigma, rom):
        if conv_crit == 'htwogap':
            return self._unstable_lti_to_poles_b_c(rom)[3]
        return super()._conv_quantity(conv_crit, sigma, rom)

    def _conv_distance(self, conv_crit, rom):
        if conv_crit == 'htwogap':
            gap_rom = self._conv_data[0]
            A = to_matrix(rom.A, format='dense')
            C = to_matrix(rom.C, format='dense')
            F = -to_matrix(gap_rom.B, format='dense')[:, :C.shape[0]]
            D = np.eye(rom.C.range.dim)
            E = None if isinstance(rom.E, IdentityOperator) else to_matrix(rom.E, format='dense')

            mi = LTIModel.from_matrices(A, F, C, D, E)
            return min((gap_rom_old - gap_rom).h2_norm() * mi.linf_norm() / rom.l2_norm()
                       if gap_rom_old is not None
                       else np.inf
                       for gap_rom_old in self._conv_data[1:])
        if conv_crit == 'ltwo':
            return min((rom_old - rom).l2_norm() / rom.l2_norm()
                       if rom_old is not None
                       else np.inf
                       for rom_old in self._conv_data[1:])
        return super()._conv_distance(conv_crit, rom)


class VectorFittingReductor(BasicObject):
    """Vector fitting reductor.

    Only for single-input single-output (SISO) systems.

    Parameters
    ----------
    s
        Sampling points in the complex plane as a 1D |NumPy array|.
    Hs
        Transfer function values at the sampling points `s` as a 1D |NumPy array|.
        Use :meth:`generate_samples` to generate data beforehand.
    weights
        Weights in the weighted least squares error as a 1D |NumPy array|.
        If not given, it is set to a vector of ones.
    force_real
        Whether to include conjugated data to enforce realness.
    """

    generate_samples = staticmethod(sample_transfer_function)

    def __init__(self, s, Hs, weights=None, force_real=True):
        assert isinstance(s, np.ndarray)
        assert s.ndim == 1

        Hs = np.asarray(Hs)
        assert Hs.ndim in (1, 3), 'Hs must contain sample data; use generate_samples to sample a model.'
        if Hs.ndim == 3 and Hs.shape[1:] == (1, 1):
            Hs = Hs[:, 0, 0]
        assert Hs.ndim == 1
        assert len(Hs) == len(s)

        if weights is None:
            weights = np.ones(len(s))
        assert isinstance(weights, np.ndarray)
        assert weights.ndim == 1
        assert len(weights) == len(s)
        assert np.all(weights > 0)

        # add complex conjugate samples
        if force_real:
            s, Hs, weights = complete_conjugate_pairs(s, Hs, weights)

        self.s = s
        self.Hs = Hs
        self.weights_sqrt = np.sqrt(weights)

    def reduce(self, r=None, lambdas=None, tol=1e-4, maxit=100):
        """Reduce using vector fitting.

        Based on :cite:`DGB15`.

        Parameters
        ----------
        r
            Reduced order (if not given, it is `len(lambdas)`).
        lambdas
            Initial poles (if not given, it is set to `-np.logspace(-1, 0, r)`).
        tol
            Tolerance for the convergence criterion.
        maxit
            Maximum number of iterations.

        Returns
        -------
        rom
            Reduced-order |LTIModel|.
        """
        if r is None and lambdas is None:
            raise ValueError('Either r or lambdas must be given.')
        if r is not None and lambdas is not None:
            raise ValueError('Only r or lambdas must be given.')

        if lambdas is None:
            lambdas = -np.logspace(-1, 0, r)
        else:
            assert isinstance(lambdas, np.ndarray)
            assert lambdas.ndim == 1
            r = len(lambdas)

        # right-hand side in least squares problems
        b = self.weights_sqrt * self.Hs

        for i in range(maxit):
            self.logger.info(f'Iteration {i + 1}')

            # least squares problem for barycentric form
            A1 = self.weights_sqrt[:, np.newaxis] / (self.s[:, np.newaxis] - lambdas)
            A2 = -self.weights_sqrt[:, np.newaxis] * self.Hs[:, np.newaxis] / (self.s[:, np.newaxis] - lambdas)
            A = np.hstack((A1, A2))
            x = spla.lstsq(A, b)[0]
            psis = x[:r]
            phis = x[r:]

            # convergence check
            conv_crit = spla.norm(phis / lambdas.real, ord=1)
            self.logger.info(f'{conv_crit=:.3e}')
            error = spla.norm(
                self.weights_sqrt * (
                    self.Hs
                    - np.sum(psis / (self.s[:, np.newaxis] - lambdas), axis=1)
                    / (1 + np.sum(phis / (self.s[:, np.newaxis] - lambdas), axis=1))
                )
                / len(self.s)
            )
            self.logger.info(f'{error=:.3e}')
            if conv_crit < tol:
                break

            # real version of A_zeros = np.diag(lambdas) + ones * phi.T
            A_zeros = np.zeros((r, r))
            ones = np.ones(r)
            phis_real = np.empty(r)
            k = 0
            for lam, phi in zip(lambdas, phis, strict=True):
                if lam.imag == 0:
                    A_zeros[k, k] = lam.real
                    phis_real[k] = phi.real
                    k += 1
                elif lam.imag > 0:
                    A_zeros[k, k] = lam.real
                    A_zeros[k, k + 1] = lam.imag
                    A_zeros[k + 1, k] = -lam.imag
                    A_zeros[k + 1, k + 1] = lam.real
                    ones[k] = 2
                    ones[k + 1] = 0
                    phis_real[k] = phi.real
                    phis_real[k + 1] = phi.imag
                    k += 2
            A_zeros += np.outer(ones, phis_real)
            zeros = spla.eigvals(A_zeros)

            # new lambdas
            lambdas = -np.abs(zeros.real) + 1j * zeros.imag

        # least squares problem for residues
        A = self.weights_sqrt[:, np.newaxis] / (self.s[:, np.newaxis] - lambdas)
        psis = spla.lstsq(A, b)[0]

        return _poles_b_c_to_lti(lambdas, psis[:, np.newaxis], np.ones((r, 1)))
