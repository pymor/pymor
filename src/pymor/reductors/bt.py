# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from functools import partial

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.discretizations.iosys import _DEFAULT_ME_SOLVER_BACKEND
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.basic import GenericPGReductor


class GenericBTReductor(GenericPGReductor):
    """Generic Balanced Truncation reductor.

    Parameters
    ----------
    d
        The system which is to be reduced.
    """
    def __init__(self, d):
        self.d = d
        self.V = None
        self.W = None
        self.typ = None

    def _compute_gramians(self):
        """Returns low-rank factors of Gramians."""
        raise NotImplementedError()

    def _compute_sv_U_V(self):
        """Returns singular values and vectors."""
        U, sv, Vh = spla.svd(self.d.E.apply2(self.of, self.cf))
        self.sv, self.sU, self.sV = sv, U.T, Vh

    def _compute_error_bounds(self):
        """Returns error bounds for all possible reduced orders."""
        raise NotImplementedError()

    def reduce(self, r=None, tol=None, method='sr'):
        """Generic Balanced Truncation.

        Parameters
        ----------
        r
            Order of the reduced model if `tol` is `None`, maximum order
            if `tol` is specified.
        tol
            Tolerance for the error bound if `r` is `None`.
        method
            Projection method used:

                - `'sr'`: square root method (default, since standard in
                    literature)
                - `'bfsr'`: balancing-free square root method (avoids
                    scaling by singular values and orthogonalizes the
                    projection matrices, which might make it more
                    accurate than the square root method)
                - `'biorth'`: like the balancing-free square root
                    method, except it biorthogonalizes the projection
                    matrices

        Returns
        -------
        rd
            Reduced system.
        """
        assert r is not None or tol is not None
        assert r is None or 0 < r < self.d.n
        assert method in ('sr', 'bfsr', 'biorth')

        self._compute_gramians()
        self._compute_sv_U_V()

        # find reduced order if tol is specified
        if tol is not None:
            self._compute_error_bounds()
            r_tol = np.argmax(self.bounds <= tol) + 1
            r = r_tol if r is None else min([r, r_tol])

        if r > min([len(self.cf), len(self.of)]):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.'
                             ' Try reducing the tolerance in the low-rank matrix equation solver.')

        # compute projection matrices and find the reduced model
        self.V = self.cf.lincomb(self.sV[:r])
        self.W = self.of.lincomb(self.sU[:r])
        if method == 'sr':
            alpha = 1 / np.sqrt(self.sv[:r])
            self.V.scal(alpha)
            self.W.scal(alpha)
            self.use_default = ['E']
            rd = super().reduce()
        elif method == 'bfsr':
            self.V = gram_schmidt(self.V, atol=0, rtol=0)
            self.W = gram_schmidt(self.W, atol=0, rtol=0)
            self.use_default = None
            rd = super().reduce()
        elif method == 'biorth':
            self.V, self.W = gram_schmidt_biorth(self.V, self.W, product=self.d.E)
            self.use_default = ['E']
            rd = super().reduce()

        return rd

    extend_source_basis = None
    extend_range_basis = None


class BTReductor(GenericBTReductor):
    """Standard (Lyapunov) Balanced Truncation reductor.

    .. [A05] A. C. Antoulas, Approximation of Large-Scale Dynamical
             Systems,
             SIAM, 2005.

    Parameters
    ----------
    d
        The system which is to be reduced.
    """
    def __init__(self, d):
        super().__init__(d)

    def _compute_gramians(self):
        self.cf = self.d.gramian('cf')
        self.of = self.d.gramian('of')

    def _compute_error_bounds(self):
        self.bounds = 2 * self.sv[:0:-1].cumsum()[::-1]


class LQGBTReductor(GenericBTReductor):
    r"""Linear Quadratic Gaussian (LQG) Balanced Truncation reductor.

    .. [A05] A. C. Antoulas, Approximation of Large-Scale Dynamical
             Systems,
             SIAM, 2005.
    .. [MG91] D. Mustafa, K. Glover, Controller Reduction by
              :math:`\mathcal{H}_\infty`-Balanced Truncation,
              IEEE Transactions on Automatic Control, 36(6), 668-682,
              1991.

    Parameters
    ----------
    d
        The system which is to be reduced.
    """
    def __init__(self, d, solver_options=None):
        super().__init__(d)
        self.solver_options = solver_options

    @defaults('default_solver_backend', qualname='pymor.reductors.bt.LQGBTReductor._ricc_solver')
    def _ricc_solver(self, default_solver_backend=_DEFAULT_ME_SOLVER_BACKEND):
        options = self.solver_options.get('ricc') if self.solver_options else None
        if options:
            solver = options if isinstance(options, str) else options['type']
            backend = solver.split('_')[0]
        else:
            backend = default_solver_backend
        if backend == 'scipy':
            from pymor.bindings.scipy import solve_ricc as solve_ricc_impl
        elif backend == 'slycot':
            from pymor.bindings.slycot import solve_ricc as solve_ricc_impl
        elif backend == 'pymess':
            from pymor.bindings.pymess import solve_ricc as solve_ricc_impl
        else:
            raise NotImplementedError
        return partial(solve_ricc_impl, options=options)

    def _compute_gramians(self):
        A = self.d.A
        B = self.d.B
        C = self.d.C
        E = self.d.E if not isinstance(self.d.E, IdentityOperator) else None

        self.cf = self._ricc_solver()(A, E=E, B=B, C=C, trans=True)
        self.of = self._ricc_solver()(A, E=E, B=B, C=C, trans=False)

    def _compute_error_bounds(self):
        self.bounds = 2 * (self.sv[:0:-1] / np.sqrt(1 + self.sv[:0:-1] ** 2)).cumsum()[::-1]


_DEFAULT_BR_SOLVER_BACKEND = 'slycot' if config.HAVE_SLYCOT else 'scipy'


class BRBTReductor(GenericBTReductor):
    """Bounded Real (BR) Balanced Truncation reductor.

    .. [OJ88] P. C. Opdenacker, E. A. Jonckheere, A Contraction Mapping
              Preserving Balanced Reduction Scheme and Its Infinity Norm
              Error Bounds,
              IEEE Transactions on Circuits and Systems, 35(2), 184-189,
              1988.

    Parameters
    ----------
    d
        The system which is to be reduced.
    gamma
        Upper bound for the :math:`\mathcal{H}_\infty`-norm.
    """
    def __init__(self, d, gamma, solver_options=None):
        super().__init__(d)
        self.gamma = gamma
        self.solver_options = solver_options

    @defaults('default_solver_backend', qualname='pymor.reductors.bt.BRBTReductor._ricc_solver')
    def _ricc_solver(self, default_solver_backend=_DEFAULT_BR_SOLVER_BACKEND):
        options = self.solver_options.get('ricc') if self.solver_options else None
        if options:
            solver = options if isinstance(options, str) else options['type']
            backend = solver.split('_')[0]
        else:
            backend = default_solver_backend
        if backend == 'scipy':
            from pymor.bindings.scipy import solve_ricc as solve_ricc_impl
        elif backend == 'slycot':
            from pymor.bindings.slycot import solve_ricc as solve_ricc_impl
        elif backend == 'pymess':
            from pymor.bindings.pymess import solve_ricc as solve_ricc_impl
        else:
            raise NotImplementedError
        return partial(solve_ricc_impl, options=options)

    def _compute_gramians(self):
        A = self.d.A
        B = self.d.B
        C = self.d.C
        E = self.d.E if not isinstance(self.d.E, IdentityOperator) else None

        self.cf = self._ricc_solver()(A, E=E, B=B, C=C, R=IdentityOperator(C.range) * (-self.gamma ** 2), trans=True)
        self.of = self._ricc_solver()(A, E=E, B=B, C=C, R=IdentityOperator(B.source) * (-self.gamma ** 2), trans=False)

    def _compute_error_bounds(self):
        self.bounds = 2 * self.sv[:0:-1].cumsum()[::-1]
