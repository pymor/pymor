# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from functools import partial

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.discretizations.iosys import _DEFAULT_ME_SOLVER_BACKEND, LTISystem
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
        assert isinstance(d, LTISystem)
        self.d = d
        self.V = None
        self.W = None
        self.sv = None
        self.sU = None
        self.sV = None

    def gramians(self):
        """Returns low-rank factors of Gramians."""
        raise NotImplementedError

    def sv_U_V(self):
        """Returns singular values and vectors."""
        if self.sv is None or self.sU is None or self.sV is None:
            cf, of = self.gramians()
            U, sv, Vh = spla.svd(self.d.E.apply2(of, cf), lapack_driver='gesvd')
            self.sv = sv
            self.sU = U.T
            self.sV = Vh
        return self.sv, self.sU, self.sV

    def error_bounds(self):
        """Returns error bounds for all possible reduced orders."""
        raise NotImplementedError

    def reduce(self, r=None, tol=None, projection='bfsr'):
        """Generic Balanced Truncation.

        Parameters
        ----------
        r
            Order of the reduced model if `tol` is `None`, maximum order
            if `tol` is specified.
        tol
            Tolerance for the error bound if `r` is `None`.
        projection
            Projection method used:

                - `'sr'`: square root method
                - `'bfsr'`: balancing-free square root method (default,
                    since it avoids scaling by singular values and
                    orthogonalizes the projection matrices, which might
                    make it more accurate than the square root method)
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
        assert projection in ('sr', 'bfsr', 'biorth')

        cf, of = self.gramians()
        sv, sU, sV = self.sv_U_V()

        # find reduced order if tol is specified
        if tol is not None:
            error_bounds = self.error_bounds()
            r_tol = np.argmax(error_bounds <= tol) + 1
            r = r_tol if r is None else min(r, r_tol)

        if r > min(len(cf), len(of)):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.')

        # compute projection matrices and find the reduced model
        self.V = cf.lincomb(sV[:r])
        self.W = of.lincomb(sU[:r])
        if projection == 'sr':
            alpha = 1 / np.sqrt(sv[:r])
            self.V.scal(alpha)
            self.W.scal(alpha)
            self.biorthogonal_product = 'E'
        elif projection == 'bfsr':
            self.V = gram_schmidt(self.V, atol=0, rtol=0)
            self.W = gram_schmidt(self.W, atol=0, rtol=0)
            self.biorthogonal_product = None
        elif projection == 'biorth':
            self.V, self.W = gram_schmidt_biorth(self.V, self.W, product=self.d.E)
            self.biorthogonal_product = 'E'

        rd = super().reduce()

        return rd


class BTReductor(GenericBTReductor):
    """Standard (Lyapunov) Balanced Truncation reductor.

    See Section 7.3 in [A05]_.

    Parameters
    ----------
    d
        The system which is to be reduced.
    """
    def gramians(self):
        return self.d.gramian('cf'), self.d.gramian('of')

    def error_bounds(self):
        sv = self.sv_U_V()[0]
        return 2 * sv[:0:-1].cumsum()[::-1]


class LQGBTReductor(GenericBTReductor):
    r"""Linear Quadratic Gaussian (LQG) Balanced Truncation reductor.

    See Section 3 in [MG91]_.

    Parameters
    ----------
    d
        The system which is to be reduced.
    """
    def __init__(self, d, solver_options=None):
        super().__init__(d)
        self.solver_options = solver_options

    @defaults('default_solver_backend')
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

    def gramians(self):
        A = self.d.A
        B = self.d.B
        C = self.d.C
        E = self.d.E if not isinstance(self.d.E, IdentityOperator) else None

        cf = self._ricc_solver()(A, E=E, B=B, C=C, trans=True)
        of = self._ricc_solver()(A, E=E, B=B, C=C, trans=False)

        return cf, of

    def error_bounds(self):
        sv = self.sv_U_V()[0]
        return 2 * (sv[:0:-1] / np.sqrt(1 + sv[:0:-1] ** 2)).cumsum()[::-1]


_DEFAULT_BR_SOLVER_BACKEND = 'slycot' if config.HAVE_SLYCOT else 'scipy'


class BRBTReductor(GenericBTReductor):
    """Bounded Real (BR) Balanced Truncation reductor.

    See [A05]_ (Section 7.5.3) and [OJ88]_.

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

    @defaults('default_solver_backend')
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

    def gramians(self):
        A = self.d.A
        B = self.d.B
        C = self.d.C
        E = self.d.E if not isinstance(self.d.E, IdentityOperator) else None

        cf = self._ricc_solver()(A, E=E, B=B, C=C, R=IdentityOperator(C.range) * (-self.gamma ** 2), trans=True)
        of = self._ricc_solver()(A, E=E, B=B, C=C, R=IdentityOperator(B.source) * (-self.gamma ** 2), trans=False)

        return cf, of

    def error_bounds(self):
        sv = self.sv_U_V()[0]
        return 2 * sv[:0:-1].cumsum()[::-1]
