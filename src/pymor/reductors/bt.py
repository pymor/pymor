# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.riccati import solve_ricc_lrcf, solve_pos_ricc_lrcf
from pymor.core.interfaces import BasicInterface
from pymor.models.iosys import LTIModel
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.basic import LTIPGReductor


class GenericBTReductor(BasicInterface):
    """Generic Balanced Truncation reductor.

    Parameters
    ----------
    fom
        The system which is to be reduced.
    """
    def __init__(self, fom):
        assert isinstance(fom, LTIModel)
        self.fom = fom
        self.V = None
        self.W = None
        self._sv = None
        self._sU = None
        self._sV = None

    def _gramians(self):
        """Return low-rank Cholesky factors of Gramians."""
        raise NotImplementedError

    def _sv_U_V(self):
        """Return singular values and vectors."""
        if self._sv is None or self._sU is None or self._sV is None:
            cf, of = self._gramians()
            U, sv, Vh = spla.svd(self.fom.E.apply2(of, cf), lapack_driver='gesvd')
            self._sv = sv
            self._sU = U.T
            self._sV = Vh
        return self._sv, self._sU, self._sV

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
              orthogonalizes the projection matrices, which might make
              it more accurate than the square root method)
            - `'biorth'`: like the balancing-free square root method,
              except it biorthogonalizes the projection matrices (using
              :func:`~pymor.algorithms.gram_schmidt.gram_schmidt_biorth`)

        Returns
        -------
        rom
            Reduced system.
        """
        assert r is not None or tol is not None
        assert r is None or 0 < r < self.fom.order
        assert projection in ('sr', 'bfsr', 'biorth')

        cf, of = self._gramians()
        sv, sU, sV = self._sv_U_V()

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
        elif projection == 'bfsr':
            self.V = gram_schmidt(self.V, atol=0, rtol=0)
            self.W = gram_schmidt(self.W, atol=0, rtol=0)
        elif projection == 'biorth':
            self.V, self.W = gram_schmidt_biorth(self.V, self.W, product=self.fom.E)

        self.pg_reductor = LTIPGReductor(self.fom, self.W, self.V, projection in ('sr', 'biorth'))
        rom = self.pg_reductor.reduce()

        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self.pg_reductor.reconstruct(u)


class BTReductor(GenericBTReductor):
    """Standard (Lyapunov) Balanced Truncation reductor.

    See Section 7.3 in [A05]_.

    Parameters
    ----------
    fom
        The system which is to be reduced.
    """
    def _gramians(self):
        return self.fom.gramian('c_lrcf'), self.fom.gramian('o_lrcf')

    def error_bounds(self):
        sv = self._sv_U_V()[0]
        return 2 * sv[:0:-1].cumsum()[::-1]


class LQGBTReductor(GenericBTReductor):
    r"""Linear Quadratic Gaussian (LQG) Balanced Truncation reductor.

    See Section 3 in [MG91]_.

    Parameters
    ----------
    fom
        The system which is to be reduced.
    solver_options
        The solver options to use to solve the Riccati equations.
    """
    def __init__(self, fom, solver_options=None):
        super().__init__(fom)
        self.solver_options = solver_options

    def _gramians(self):
        A = self.fom.A
        B = self.fom.B
        C = self.fom.C
        E = self.fom.E if not isinstance(self.fom.E, IdentityOperator) else None
        options = self.solver_options

        cf = solve_ricc_lrcf(A, E, B.as_range_array(), C.as_source_array(), trans=False, options=options)
        of = solve_ricc_lrcf(A, E, B.as_range_array(), C.as_source_array(), trans=True, options=options)
        return cf, of

    def error_bounds(self):
        sv = self._sv_U_V()[0]
        return 2 * (sv[:0:-1] / np.sqrt(1 + sv[:0:-1] ** 2)).cumsum()[::-1]


class BRBTReductor(GenericBTReductor):
    r"""Bounded Real (BR) Balanced Truncation reductor.

    See [A05]_ (Section 7.5.3) and [OJ88]_.

    Parameters
    ----------
    fom
        The system which is to be reduced.
    gamma
        Upper bound for the :math:`\mathcal{H}_\infty`-norm.
    solver_options
        The solver options to use to solve the positive Riccati equations.
    """
    def __init__(self, fom, gamma, solver_options=None):
        super().__init__(fom)
        self.gamma = gamma
        self.solver_options = solver_options

    def _gramians(self):
        fom = self.fom
        A = fom.A
        B = fom.B
        C = fom.C
        E = fom.E if not isinstance(fom.E, IdentityOperator) else None
        options = self.solver_options

        cf = solve_pos_ricc_lrcf(A, E, B.as_range_array(), C.as_source_array(),
                                 R=self.gamma**2 * np.eye(fom.output_dim),
                                 trans=False, options=options)
        of = solve_pos_ricc_lrcf(A, E, B.as_range_array(), C.as_source_array(),
                                 R=self.gamma**2 * np.eye(fom.input_dim),
                                 trans=True, options=options)
        return cf, of

    def error_bounds(self):
        sv = self._sv_U_V()[0]
        return 2 * sv[:0:-1].cumsum()[::-1]
