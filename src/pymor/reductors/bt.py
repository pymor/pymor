# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.bernoulli import bernoulli_stabilize
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.lyapunov import solve_cont_lyap_lrcf
from pymor.algorithms.riccati import solve_ricc_lrcf, solve_pos_ricc_lrcf
from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel
from pymor.operators.constructions import IdentityOperator, LowRankOperator
from pymor.parameters.base import Mu
from pymor.reductors.basic import LTIPGReductor


class GenericBTReductor(BasicObject):
    """Generic Balanced Truncation reductor.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        assert isinstance(fom, LTIModel)
        if not isinstance(mu, Mu):
            mu = fom.parameters.parse(mu)
        assert fom.parameters.assert_compatible(mu)
        self.fom = fom
        self.mu = mu
        self.V = None
        self.W = None
        self._pg_reductor = None
        self._sv_U_V_cache = None

    def _gramians(self):
        """Return low-rank Cholesky factors of Gramians."""
        raise NotImplementedError

    def _sv_U_V(self):
        """Return singular values and vectors."""
        if self._sv_U_V_cache is None:
            cf, of = self._gramians()
            U, sv, Vh = spla.svd(self.fom.E.apply2(of, cf, mu=self.mu), lapack_driver='gesvd')
            self._sv_U_V_cache = (sv, U.T, Vh)
        return self._sv_U_V_cache

    def error_bounds(self):
        """Returns error bounds for all possible reduced orders."""
        raise NotImplementedError

    def reduce(self, r=None, tol=None, projection='bfsr'):
        """Generic Balanced Truncation.

        Parameters
        ----------
        r
            Order of the reduced model if `tol` is `None`, maximum order if `tol` is specified.
        tol
            Tolerance for the error bound if `r` is `None`.
        projection
            Projection method used:

            - `'sr'`: square root method
            - `'bfsr'`: balancing-free square root method (default, since it avoids scaling by
              singular values and orthogonalizes the projection matrices, which might make it more
              accurate than the square root method)
            - `'biorth'`: like the balancing-free square root method, except it biorthogonalizes the
              projection matrices (using :func:`~pymor.algorithms.gram_schmidt.gram_schmidt_biorth`)

        Returns
        -------
        rom
            Reduced-order model.
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

        # compute projection matrices
        self.V = cf.lincomb(sV[:r])
        self.W = of.lincomb(sU[:r])
        if projection == 'sr':
            alpha = 1 / np.sqrt(sv[:r])
            self.V.scal(alpha)
            self.W.scal(alpha)
        elif projection == 'bfsr':
            gram_schmidt(self.V, atol=0, rtol=0, copy=False)
            gram_schmidt(self.W, atol=0, rtol=0, copy=False)
        elif projection == 'biorth':
            gram_schmidt_biorth(self.V, self.W, product=self.fom.E, copy=False)

        # find reduced-order model
        if self.fom.parametric:
            fom_mu = self.fom.with_(**{op: getattr(self.fom, op).assemble(mu=self.mu)
                                       for op in ['A', 'B', 'C', 'D', 'E']})
        else:
            fom_mu = self.fom
        self._pg_reductor = LTIPGReductor(fom_mu, self.W, self.V, projection in ('sr', 'biorth'))
        rom = self._pg_reductor.reduce()
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)


class BTReductor(GenericBTReductor):
    """Standard (Lyapunov) Balanced Truncation reductor.

    See Section 7.3 in :cite:`A05`.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    mu
        |Parameter values|.
    """

    def _gramians(self):
        return self.fom.gramian('c_lrcf', mu=self.mu), self.fom.gramian('o_lrcf', mu=self.mu)

    def error_bounds(self):
        sv = self._sv_U_V()[0]
        return 2 * sv[:0:-1].cumsum()[::-1]


class FDBTReductor(GenericBTReductor):
    """Balanced Truncation reductor using frequency domain representation of Gramians.

    See :cite:`ZSW99`.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    ast_pole_data
        Can be:

        - dictionary of parameters for :func:`~pymor.algorithms.eigs.eigs`,
        - list of anti-stable eigenvalues (scalars),
        - tuple `(lev, ew, rev)` where `ew` contains the anti-stable eigenvalues
          and `lev` and `rev` are |VectorArrays| representing the eigenvectors.
        - `None` if anti-stable eigenvalues should be computed via dense methods.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, ast_pole_data=None, mu=None, solver_options=None):
        super().__init__(fom, mu=mu)
        self.ast_pole_data = ast_pole_data
        self.solver_options = solver_options

    def _gramians(self):
        if self.fom.sampling_time > 0:
            raise NotImplementedError

        A, B, C, E = (getattr(self.fom, op).assemble(mu=self.mu)
                      for op in ['A', 'B', 'C', 'E'])
        options = self.solver_options

        self.ast_spectrum = self.fom.get_ast_spectrum(self.ast_pole_data, mu=self.mu)
        K = bernoulli_stabilize(A, E, B.as_range_array(mu=self.mu), self.ast_spectrum, trans=True)
        BK = LowRankOperator(B.as_range_array(mu=self.mu), np.eye(len(K)), K)
        bsc_lrcf = solve_cont_lyap_lrcf(A-BK, E, B.as_range_array(mu=self.mu),
                                        trans=False, options=options)

        K = bernoulli_stabilize(A, E, C.as_source_array(mu=self.mu), self.ast_spectrum, trans=False)
        KC = LowRankOperator(K, np.eye(len(K)), C.as_source_array(mu=self.mu))
        bso_lrcf = solve_cont_lyap_lrcf(A-KC, E, C.as_source_array(mu=self.mu),
                                        trans=True, options=options)

        return bsc_lrcf, bso_lrcf

    def error_bounds(self):
        """L-infinity error bounds for reduced order models."""
        sv = self._sv_U_V()[0]
        return 2 * sv[:0:-1].cumsum()[::-1]


class LQGBTReductor(GenericBTReductor):
    r"""Linear Quadratic Gaussian (LQG) Balanced Truncation reductor.

    See Section 3 in :cite:`MG91`.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    mu
        |Parameter values|.
    solver_options
        The solver options to use to solve the Riccati equations.
    """

    def __init__(self, fom, mu=None, solver_options=None):
        super().__init__(fom, mu=mu)
        self.solver_options = solver_options

    def _gramians(self):
        if self.fom.sampling_time > 0:
            raise NotImplementedError

        A, B, C, E = (getattr(self.fom, op).assemble(mu=self.mu)
                      for op in ['A', 'B', 'C', 'E'])
        if isinstance(E, IdentityOperator):
            E = None
        options = self.solver_options

        cf = solve_ricc_lrcf(A, E, B.as_range_array(), C.as_source_array(),
                             trans=False, options=options)
        of = solve_ricc_lrcf(A, E, B.as_range_array(), C.as_source_array(),
                             trans=True, options=options)
        return cf, of

    def error_bounds(self):
        sv = self._sv_U_V()[0]
        return 2 * (sv[:0:-1] / np.sqrt(1 + sv[:0:-1]**2)).cumsum()[::-1]


class BRBTReductor(GenericBTReductor):
    r"""Bounded Real (BR) Balanced Truncation reductor.

    See :cite:`A05` (Section 7.5.3) and :cite:`OJ88`.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    gamma
        Upper bound for the :math:`\mathcal{H}_\infty`-norm.
    mu
        |Parameter values|.
    solver_options
        The solver options to use to solve the positive Riccati equations.
    """

    def __init__(self, fom, gamma=1, mu=None, solver_options=None):
        super().__init__(fom, mu=mu)
        self.gamma = gamma
        self.solver_options = solver_options

    def _gramians(self):
        if self.fom.sampling_time > 0:
            raise NotImplementedError

        A, B, C, E = (getattr(self.fom, op).assemble(mu=self.mu)
                      for op in ['A', 'B', 'C', 'E'])
        if isinstance(E, IdentityOperator):
            E = None
        options = self.solver_options

        cf = solve_pos_ricc_lrcf(A, E, B.as_range_array(), C.as_source_array(),
                                 R=self.gamma**2 * np.eye(self.fom.dim_output) if self.gamma != 1 else None,
                                 trans=False, options=options)
        of = solve_pos_ricc_lrcf(A, E, B.as_range_array(), C.as_source_array(),
                                 R=self.gamma**2 * np.eye(self.fom.dim_input) if self.gamma != 1 else None,
                                 trans=True, options=options)
        return cf, of

    def error_bounds(self):
        sv = self._sv_U_V()[0]
        return 2 * sv[:0:-1].cumsum()[::-1]
