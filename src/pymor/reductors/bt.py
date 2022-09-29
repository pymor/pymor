# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel
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

    def _gramians(self):
        """Return low-rank Cholesky factors of Gramians."""
        raise NotImplementedError

    def _sv_U_V(self):
        """Return singular values and vectors."""
        raise NotImplementedError

    def error_bounds(self):
        """Return error bounds for all possible reduced orders."""
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

    def _sv_U_V(self):
        return self.fom._sv_U_V(mu=self.mu)

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
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        if fom.sampling_time > 0:
            raise NotImplementedError
        super().__init__(fom, mu=mu)

    def _gramians(self):
        return self.fom.gramian('bs_c_lrcf', mu=self.mu), self.fom.gramian('bs_o_lrcf', mu=self.mu)

    def _sv_U_V(self):
        return self.fom._sv_U_V('bs', mu=self.mu)

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
    """

    def __init__(self, fom, mu=None):
        if fom.sampling_time > 0:
            raise NotImplementedError
        super().__init__(fom, mu=mu)

    def _gramians(self):
        return self.fom.gramian('lqg_c_lrcf', mu=self.mu), self.fom.gramian('lqg_o_lrcf', mu=self.mu)

    def _sv_U_V(self):
        return self.fom._sv_U_V('lqg', mu=self.mu)

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
    """

    def __init__(self, fom, gamma=1, mu=None):
        if fom.sampling_time > 0:
            raise NotImplementedError
        super().__init__(fom, mu=mu)
        self.gamma = gamma

    def _gramians(self):
        cf = self.fom.gramian(('br_c_lrcf', self.gamma), mu=self.mu)
        of = self.fom.gramian(('br_o_lrcf', self.gamma), mu=self.mu)
        return cf, of

    def _sv_U_V(self):
        return self.fom._sv_U_V(('br', self.gamma), mu=self.mu)

    def error_bounds(self):
        sv = self._sv_U_V()[0]
        return 2 * sv[:0:-1].cumsum()[::-1]
