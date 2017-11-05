# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
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
        self.cf = self.d.gramian(self.typ, 'cf')
        self.of = self.d.gramian(self.typ, 'of')

    def _compute_sv_U_V(self):
        """Returns singular values and vectors."""
        self.sv, self.sU, self.sV = self.d.sv_U_V(self.typ)

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
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.' +
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
        self.typ = 'lyap'

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
    def __init__(self, d):
        super().__init__(d)
        self.typ = 'lqg'

    def _compute_error_bounds(self):
        self.bounds = 2 * (self.sv[:0:-1] / np.sqrt(1 + self.sv[:0:-1] ** 2)).cumsum()[::-1]


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
    def __init__(self, d, gamma):
        super().__init__(d)
        self.typ = ('br', gamma)
        self.gamma = gamma

    def _compute_error_bounds(self, sv):
        self.bounds = 2 * self.gamma * self.sv[:0:-1].cumsum()[::-1]
