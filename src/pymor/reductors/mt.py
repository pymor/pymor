# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.samdp import samdp
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel
from pymor.operators.constructions import IdentityOperator
from pymor.parameters.base import Mu
from pymor.reductors.basic import LTIPGReductor


class MTReductor(BasicObject):
    """Modal Truncation reductor.

    See Section 9.2 in [A05]_.

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

    def reduce(self, r=None, decomposition='samdp', projection='orth',
               symmetric=False, which="LR", method_options=None):
        """Modal Truncation.

        Parameters
        ----------
        r
            Order of the reduced model.
        decomposition
            Algortihm to use for the decomposition:

            - `'eig'`: use scipy.linalg.eig algorithm
            - `'samdp'`: find dominant poles using samdp algorithm
        projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogolized with
              respect to the E product
        symmetric
            True if Operator A is symmetric and E is symmetric, positive
            definite, False if not.
        method_options
            Optional dict with more options for the samdp algorithm.
        which
            A string specifying which `k` eigenvalues and eigenvectors to
            compute when using the eig decomposition, default is "LR":

            - `'SM'`: select eigenvalues with smallest magnitude
            - `'LR'`: select eigenvalues with largest real part
            - `'NR'`: select eigenvalues with largest norm(residual) / abs(Re(pole))
            - `'NS'`: select eigenvalues with largest norm(residual) / abs(pole)
            - `'NM'`: select eigenvalues with largest norm(residual)
             
        Returns
        -------
        rom
            Reduced-order model.
        """
        assert 0 < r < self.fom.order
        assert projection in ('orth', 'biorth')
        assert decomposition in ('eig', 'samdp')
        assert method_options is None or isinstance(method_options, dict)
        if not method_options:
            method_options = {}

        self.V = self.fom.B.as_range_array().empty(reserve=r)
        self.W = self.fom.C.as_source_array().empty(reserve=r)

        if decomposition == 'eig':
            if self.fom.A.sparse:
                self.logger.warning('Converting operator A to a NumPy array.')
            A = to_matrix(self.fom.A, format='dense')
            if not isinstance(self.fom.E, IdentityOperator):
                if self.fom.E.sparse:
                    self.logger.warning('Converting operator E to a NumPy array.')
            E = None if isinstance(self.fom.E, IdentityOperator) else to_matrix(self.fom.E, format='dense')

            if symmetric:
                poles, ev_r = spla.eig(A, E, right=True)
                rev = self.fom.A.source.from_numpy(ev_r.T)
                if which == 'SM':
                    idx = np.argsort(np.abs(poles))
                elif which == 'LR':
                    idx = np.argsort(-poles.real)
                else:
                    absres = np.empty(len(poles))
                    for i in range(len(poles)):
                        b_norm = spla.norm(rev[i].inner(self.fom.B.as_range_array()), ord=2)
                        c_norm = spla.norm(self.fom.C.as_source_array().inner(rev[i]), ord=2)
                        absres[i] = b_norm * c_norm
                    if which == 'NR':
                        idx = np.argsort(-absres / np.abs(np.real(poles)))
                    elif which == 'NS':
                        idx = np.argsort(-absres / np.abs(poles))
                    elif which == 'NM':
                        idx = np.argsort(-absres)
                rev = rev[idx]
                poles = poles[:r]
                rev = rev[:r]
                lev = rev.copy()
            else:
                poles, ev_l, ev_r = spla.eig(A, E, left=True, right=True)
                rev = self.fom.A.source.from_numpy(ev_r.T)
                lev = self.fom.A.source.from_numpy(ev_l.T)
                if which == 'SM':
                    idx = np.argsort(np.abs(poles))
                elif which == 'LR':
                    idx = np.argsort(-poles.real)
                else:
                    absres = np.empty(len(poles))
                    for i in range(len(poles)):
                        b_norm = spla.norm(lev[i].inner(self.fom.B.as_range_array()), ord=2)
                        c_norm = spla.norm(self.fom.C.as_source_array().inner(rev[i]), ord=2)
                        absres[i] = b_norm * c_norm
                    if which == 'NR':
                        idx = np.argsort(-absres / np.abs(np.real(poles)))
                    elif which == 'NS':
                        idx = np.argsort(-absres / np.abs(poles))
                    elif which == 'NM':
                        idx = np.argsort(-absres)
                poles = poles[idx]
                rev = rev[idx]
                lev = lev[idx]
                poles = poles[:r]
                rev = rev[:r]
                lev = lev[:r]
        elif decomposition == 'samdp':
            poles, res, rev, lev = samdp(self.fom.A, self.fom.E,
                                         self.fom.B.as_range_array(),
                                         self.fom.C.as_source_array(),
                                         r, **method_options)
        if np.iscomplexobj(poles):
            real_index = np.where(np.isreal(poles))[0]
            complex_index = np.where(poles.imag > 0)[0]

            self.V.append(rev[real_index].real)
            self.V.append(rev[complex_index].real)
            self.V.append(rev[complex_index].imag)

            self.W.append(lev[real_index].real)
            self.W.append(lev[complex_index].real)
            self.W.append(lev[complex_index].imag)

        else:
            self.V = rev
            self.W = lev

        if projection == 'orth':
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
        self._pg_reductor = LTIPGReductor(fom_mu, self.W, self.V, projection == 'biorth')
        rom = self._pg_reductor.reduce()
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)
