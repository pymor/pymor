# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.eigs import eigs
from pymor.algorithms.samdp import samdp
from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel
from pymor.parameters.base import Mu
from pymor.reductors.basic import LTIPGReductor
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth


class MTReductor(BasicObject):
    """Generic Modal Truncation reductor.

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
               symmetric=False, method_options=None):
        """Generic Modal Truncation.

        Parameters
        ----------
        r
            Order of the reduced model if `tol` is `None`, maximum order
            if `tol` is specified.
        decomposition
            Algortihm to use for the decomposition:

            -`'eigs'`: approximate eigenvalues with implicitly restarted arnoldi
            algorithm
            -`'samdp'`: find dominant poles
            projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogolized with
              respect to the E product
        symmetric
            True if Operator A is symmetric, False if not
        method_options
            optional dict with options for the method used to calculate eigenvalues
            and eigenvectors

        Returns
        -------
        rom
            Reduced-order model.
        """
        assert min(self.fom.B.source.dim, self.fom.C.range.dim) <= r < self.fom.order
        assert projection in ('orth', 'biorth')
        assert decomposition in ('eigs', 'samdp')
        assert method_options is None or isinstance(method_options, dict)
        if not method_options:
            method_options = {}

        if decomposition == 'eigs':
            if 'which' not in method_options.keys():
                method_options['which'] = 'SR'
            if symmetric:
                self.V = eigs(self.fom.A, k=r, **method_options)[1]
                self.W = self.V
            else:
                self.V = eigs(self.fom.A, k=r, **method_options)[1]
                self.W = eigs(self.fom.A.H, k=r, **method_options)[1]

        elif decomposition == 'samdp':

            poles, res, rev, lev = samdp(self.fom.A, self.fom.E,
                                         self.fom.B.as_range_array(),
                                         self.fom.C.as_source_array(),
                                         r, **method_options)
            if np.iscomplexobj(poles):
                self.V = rev.empty(reserve=r)
                self.W = lev.empty(reserve=r)

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
        self._pg_reductor = LTIPGReductor(fom_mu, self.W,
                                          self.V, projection == 'biorth')
        rom = self._pg_reductor.reduce()
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)
