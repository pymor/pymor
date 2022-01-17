# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.samdp import samdp
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel, sparse_min_size
from pymor.operators.constructions import IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Mu
from pymor.reductors.basic import LTIPGReductor


class MTReductor(BasicObject):
    """Modal Truncation reductor.

    See Section 9.2 in :cite:`A05`.

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
               symmetric=False, which='NR', method_options=None, allow_complex_rom=False):
        """Modal Truncation.

        Parameters
        ----------
        r
            Order of the reduced model.
        decomposition
            Algorithm used for the decomposition:

            - `'eig'`: scipy.linalg.eig algorithm
            - `'samdp'`: find dominant poles using
              :meth:`~pymor.algorithms.samdp.samdp` algorithm
        projection
            Projection method used:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogolized with
              respect to the E product
        symmetric
            If `True`, assume A is symmetric and E is symmetric positive
            definite.
        which
            A string specifying which `r` eigenvalues and eigenvectors to
            compute.
            Possible values are:

            - `'SM'`: select eigenvalues with smallest magnitude (only for decomposition with eig)
            - `'LR'`: select eigenvalues with largest real part (only for decomposition with eig)
            - `'NR'`: select eigenvalues with largest norm(residual) / abs(Re(pole))
            - `'NS'`: select eigenvalues with largest norm(residual) / abs(pole)
            - `'NM'`: select eigenvalues with largest norm(residual)
        method_options
            Optional dict with more options for the samdp algorithm.
        allow_complex_rom
            If `True`, the reduced model is complex when the poles of the reduced
            model are not closed under complex conjugation.

        Returns
        -------
        rom
            Reduced-order model.
        """
        assert 0 < r < self.fom.order
        assert projection in ('orth', 'biorth')
        assert decomposition in ('eig', 'samdp')
        if decomposition == 'samdp':
            assert which in ('NR', 'NS', 'NM')
        else:
            assert which in ('LR', 'SM', 'NR', 'NS', 'NM')
        assert method_options is None or isinstance(method_options, dict)
        if not method_options:
            method_options = {'which': which}
        else:
            method_options['which'] = which

        if self.fom.parametric:
            fom = self.fom.with_(**{op: getattr(self.fom, op).assemble(mu=self.mu)
                                    for op in ['A', 'B', 'C', 'D', 'E']})
        else:
            fom = self.fom

        if decomposition == 'eig':
            if fom.order >= sparse_min_size():
                if not isinstance(fom.A, NumpyMatrixOperator) or fom.A.sparse:
                    self.logger.warning('Converting operator A to a NumPy array.')
                if not isinstance(fom.E, IdentityOperator):
                    if not isinstance(fom.E, NumpyMatrixOperator) or fom.E.sparse:
                        self.logger.warning('Converting operator E to a NumPy array.')
            A = to_matrix(fom.A, format='dense')
            E = None if isinstance(fom.E, IdentityOperator) else to_matrix(fom.E, format='dense')

            if symmetric:
                poles, ev_r = spla.eigh(A, E)
                rev = fom.A.source.from_numpy(ev_r.T)
                lev = rev.copy()
            else:
                poles, ev_l, ev_r = spla.eig(A, E, left=True)
                rev = fom.A.source.from_numpy(ev_r.T)
                lev = fom.A.source.from_numpy(ev_l.T)
            if which == 'SM':
                dominance = np.abs(poles)
            elif which == 'LR':
                dominance = -poles.real
            else:
                absres = np.empty(len(poles))
                for i in range(len(poles)):
                    lev[i].scal(1 / lev[i].inner(fom.E.apply(rev[i]))[0][0])
                    b_norm = fom.B.apply_adjoint(lev[i]).norm()[0]
                    c_norm = fom.C.apply(rev[i]).norm()[0]
                    absres[i] = b_norm * c_norm
                if which == 'NR':
                    dominance = -(absres / np.abs(np.real(poles)))
                elif which == 'NS':
                    dominance = -(absres / np.abs(poles))
                elif which == 'NM':
                    dominance = -np.array(absres)
        elif decomposition == 'samdp':
            poles, res, rev, lev = samdp(fom.A, fom.E,
                                         fom.B.as_range_array(),
                                         fom.C.as_source_array(),
                                         r, **method_options)
            absres = spla.norm(res, axis=(1, 2), ord=2)
            if method_options['which'] == 'NR':
                dominance = -(absres / np.abs(np.real(poles)))
            elif method_options['which'] == 'NS':
                dominance = -(absres / np.abs(poles))
            elif method_options['which'] == 'NM':
                dominance = -np.array(absres)
        idx = sorted(range(len(poles)),
                     key=lambda i: (dominance[i], -poles[i].real,
                                    abs(poles[i].imag), 0 if poles[i].imag >= 0 else 1))
        idx = idx[:r]
        poles = poles[idx]
        rev = rev[idx]
        lev = lev[idx]

        self.V = fom.A.source.empty(reserve=r)
        self.W = fom.A.source.empty(reserve=r)

        real_index = np.where(np.abs(poles.imag) / np.abs(poles) < 1e-6)[0]
        complex_index = np.where((np.abs(poles.imag) / np.abs(poles) >= 1e-6) & (poles.imag > 0))[0]

        if complex_index.size > 0 and complex_index[-1] == r-1:
            self.logger.warning('Chosen order r will split complex conjugated pair of poles.')
            if allow_complex_rom:
                self.logger.info('Reduced model will be complex.')
                self.V.append(rev[-1])
                self.W.append(lev[-1])
                complex_index = complex_index[:-1]
            else:
                self.logger.info('Only real part of complex conjugated pair taken.')
                self.V.append(rev[-1].real)
                self.W.append(lev[-1].real)
                complex_index = complex_index[:-1]

        self.V.append(rev[real_index].real)
        self.V.append(rev[complex_index].real)
        self.V.append(rev[complex_index].imag)

        self.W.append(lev[real_index].real)
        self.W.append(lev[complex_index].real)
        self.W.append(lev[complex_index].imag)

        if projection == 'orth':
            gram_schmidt(self.V, atol=0, rtol=0, copy=False)
            gram_schmidt(self.W, atol=0, rtol=0, copy=False)
        elif projection == 'biorth':
            gram_schmidt_biorth(self.V, self.W, product=fom.E, copy=False)

        # find reduced model
        self._pg_reductor = LTIPGReductor(fom, self.W, self.V, projection == 'biorth')
        rom = self._pg_reductor.reduce()
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)
