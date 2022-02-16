# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.projection import project
from pymor.core.base import BasicObject
from pymor.models.iosys import SecondOrderModel
from pymor.operators.constructions import IdentityOperator
from pymor.parameters.base import Mu
from pymor.reductors.basic import SOLTIPGReductor
from pymor.vectorarrays.numpy import NumpyVectorSpace


class GenericSOBTpvReductor(BasicObject):
    """Generic Second-Order Balanced Truncation position/velocity reductor.

    See :cite:`RS08`.

    Parameters
    ----------
    fom
        The full-order |SecondOrderModel| to reduce.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        assert isinstance(fom, SecondOrderModel)
        if not isinstance(mu, Mu):
            mu = fom.parameters.parse(mu)
        assert fom.parameters.assert_compatible(mu)
        self.fom = fom
        self.mu = mu
        self.V = None
        self.W = None
        self._pg_reductor = None

    def _gramians(self):
        """Return Gramians."""
        raise NotImplementedError

    def _projection_matrices_and_singular_values(self, r, gramians):
        """Return projection matrices and singular values."""
        raise NotImplementedError

    def reduce(self, r, projection='bfsr'):
        """Reduce using GenericSOBTpv.

        Parameters
        ----------
        r
            Order of the reduced model.
        projection
            Projection method used:

            - `'sr'`: square root method
            - `'bfsr'`: balancing-free square root method (default, since it avoids scaling by
              singular values and orthogonalizes the projection matrices, which might make it more
              accurate than the square root method)
            - `'biorth'`: like the balancing-free square root method, except it biorthogonalizes the
              projection matrices

        Returns
        -------
        rom
            Reduced-order |SecondOrderModel|.
        """
        assert 0 < r < self.fom.order
        assert projection in ('sr', 'bfsr', 'biorth')

        # compute all necessary Gramian factors
        gramians = self._gramians()

        if r > min(len(g) for g in gramians):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.')

        # compute projection matrices
        self.V, self.W, singular_values = self._projection_matrices_and_singular_values(r, gramians)
        if projection == 'sr':
            alpha = 1 / np.sqrt(singular_values[:r])
            self.V.scal(alpha)
            self.W.scal(alpha)
        elif projection == 'bfsr':
            gram_schmidt(self.V, atol=0, rtol=0, copy=False)
            gram_schmidt(self.W, atol=0, rtol=0, copy=False)
        elif projection == 'biorth':
            gram_schmidt_biorth(self.V, self.W, product=self.fom.M, copy=False)

        # find the reduced model
        if self.fom.parametric:
            fom_mu = self.fom.with_(**{op: getattr(self.fom, op).assemble(mu=self.mu)
                                       for op in ['M', 'E', 'K', 'B', 'Cp', 'Cv']})
        else:
            fom_mu = self.fom
        self._pg_reductor = SOLTIPGReductor(fom_mu, self.W, self.V, projection == 'biorth')
        rom = self._pg_reductor.reduce()
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)


class SOBTpReductor(GenericSOBTpvReductor):
    """Second-Order Balanced Truncation position reductor.

    See :cite:`RS08`.

    Parameters
    ----------
    fom
        The full-order |SecondOrderModel| to reduce.
    mu
        |Parameter values|.
    """

    def _gramians(self):
        pcf = self.fom.gramian('pc_lrcf', mu=self.mu)
        pof = self.fom.gramian('po_lrcf', mu=self.mu)
        vcf = self.fom.gramian('vc_lrcf', mu=self.mu)
        vof = self.fom.gramian('vo_lrcf', mu=self.mu)
        return pcf, pof, vcf, vof

    def _projection_matrices_and_singular_values(self, r, gramians):
        pcf, pof, vcf, vof = gramians
        _, sp, Vp = spla.svd(pof.inner(pcf), lapack_driver='gesvd')
        Uv, _, _ = spla.svd(vof.inner(vcf, product=self.fom.M), lapack_driver='gesvd')
        Uv = Uv.T
        return pcf.lincomb(Vp[:r]), vof.lincomb(Uv[:r]), sp


class SOBTvReductor(GenericSOBTpvReductor):
    """Second-Order Balanced Truncation velocity reductor.

    See :cite:`RS08`.

    Parameters
    ----------
    fom
        The full-order |SecondOrderModel| to reduce.
    mu
        |Parameter values|.
    """

    def _gramians(self):
        vcf = self.fom.gramian('vc_lrcf', mu=self.mu)
        vof = self.fom.gramian('vo_lrcf', mu=self.mu)
        return vcf, vof

    def _projection_matrices_and_singular_values(self, r, gramians):
        vcf, vof = gramians
        Uv, sv, Vv = spla.svd(vof.inner(vcf, product=self.fom.M), lapack_driver='gesvd')
        Uv = Uv.T
        return vcf.lincomb(Vv[:r]), vof.lincomb(Uv[:r]), sv


class SOBTpvReductor(GenericSOBTpvReductor):
    """Second-Order Balanced Truncation position-velocity reductor.

    See :cite:`RS08`.

    Parameters
    ----------
    fom
        The full-order |SecondOrderModel| to reduce.
    mu
        |Parameter values|.
    """

    def _gramians(self):
        pcf = self.fom.gramian('pc_lrcf', mu=self.mu)
        vof = self.fom.gramian('vo_lrcf', mu=self.mu)
        return pcf, vof

    def _projection_matrices_and_singular_values(self, r, gramians):
        pcf, vof = gramians
        Upv, spv, Vpv = spla.svd(vof.inner(pcf, product=self.fom.M), lapack_driver='gesvd')
        Upv = Upv.T
        return pcf.lincomb(Vpv[:r]), vof.lincomb(Upv[:r]), spv


class SOBTvpReductor(GenericSOBTpvReductor):
    """Second-Order Balanced Truncation velocity-position reductor.

    See :cite:`RS08`.

    Parameters
    ----------
    fom
        The full-order |SecondOrderModel| to reduce.
    mu
        |Parameter values|.
    """

    def _gramians(self):
        pof = self.fom.gramian('po_lrcf', mu=self.mu)
        vcf = self.fom.gramian('vc_lrcf', mu=self.mu)
        vof = self.fom.gramian('vo_lrcf', mu=self.mu)
        return pof, vcf, vof

    def _projection_matrices_and_singular_values(self, r, gramians):
        pof, vcf, vof = gramians
        Uv, _, _ = spla.svd(vof.inner(vcf, product=self.fom.M), lapack_driver='gesvd')
        Uv = Uv.T
        _, svp, Vvp = spla.svd(pof.inner(vcf), lapack_driver='gesvd')
        return vcf.lincomb(Vvp[:r]), vof.lincomb(Uv[:r]), svp


class SOBTfvReductor(BasicObject):
    """Free-velocity Second-Order Balanced Truncation reductor.

    See :cite:`MS96`.

    Parameters
    ----------
    fom
        The full-order |SecondOrderModel| to reduce.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        assert isinstance(fom, SecondOrderModel)
        if not isinstance(mu, Mu):
            mu = fom.parameters.parse(mu)
        assert fom.parameters.assert_compatible(mu)
        self.fom = fom
        self.mu = mu
        self.V = None
        self.W = None
        self._pg_reductor = None

    def reduce(self, r, projection='bfsr'):
        """Reduce using SOBTfv.

        Parameters
        ----------
        r
            Order of the reduced model.
        projection
            Projection method used:

            - `'sr'`: square root method
            - `'bfsr'`: balancing-free square root method (default, since it avoids scaling by
              singular values and orthogonalizes the projection matrices, which might make it more
              accurate than the square root method)
            - `'biorth'`: like the balancing-free square root method, except it biorthogonalizes the
              projection matrices

        Returns
        -------
        rom
            Reduced-order |SecondOrderModel|.
        """
        assert 0 < r < self.fom.order
        assert projection in ('sr', 'bfsr', 'biorth')

        # compute all necessary Gramian factors
        pcf = self.fom.gramian('pc_lrcf', mu=self.mu)
        pof = self.fom.gramian('po_lrcf', mu=self.mu)

        if r > min(len(pcf), len(pof)):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.')

        # find necessary SVDs
        _, sp, Vp = spla.svd(pof.inner(pcf), lapack_driver='gesvd')

        # compute projection matrices
        self.V = pcf.lincomb(Vp[:r])
        if projection == 'sr':
            alpha = 1 / np.sqrt(sp[:r])
            self.V.scal(alpha)
        elif projection == 'bfsr':
            gram_schmidt(self.V, atol=0, rtol=0, copy=False)
        elif projection == 'biorth':
            gram_schmidt(self.V, product=self.fom.M, atol=0, rtol=0, copy=False)
        self.W = self.V

        # find the reduced model
        if self.fom.parametric:
            fom_mu = self.fom.with_(**{op: getattr(self.fom, op).assemble(mu=self.mu)
                                       for op in ['M', 'E', 'K', 'B', 'Cp', 'Cv']})
        else:
            fom_mu = self.fom
        self._pg_reductor = SOLTIPGReductor(fom_mu, self.W, self.V, projection == 'biorth')
        rom = self._pg_reductor.reduce()
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)


class SOBTReductor(BasicObject):
    """Second-Order Balanced Truncation reductor.

    See :cite:`CLVV06`.

    Parameters
    ----------
    fom
        The full-order |SecondOrderModel| to reduce.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        assert isinstance(fom, SecondOrderModel)
        if not isinstance(mu, Mu):
            mu = fom.parameters.parse(mu)
        assert fom.parameters.assert_compatible(mu)
        self.fom = fom
        self.mu = mu
        self.V1 = None
        self.W1 = None
        self.V2 = None
        self.W2 = None

    def reduce(self, r, projection='bfsr'):
        """Reduce using SOBT.

        Parameters
        ----------
        r
            Order of the reduced model.
        projection
            Projection method used:

            - `'sr'`: square root method
            - `'bfsr'`: balancing-free square root method (default, since it avoids scaling by
              singular values and orthogonalizes the projection matrices, which might make it more
              accurate than the square root method)
            - `'biorth'`: like the balancing-free square root method, except it biorthogonalizes the
              projection matrices

        Returns
        -------
        rom
            Reduced-order |SecondOrderModel|.
        """
        assert 0 < r < self.fom.order
        assert projection in ('sr', 'bfsr', 'biorth')

        # compute all necessary Gramian factors
        pcf = self.fom.gramian('pc_lrcf', mu=self.mu)
        pof = self.fom.gramian('po_lrcf', mu=self.mu)
        vcf = self.fom.gramian('vc_lrcf', mu=self.mu)
        vof = self.fom.gramian('vo_lrcf', mu=self.mu)

        if r > min(len(pcf), len(pof), len(vcf), len(vof)):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.')

        # find necessary SVDs
        Up, sp, Vp = spla.svd(pof.inner(pcf), lapack_driver='gesvd')
        Up = Up.T
        Uv, sv, Vv = spla.svd(vof.inner(vcf, product=self.fom.M), lapack_driver='gesvd')
        Uv = Uv.T

        # compute projection matrices and find the reduced model
        self.V1 = pcf.lincomb(Vp[:r])
        self.W1 = pof.lincomb(Up[:r])
        self.V2 = vcf.lincomb(Vv[:r])
        self.W2 = vof.lincomb(Uv[:r])
        if projection == 'sr':
            alpha1 = 1 / np.sqrt(sp[:r])
            self.V1.scal(alpha1)
            self.W1.scal(alpha1)
            alpha2 = 1 / np.sqrt(sv[:r])
            self.V2.scal(alpha2)
            self.W2.scal(alpha2)
            W1TV1invW1TV2 = self.W1.inner(self.V2)
            projected_ops = {'M': IdentityOperator(NumpyVectorSpace(r))}
        elif projection == 'bfsr':
            gram_schmidt(self.V1, atol=0, rtol=0, copy=False)
            gram_schmidt(self.W1, atol=0, rtol=0, copy=False)
            gram_schmidt(self.V2, atol=0, rtol=0, copy=False)
            gram_schmidt(self.W2, atol=0, rtol=0, copy=False)
            W1TV1invW1TV2 = spla.solve(self.W1.inner(self.V1), self.W1.inner(self.V2))
            projected_ops = {'M': project(self.fom.M, range_basis=self.W2, source_basis=self.V2)}
        elif projection == 'biorth':
            gram_schmidt_biorth(self.V1, self.W1, copy=False)
            gram_schmidt_biorth(self.V2, self.W2, product=self.fom.M, copy=False)
            W1TV1invW1TV2 = self.W1.inner(self.V2)
            projected_ops = {'M': IdentityOperator(NumpyVectorSpace(r))}

        projected_ops.update({
            'E': project(self.fom.E.assemble(mu=self.mu),
                         range_basis=self.W2,
                         source_basis=self.V2),
            'K': project(self.fom.K.assemble(mu=self.mu),
                         range_basis=self.W2,
                         source_basis=self.V1.lincomb(W1TV1invW1TV2.T)),
            'B': project(self.fom.B.assemble(mu=self.mu),
                         range_basis=self.W2,
                         source_basis=None),
            'Cp': project(self.fom.Cp.assemble(mu=self.mu),
                          range_basis=None,
                          source_basis=self.V1.lincomb(W1TV1invW1TV2.T)),
            'Cv': project(self.fom.Cv.assemble(mu=self.mu),
                          range_basis=None,
                          source_basis=self.V2),
            'D': self.fom.D.assemble(mu=self.mu),
        })

        rom = SecondOrderModel(name=self.fom.name + '_reduced', **projected_ops)
        rom.disable_logging()
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        raise TypeError(f'The reconstruct method is not available for {self.__class__.__name__}.')
