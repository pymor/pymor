# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.projection import project
from pymor.discretizations.iosys import SecondOrderSystem
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.basic import GenericPGReductor
from pymor.vectorarrays.numpy import NumpyVectorSpace


class SOBTpReductor(GenericPGReductor):
    """Second-Order Balanced Truncation position reductor.

    .. [RS08] T. Reis and T. Stykel,
              Balanced truncation model reduction of second-order
              systems,
              Math. Comput. Model. Dyn. Syst., 2008, 14(5), 391-406

    Parameters
    ----------
    d
        The system which is to be reduced.
    """
    def __init__(self, d):
        assert isinstance(d, SecondOrderSystem)
        self.d = d
        self.V = None
        self.W = None

    def reduce(self, r, projection='sr'):
        """Reduce using SOBTp.

        Parameters
        ----------
        r
            Order of the reduced model.
        projection
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
        assert 0 < r < self.d.n
        assert projection in ('sr', 'bfsr', 'biorth')

        # compute all necessary Gramian factors
        pcf = self.d.gramian('pcf')
        pof = self.d.gramian('pof')
        vcf = self.d.gramian('vcf')
        vof = self.d.gramian('vof')

        if r > min([len(pcf), len(pof), len(vcf), len(vof)]):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.'
                             ' Try reducing the tolerance in the low-rank matrix equation solver.')

        # find necessary SVDs
        _, sp, Vp = spla.svd(pof.inner(pcf))
        Uv, _, _ = spla.svd(vof.inner(vcf, product=self.d.M))
        Uv = Uv.T

        # compute projection matrices and find the reduced model
        self.V = pcf.lincomb(Vp[:r])
        self.W = vof.lincomb(Uv[:r])
        if projection == 'sr':
            alpha = 1 / np.sqrt(sp[:r])
            self.V.scal(alpha)
            self.W.scal(alpha)
            self.biorthogonal_product = None
        elif projection == 'bfsr':
            self.V = gram_schmidt(self.V, atol=0, rtol=0)
            self.W = gram_schmidt(self.W, atol=0, rtol=0)
            self.biorthogonal_product = None
        elif projection == 'biorth':
            self.V, self.W = gram_schmidt_biorth(self.V, self.W, product=self.d.M)
            self.biorthogonal_product = 'M'

        rd = super().reduce()

        return rd

    extend_source_basis = None
    extend_range_basis = None


class SOBTvReductor(GenericPGReductor):
    """Second-Order Balanced Truncation velocity reductor.

    .. [RS08] T. Reis and T. Stykel,
              Balanced truncation model reduction of second-order
              systems,
              Math. Comput. Model. Dyn. Syst., 2008, 14(5), 391-406

    Parameters
    ----------
    d
        The system which is to be reduced.
    """
    def __init__(self, d):
        assert isinstance(d, SecondOrderSystem)
        self.d = d
        self.V = None
        self.W = None

    def reduce(self, r, projection='sr'):
        """Reduce using SOBTv.

        Parameters
        ----------
        r
            Order of the reduced model.
        projection
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
        assert 0 < r < self.d.n
        assert projection in ('sr', 'bfsr', 'biorth')

        # compute all necessary Gramian factors
        vcf = self.d.gramian('vcf')
        vof = self.d.gramian('vof')

        if r > min([len(vcf), len(vof)]):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.'
                             ' Try reducing the tolerance in the low-rank matrix equation solver.')

        # find necessary SVDs
        Uv, sv, Vv = spla.svd(vof.inner(vcf, product=self.d.M))
        Uv = Uv.T

        # compute projection matrices and find the reduced model
        self.V = vcf.lincomb(Vv[:r])
        self.W = vof.lincomb(Uv[:r])
        if projection == 'sr':
            alpha = 1 / np.sqrt(sv[:r])
            self.V.scal(alpha)
            self.W.scal(alpha)
            self.biorthogonal_product = 'M'
        elif projection == 'bfsr':
            self.V = gram_schmidt(self.V, atol=0, rtol=0)
            self.W = gram_schmidt(self.W, atol=0, rtol=0)
            self.biorthogonal_product = None
        elif projection == 'biorth':
            self.V, self.W = gram_schmidt_biorth(self.V, self.W, product=self.d.M)
            self.biorthogonal_product = 'M'

        rd = super().reduce()

        return rd

    extend_source_basis = None
    extend_range_basis = None


class SOBTpvReductor(GenericPGReductor):
    """Second-Order Balanced Truncation position-velocity reductor.

    .. [RS08] T. Reis and T. Stykel,
              Balanced truncation model reduction of second-order
              systems,
              Math. Comput. Model. Dyn. Syst., 2008, 14(5), 391-406

    Parameters
    ----------
    d
        The system which is to be reduced.
    """
    def __init__(self, d):
        assert isinstance(d, SecondOrderSystem)
        self.d = d
        self.V = None
        self.W = None

    def reduce(self, r, projection='sr'):
        """Reduce using SOBTpv.

        Parameters
        ----------
        r
            Order of the reduced model.
        projection
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
        assert 0 < r < self.d.n
        assert projection in ('sr', 'bfsr', 'biorth')

        # compute all necessary Gramian factors
        pcf = self.d.gramian('pcf')
        vof = self.d.gramian('vof')

        if r > min([len(pcf), len(vof)]):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.'
                             ' Try reducing the tolerance in the low-rank matrix equation solver.')

        # find necessary SVDs
        Upv, spv, Vpv = spla.svd(vof.inner(pcf, product=self.d.M))
        Upv = Upv.T

        # compute projection matrices and find the reduced model
        self.V = pcf.lincomb(Vpv[:r])
        self.W = vof.lincomb(Upv[:r])
        if projection == 'sr':
            alpha = 1 / np.sqrt(spv[:r])
            self.V.scal(alpha)
            self.W.scal(alpha)
            self.biorthogonal_product = 'M'
        elif projection == 'bfsr':
            self.V = gram_schmidt(self.V, atol=0, rtol=0)
            self.W = gram_schmidt(self.W, atol=0, rtol=0)
            self.biorthogonal_product = None
        elif projection == 'biorth':
            self.V, self.W = gram_schmidt_biorth(self.V, self.W, product=self.d.M)
            self.biorthogonal_product = 'M'

        rd = super().reduce()

        return rd

    extend_source_basis = None
    extend_range_basis = None


class SOBTvpReductor(GenericPGReductor):
    """Second-Order Balanced Truncation velocity-position reductor.

    .. [RS08] T. Reis and T. Stykel,
              Balanced truncation model reduction of second-order
              systems,
              Math. Comput. Model. Dyn. Syst., 2008, 14(5), 391-406

    Parameters
    ----------
    d
        The system which is to be reduced.
    """
    def __init__(self, d):
        assert isinstance(d, SecondOrderSystem)
        self.d = d
        self.V = None
        self.W = None

    def reduce(self, r, projection='sr'):
        """Reduce using SOBTvp.

        Parameters
        ----------
        r
            Order of the reduced model.
        projection
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
        assert 0 < r < self.d.n
        assert projection in ('sr', 'bfsr', 'biorth')

        # compute all necessary Gramian factors
        pof = self.d.gramian('pof')
        vcf = self.d.gramian('vcf')
        vof = self.d.gramian('vof')

        if r > min([len(pof), len(vcf), len(vof)]):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.'
                             ' Try reducing the tolerance in the low-rank matrix equation solver.')

        # find necessary SVDs
        Uv, _, _ = spla.svd(vof.inner(vcf, product=self.d.M))
        Uv = Uv.T
        _, svp, Vvp = spla.svd(pof.inner(vcf))

        # compute projection matrices and find the reduced model
        self.V = vcf.lincomb(Vvp[:r])
        self.W = vof.lincomb(Uv[:r])
        if projection == 'sr':
            alpha = 1 / np.sqrt(svp[:r])
            self.V.scal(alpha)
            self.W.scal(alpha)
            self.biorthogonal_product = None
        elif projection == 'bfsr':
            self.V = gram_schmidt(self.V, atol=0, rtol=0)
            self.W = gram_schmidt(self.W, atol=0, rtol=0)
            self.biorthogonal_product = None
        elif projection == 'biorth':
            self.V, self.W = gram_schmidt_biorth(self.V, self.W, product=self.d.M)
            self.biorthogonal_product = 'M'

        rd = super().reduce()

        return rd

    extend_source_basis = None
    extend_range_basis = None


class SOBTfvReductor(GenericPGReductor):
    """Free-velocity Second-Order Balanced Truncation reductor.

    .. [MS96] D. G. Meyer and S. Srinivasan,
              Balancing and model reduction for second-order form linear
              systems,
              IEEE Trans. Automat. Control, 1996, 41, 1632–1644

    Parameters
    ----------
    d
        The system which is to be reduced.
    """
    def __init__(self, d):
        assert isinstance(d, SecondOrderSystem)
        self.d = d
        self.V = None
        self.W = None

    def reduce(self, r, projection='sr'):
        """Reduce using SOBTfv.

        Parameters
        ----------
        r
            Order of the reduced model.
        projection
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
        assert 0 < r < self.d.n
        assert projection in ('sr', 'bfsr', 'biorth')

        # compute all necessary Gramian factors
        pcf = self.d.gramian('pcf')
        pof = self.d.gramian('pof')

        if r > min([len(pcf), len(pof)]):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.'
                             ' Try reducing the tolerance in the low-rank matrix equation solver.')

        # find necessary SVDs
        _, sp, Vp = spla.svd(pof.inner(pcf))

        # compute projection matrices and find the reduced model
        self.V = pcf.lincomb(Vp[:r])
        if projection == 'sr':
            alpha = 1 / np.sqrt(sp[:r])
            self.V.scal(alpha)
            self.biorthogonal_product = None
        elif projection == 'bfsr':
            self.V = gram_schmidt(self.V, atol=0, rtol=0)
            self.biorthogonal_product = None
        elif projection == 'biorth':
            self.V = gram_schmidt(self.V, product=self.d.M, atol=0, rtol=0)
            self.biorthogonal_product = 'M'

        self.W = self.V

        rd = super().reduce()

        return rd

    extend_source_basis = None
    extend_range_basis = None


class SOBTReductor():
    """Second-Order Balanced Truncation reductor.

        - SOBT [CLVV06]_

    .. [CLVV06] Y. Chahlaoui, D. Lemonnier, A. Vandendorpe, P. Van
                Dooren,
                Second-order balanced truncation,
                Linear Algebra and its Applications, 2006, 415(2–3),
                373-384

    Parameters
    ----------
    d
        The system which is to be reduced.
    """
    def __init__(self, d):
        assert isinstance(d, SecondOrderSystem)
        self.d = d
        self.V1 = None
        self.W1 = None
        self.V2 = None
        self.W2 = None

    def reduce(self, r, projection='sr'):
        """Reduce using SOBT.

        Parameters
        ----------
        r
            Order of the reduced model.
        projection
            Projection method used:

                - `'sr'`: square root method
                - `'bfsr'`: balancing-free square root method
                - `'biorth'`: biorthogonalization

        Returns
        -------
        rd
            Reduced system.
        """
        assert 0 < r < self.d.n
        assert projection in ('sr', 'bfsr', 'biorth')

        # compute all necessary Gramian factors
        pcf = self.d.gramian('pcf')
        pof = self.d.gramian('pof')
        vcf = self.d.gramian('vcf')
        vof = self.d.gramian('vof')

        if r > min([len(pcf), len(pof), len(vcf), len(vof)]):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.'
                             ' Try reducing the tolerance in the low-rank matrix equation solver.')

        # find necessary SVDs
        Up, sp, Vp = spla.svd(pof.inner(pcf))
        Up = Up.T
        Uv, sv, Vv = spla.svd(vof.inner(vcf, product=self.d.M))
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
            projected_ops = {'M': IdentityOperator(NumpyVectorSpace(r, id_='STATE'))}
        elif projection == 'bfsr':
            self.V1 = gram_schmidt(self.V1, atol=0, rtol=0)
            self.W1 = gram_schmidt(self.W1, atol=0, rtol=0)
            self.V2 = gram_schmidt(self.V2, atol=0, rtol=0)
            self.W2 = gram_schmidt(self.W2, atol=0, rtol=0)
            W1TV1invW1TV2 = spla.solve(self.W1.inner(self.V1), self.W1.inner(self.V2))
            projected_ops = {'M': project(self.d.M, range_basis=self.W2, source_basis=self.V2)}
        elif projection == 'biorth':
            self.V1, self.W1 = gram_schmidt_biorth(self.V1, self.W1)
            self.V2, self.W2 = gram_schmidt_biorth(self.V2, self.W2, product=self.d.M)
            W1TV1invW1TV2 = self.W1.inner(self.V2)
            projected_ops = {'M': IdentityOperator(NumpyVectorSpace(r, id_='STATE'))}

        projected_ops.update({'D': project(self.d.D,
                                           range_basis=self.W2,
                                           source_basis=self.V2),
                              'K': project(self.d.K,
                                           range_basis=self.W2,
                                           source_basis=self.V1.lincomb(W1TV1invW1TV2.T)),
                              'B': project(self.d.B,
                                           range_basis=self.W2,
                                           source_basis=None),
                              'Cp': project(self.d.Cp,
                                            range_basis=None,
                                            source_basis=self.V1.lincomb(W1TV1invW1TV2.T)),
                              'Cv': project(self.d.Cv,
                                            range_basis=None,
                                            source_basis=self.V2)})

        rd = self.d.with_(operators=projected_ops,
                          visualizer=None, estimator=None,
                          cache_region=None, name=self.d.name + '_reduced')
        rd.disable_logging()

        return rd
