# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.arnoldi import arnoldi
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.discretizations.iosys import LTISystem, SecondOrderSystem, LinearDelaySystem
from pymor.operators.constructions import LincombOperator
from pymor.reductors.basic import GenericPGReductor


class GenericBHIReductor(GenericPGReductor):
    """Generic bitangential Hermite interpolation reductor.

    This reductor can be used for any system with generalized coprime
    factorization :math:`\mathcal{C}(s) \mathcal{K}(s)^{-1}
    \mathcal{B}(s)` as in [BG09]_. The interpolation here is limited to
    only up to the first derivative. Hence, interpolation points are
    assumed to be pairwise distinct.

    .. [BG09] C. A. Beattie, S. Gugercin, Interpolatory projection
              methods for structure-preserving model reduction,
              Systems & Control Letters 58, 2009

    Parameters
    ----------
    d
        Discretization.
    """
    def __init__(self, d):
        self.d = d
        self._B_source = None
        self._C_range = None
        self._K_source = None
        self._product = None
        self._biorthogonal_product = None

    def _B_apply(self, s, V):
        raise NotImplementedError()

    def _C_apply_transpose(self, s, V):
        raise NotImplementedError()

    def _K_apply_inverse(self, s, V):
        raise NotImplementedError()

    def _K_apply_inverse_transpose(self, s, V):
        raise NotImplementedError()

    def reduce(self, sigma, b, c, projection='orth'):
        """Bitangential Hermite interpolation.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), list of
            length `r`.
        b
            Right tangential directions, |VectorArray| of length `r`
            from `self._B_source`.
        c
            Left tangential directions, |VectorArray| of length `r` from
            `self._C_range`.
        projection
            Projection method:

                - `'orth'`: projection matrices are orthogonalized with
                    respect to the Euclidean inner product
                - `'biorth'`: projection matrices are biorthogolized
                    with respect to the E product

        Returns
        -------
        rd
            Reduced discretization.
        """
        r = len(sigma)
        assert b in self._B_source and len(b) == r
        assert c in self._C_range and len(c) == r
        assert projection in ('orth', 'biorth')

        # rescale tangential directions (to avoid overflow or underflow)
        b.scal(1 / b.l2_norm())
        c.scal(1 / c.l2_norm())

        # compute projection matrices
        self.V = self._K_source.empty(reserve=r)
        self.W = self._K_source.empty(reserve=r)
        for i in range(r):
            if sigma[i].imag == 0:
                Bb = self._B_apply(sigma[i].real, b.real[i])
                self.V.append(self._K_apply_inverse(sigma[i].real, Bb))

                CTc = self._C_apply_transpose(sigma[i].real, c.real[i])
                self.W.append(self._K_apply_inverse_transpose(sigma[i].real, CTc))
            elif sigma[i].imag > 0:
                Bb = self._B_apply(sigma[i], b[i])
                v = self._K_apply_inverse(sigma[i], Bb)
                self.V.append(v.real)
                self.V.append(v.imag)

                CTc = self._C_apply_transpose(sigma[i], c[i])
                w = self._K_apply_inverse_transpose(sigma[i], CTc)
                self.W.append(w.real)
                self.W.append(w.imag)

        if projection == 'orth':
            self.V = gram_schmidt(self.V, atol=0, rtol=0)
            self.W = gram_schmidt(self.W, atol=0, rtol=0)
            self.biorthogonal_product = None
        elif projection == 'biorth':
            self.V, self.W = gram_schmidt_biorth(self.V, self.W, product=self._product)
            self.biorthogonal_product = self._biorthogonal_product

        rd = super().reduce()
        return rd

    extend_source_basis = None
    extend_range_basis = None


class LTI_BHIReductor(GenericBHIReductor):
    """Bitangential Hermite interpolation for |LTISystems|.

    Parameters
    ----------
    d
        |LTISystem|.
    """
    def __init__(self, d):
        assert isinstance(d, LTISystem)
        self.d = d
        self._B_source = d.B.source
        self._C_range = d.C.range
        self._K_source = d.A.source
        self._product = d.E
        self._biorthogonal_product = 'E'

    def _B_apply(self, s, V):
        return self.d.B.apply(V)

    def _C_apply_transpose(self, s, V):
        return self.d.C.apply_transpose(V)

    def _K_apply_inverse(self, s, V):
        sEmA = LincombOperator((self.d.E, self.d.A), (s, -1))
        return sEmA.apply_inverse(V)

    def _K_apply_inverse_transpose(self, s, V):
        sEmA = LincombOperator((self.d.E, self.d.A), (s, -1))
        return sEmA.apply_inverse_transpose(V)

    def reduce(self, sigma, b, c, projection='orth', use_arnoldi=False):
        """Bitangential Hermite interpolation.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), list of
            length `r`.
        b
            Right tangential directions, |VectorArray| of length `r`
            from `self._B_source`.
        c
            Left tangential directions, |VectorArray| of length `r` from
            `self._C_range`.
        projection
            Projection method:

                - `'orth'`: projection matrices are orthogonalized with
                    respect to the Euclidean inner product
                - `'biorth'`: projection matrices are biorthogolized
                    with respect to the E product
        use_arnoldi
            Should the Arnoldi process be used for rational
            interpolation. Available only for SISO systems. Otherwise,
            it is ignored.

        Returns
        -------
        rd
            Reduced discretization.
        """
        if use_arnoldi and self.d.m == 1 and self.d.p == 1:
            return self.reduce_arnoldi(sigma, b, c)
        else:
            return super().reduce(sigma, b, c, projection=projection)

    def reduce_arnoldi(self, sigma, b, c):
        """Bitangential Hermite interpolation for SISO |LTISystems|.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), list of
            length `r`.
        b
            Right tangential directions, |VectorArray| of length `r`
            from `d.B.source`.
        c
            Left tangential directions, |VectorArray| of length `r` from
            `d.C.range`.

        Returns
        -------
        rd
            Reduced |LTISystem| model.
        """
        d = self.d
        assert d.m == 1 and d.p == 1
        r = len(sigma)
        assert b in d.B.source and len(b) == r
        assert c in d.C.range and len(c) == r

        self.V = arnoldi(d.A, d.E, d.B, sigma)
        self.W = arnoldi(d.A, d.E, d.C, sigma, trans=True)
        self.biorthogonal_product = None

        rd = super(GenericBHIReductor, self).reduce()
        return rd


class SO_BHIReductor(GenericBHIReductor):
    """Bitangential Hermite interpolation for second-order systems.

    Parameters
    ----------
    d
        :class:`~pymor.discretizations.iosys.SecondOrderSystem`.
    """
    def __init__(self, d):
        assert isinstance(d, SecondOrderSystem)
        self.d = d
        self._B_source = d.B.source
        self._C_range = d.Cp.range
        self._K_source = d.K.source
        self._product = d.M
        self._biorthogonal_product = 'M'

    def _B_apply(self, s, V):
        return self.d.B.apply(V)

    def _C_apply_transpose(self, s, V):
        x = self.d.Cp.apply_transpose(V)
        y = self.d.Cv.apply_transpose(V)
        return x + y * s

    def _K_apply_inverse(self, s, V):
        s2MpsDpK = LincombOperator((self.d.M, self.d.D, self.d.K), (s ** 2, s, 1))
        return s2MpsDpK.apply_inverse(V)

    def _K_apply_inverse_transpose(self, s, V):
        s2MpsDpK = LincombOperator((self.d.M, self.d.D, self.d.K), (s ** 2, s, 1))
        return s2MpsDpK.apply_inverse_transpose(V)


class DelayBHIReductor(GenericBHIReductor):
    """Bitangential Hermite interpolation for delay systems.

    Parameters
    ----------
    d
        :class:`~pymor.discretizations.iosys.LinearDelaySystem`.
    """
    def __init__(self, d):
        assert isinstance(d, LinearDelaySystem)
        self.d = d
        self._B_source = d.B.source
        self._C_range = d.C.range
        self._K_source = d.A.source
        self._product = d.E
        self._biorthogonal_product = 'E'

    def _B_apply(self, s, V):
        return self.d.B.apply(V)

    def _C_apply_transpose(self, s, V):
        return self.d.C.apply_transpose(V)

    def _K_apply_inverse(self, s, V):
        Ks = LincombOperator((self.d.E, self.d.A) + self.d.Ad,
                             (s, -1) + tuple(-np.exp(-taui * s) for taui in self.d.tau))
        return Ks.apply_inverse(V)

    def _K_apply_inverse_transpose(self, s, V):
        Ks = LincombOperator((self.d.E, self.d.A) + self.d.Ad,
                             (s, -1) + tuple(-np.exp(-taui * s) for taui in self.d.tau))
        return Ks.apply_inverse_transpose(V)
