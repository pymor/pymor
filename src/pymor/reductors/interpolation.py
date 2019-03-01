# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.arnoldi import arnoldi
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.core.interfaces import BasicInterface
from pymor.models.iosys import LTIModel, SecondOrderModel, LinearDelayModel
from pymor.operators.constructions import LincombOperator
from pymor.reductors.basic import LTIPGReductor, SOLTIPGReductor, DelayLTIPGReductor


class GenericBHIReductor(BasicInterface):
    r"""Generic bitangential Hermite interpolation reductor.

    This is a generic reductor for reducing any linear
    :class:`~pymor.models.iosys.InputStateOutputModel` with
    the transfer function which can be written in the generalized
    coprime factorization :math:`\mathcal{C}(s) \mathcal{K}(s)^{-1}
    \mathcal{B}(s)` as in [BG09]_.
    The interpolation here is limited to only up to the first
    derivative.
    Hence, interpolation points are assumed to be pairwise distinct.

    Parameters
    ----------
    fom
        Model.
    """

    PGReductor = None

    def __init__(self, fom):
        self.fom = fom
        self._product = None

    def _B_apply(self, s, V):
        raise NotImplementedError

    def _C_apply_adjoint(self, s, V):
        raise NotImplementedError

    def _K_apply_inverse(self, s, V):
        raise NotImplementedError

    def _K_apply_inverse_adjoint(self, s, V):
        raise NotImplementedError

    def reduce(self, sigma, b, c, projection='orth'):
        """Bitangential Hermite interpolation.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), list of
            length `r`.
        b
            Right tangential directions, |VectorArray| of length `r`
            from `self.fom.input_space`.
        c
            Left tangential directions, |VectorArray| of length `r` from
            `self.fom.output_space`.
        projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogolized with
              respect to the E product

        Returns
        -------
        rom
            Reduced model.
        """
        r = len(sigma)
        assert b in self.fom.input_space and len(b) == r
        assert c in self.fom.output_space and len(c) == r
        assert projection in ('orth', 'biorth')

        # rescale tangential directions (to avoid overflow or underflow)
        if b.dim > 1:
            b.scal(1 / b.l2_norm())
        else:
            b = self.fom.input_space.from_numpy(np.ones((r, 1)))
        if c.dim > 1:
            c.scal(1 / c.l2_norm())
        else:
            c = self.fom.output_space.from_numpy(np.ones((r, 1)))

        # compute projection matrices
        self.V = self.fom.state_space.empty(reserve=r)
        self.W = self.fom.state_space.empty(reserve=r)
        for i in range(r):
            if sigma[i].imag == 0:
                Bb = self._B_apply(sigma[i].real, b.real[i])
                self.V.append(self._K_apply_inverse(sigma[i].real, Bb))

                CTc = self._C_apply_adjoint(sigma[i].real, c.real[i])
                self.W.append(self._K_apply_inverse_adjoint(sigma[i].real, CTc))
            elif sigma[i].imag > 0:
                Bb = self._B_apply(sigma[i], b[i])
                v = self._K_apply_inverse(sigma[i], Bb)
                self.V.append(v.real)
                self.V.append(v.imag)

                CTc = self._C_apply_adjoint(sigma[i], c[i].conj())
                w = self._K_apply_inverse_adjoint(sigma[i], CTc)
                self.W.append(w.real)
                self.W.append(w.imag)

        if projection == 'orth':
            self.V = gram_schmidt(self.V, atol=0, rtol=0)
            self.W = gram_schmidt(self.W, atol=0, rtol=0)
        elif projection == 'biorth':
            self.V, self.W = gram_schmidt_biorth(self.V, self.W, product=self._product)

        self.pg_reductor = self.PGReductor(self.fom, self.W, self.V, projection == 'biorth')

        rom = self.pg_reductor.reduce()
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self.RB[:u.dim].lincomb(u.to_numpy())


class LTI_BHIReductor(GenericBHIReductor):
    """Bitangential Hermite interpolation for |LTIModels|.

    Parameters
    ----------
    fom
        |LTIModel|.
    """

    PGReductor = LTIPGReductor

    def __init__(self, fom):
        assert isinstance(fom, LTIModel)
        self.fom = fom
        self._product = fom.E

    def _B_apply(self, s, V):
        return self.fom.B.apply(V)

    def _C_apply_adjoint(self, s, V):
        return self.fom.C.apply_adjoint(V)

    def _K_apply_inverse(self, s, V):
        sEmA = s * self.fom.E - self.fom.A
        return sEmA.apply_inverse(V)

    def _K_apply_inverse_adjoint(self, s, V):
        sEmA = s * self.fom.E - self.fom.A
        return sEmA.apply_inverse_adjoint(V)

    def reduce(self, sigma, b, c, projection='orth', use_arnoldi=False):
        """Bitangential Hermite interpolation.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), list of
            length `r`.
        b
            Right tangential directions, |VectorArray| of length `r`
            from `self.fom.input_space`.
        c
            Left tangential directions, |VectorArray| of length `r` from
            `self.fom.output_space`.
        projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogolized with
              respect to the E product
        use_arnoldi
            Should the Arnoldi process be used for rational
            interpolation. Available only for SISO systems. Otherwise,
            it is ignored.

        Returns
        -------
        rom
            Reduced model.
        """
        if use_arnoldi and self.fom.input_dim == 1 and self.fom.output_dim == 1:
            return self.reduce_arnoldi(sigma, b, c)
        else:
            return super().reduce(sigma, b, c, projection=projection)

    def reduce_arnoldi(self, sigma, b, c):
        """Bitangential Hermite interpolation for SISO |LTIModels|.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), list of
            length `r`.
        b
            Right tangential directions, |VectorArray| of length `r`
            from `self.fom.B.source`.
        c
            Left tangential directions, |VectorArray| of length `r` from
            `self.fom.C.range`.

        Returns
        -------
        rom
            Reduced |LTIModel| model.
        """
        fom = self.fom
        assert fom.input_dim == 1 and fom.output_dim == 1
        r = len(sigma)
        assert b in fom.B.source and len(b) == r
        assert c in fom.C.range and len(c) == r

        self.V = arnoldi(fom.A, fom.E, fom.B, sigma)
        self.W = arnoldi(fom.A, fom.E, fom.C, sigma, trans=True)

        rom = super(GenericBHIReductor, self).reduce()
        return rom


class SO_BHIReductor(GenericBHIReductor):
    """Bitangential Hermite interpolation for second-order systems.

    Parameters
    ----------
    fom
        :class:`~pymor.models.iosys.SecondOrderModel`.
    """

    PGReductor = SOLTIPGReductor

    def __init__(self, fom):
        assert isinstance(fom, SecondOrderModel)
        self.fom = fom
        self._product = fom.M

    def _B_apply(self, s, V):
        return self.fom.B.apply(V)

    def _C_apply_adjoint(self, s, V):
        x = self.fom.Cp.apply_adjoint(V)
        y = self.fom.Cv.apply_adjoint(V)
        return x + y * s.conjugate()

    def _K_apply_inverse(self, s, V):
        s2MpsEpK = s**2 * self.fom.M + s * self.fom.E + self.fom.K
        return s2MpsEpK.apply_inverse(V)

    def _K_apply_inverse_adjoint(self, s, V):
        s2MpsEpK = s**2 * self.fom.M + s * self.fom.E + self.fom.K
        return s2MpsEpK.apply_inverse_adjoint(V)


class DelayBHIReductor(GenericBHIReductor):
    """Bitangential Hermite interpolation for delay systems.

    Parameters
    ----------
    fom
        :class:`~pymor.models.iosys.LinearDelayModel`.
    """

    PGReductor = DelayLTIPGReductor

    def __init__(self, fom):
        assert isinstance(fom, LinearDelayModel)
        self.fom = fom
        self._product = fom.E

    def _B_apply(self, s, V):
        return self.fom.B.apply(V)

    def _C_apply_adjoint(self, s, V):
        return self.fom.C.apply_adjoint(V)

    def _K_apply_inverse(self, s, V):
        Ks = LincombOperator((self.fom.E, self.fom.A) + self.fom.Ad,
                             (s, -1) + tuple(-np.exp(-taui * s) for taui in self.fom.tau))
        return Ks.apply_inverse(V)

    def _K_apply_inverse_adjoint(self, s, V):
        Ks = LincombOperator((self.fom.E, self.fom.A) + self.fom.Ad,
                             (s, -1) + tuple(-np.exp(-taui * s) for taui in self.fom.tau))
        return Ks.apply_inverse_adjoint(V)


class TFInterpReductor(BasicInterface):
    """Loewner bitangential Hermite interpolation reductor.

    See [BG12]_.

    Parameters
    ----------
    fom
        Model with `eval_tf` and `eval_dtf` methods.
    """
    def __init__(self, fom):
        self.fom = fom

    def reduce(self, sigma, b, c):
        """Realization-independent tangential Hermite interpolation.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), list of
            length `r`.
        b
            Right tangential directions, |NumPy array| of shape
            `(fom.input_dim, r)`.
        c
            Left tangential directions, |NumPy array| of shape
            `(fom.output_dim, r)`.

        Returns
        -------
        lti
            |LTIModel| interpolating the transfer function of `fom`.
        """
        fom = self.fom
        r = len(sigma)
        assert isinstance(b, np.ndarray) and b.shape == (fom.input_dim, r)
        assert isinstance(c, np.ndarray) and c.shape == (fom.output_dim, r)

        # rescale tangential directions (to avoid overflow or underflow)
        if b.shape[0] > 1:
            for i in range(r):
                b[:, i] /= spla.norm(b[:, i])
        else:
            b = np.ones((1, r))
        if c.shape[0] > 1:
            for i in range(r):
                c[:, i] /= spla.norm(c[:, i])
        else:
            c = np.ones((1, r))

        # matrices of the interpolatory LTI system
        Er = np.empty((r, r), dtype=complex)
        Ar = np.empty((r, r), dtype=complex)
        Br = np.empty((r, fom.input_dim), dtype=complex)
        Cr = np.empty((fom.output_dim, r), dtype=complex)

        Hs = [fom.eval_tf(s) for s in sigma]
        dHs = [fom.eval_dtf(s) for s in sigma]

        for i in range(r):
            for j in range(r):
                if i != j:
                    Er[i, j] = -c[:, i].dot((Hs[i] - Hs[j]).dot(b[:, j])) / (sigma[i] - sigma[j])
                    Ar[i, j] = -c[:, i].dot((sigma[i] * Hs[i] - sigma[j] * Hs[j])).dot(b[:, j]) / (sigma[i] - sigma[j])
                else:
                    Er[i, i] = -c[:, i].dot(dHs[i].dot(b[:, i]))
                    Ar[i, i] = -c[:, i].dot((Hs[i] + sigma[i] * dHs[i]).dot(b[:, i]))
            Br[i, :] = Hs[i].T.dot(c[:, i])
            Cr[:, i] = Hs[i].dot(b[:, i])

        # transform the system to have real matrices
        T = np.zeros((r, r), dtype=complex)
        for i in range(r):
            if sigma[i].imag == 0:
                T[i, i] = 1
            else:
                indices = np.nonzero(np.isclose(sigma[i + 1:], sigma[i].conjugate()))[0]
                if len(indices) > 0:
                    j = i + 1 + indices[0]
                    T[i, i] = 1
                    T[i, j] = 1
                    T[j, i] = -1j
                    T[j, j] = 1j
        Er = (T.dot(Er).dot(T.conj().T)).real
        Ar = (T.dot(Ar).dot(T.conj().T)).real
        Br = (T.dot(Br)).real
        Cr = (Cr.dot(T.conj().T)).real

        return LTIModel.from_matrices(Ar, Br, Cr, D=None, E=Er, cont_time=fom.cont_time)
