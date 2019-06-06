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
from pymor.reductors.basic import (
    ProjectionBasedReductor,
    LTIPGReductor,
    SOLTIPGReductor,
    DelayLTIPGReductor,
)


class GenericBHIReductor(BasicInterface):
    r"""Generic bitangential Hermite interpolation reductor.

    This is a generic reductor for reducing any linear
    :class:`~pymor.models.iosys.InputStateOutputModel` with the transfer function which can be
    written in the generalized coprime factorization :math:`\mathcal{C}(s) \mathcal{K}(s)^{-1}
    \mathcal{B}(s)` as in [BG09]_.
    The interpolation here is limited to only up to the first derivative.
    Hence, interpolation points are assumed to be pairwise distinct.

    Parameters
    ----------
    fom
        The full-order |Model| to reduce.
    """

    _PGReductor = ProjectionBasedReductor

    def __init__(self, fom):
        self.fom = fom
        self.V = None
        self.W = None
        self._pg_reductor = None
        self._product = None

    def _B_apply(self, s, V):
        raise NotImplementedError

    def _C_apply_adjoint(self, s, V):
        raise NotImplementedError

    def _K_apply_inverse(self, s, V):
        raise NotImplementedError

    def _K_apply_inverse_adjoint(self, s, V):
        raise NotImplementedError

    def reduce(self, sigma, b, c, projection="orth"):
        """Bitangential Hermite interpolation.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), list of length `r`.
        b
            Right tangential directions, |VectorArray| of length `r` from `self.fom.input_space`.
        c
            Left tangential directions, |VectorArray| of length `r` from `self.fom.output_space`.
        projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with respect to the Euclidean inner
              product
            - `'biorth'`: projection matrices are biorthogolized with respect to the E product

        Returns
        -------
        rom
            Reduced-order model.
        """
        r = len(sigma)
        assert b in self.fom.input_space and len(b) == r
        assert c in self.fom.output_space and len(c) == r
        assert projection in ("orth", "biorth")

        # rescale tangential directions (to avoid overflow or underflow)
        if b.dim > 1:
            b.scal(1 / b.l2_norm())
        else:
            b = self.fom.input_space.ones(r)
        if c.dim > 1:
            c.scal(1 / c.l2_norm())
        else:
            c = self.fom.output_space.ones(r)

        # compute projection matrices
        self.V = self.fom.solution_space.empty(reserve=r)
        self.W = self.fom.solution_space.empty(reserve=r)
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
        if projection == "orth":
            self.V = gram_schmidt(self.V, atol=0, rtol=0)
            self.W = gram_schmidt(self.W, atol=0, rtol=0)
        elif projection == "biorth":
            self.V, self.W = gram_schmidt_biorth(self.V, self.W, product=self._product)

        # find reduced-order model
        self._pg_reductor = self._PGReductor(
            self.fom, self.W, self.V, projection == "biorth"
        )
        rom = self._pg_reductor.reduce()
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)


class LTIBHIReductor(GenericBHIReductor):
    """Bitangential Hermite interpolation for |LTIModels|.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    """

    _PGReductor = LTIPGReductor

    def __init__(self, fom):
        assert isinstance(fom, LTIModel)
        super().__init__(fom)
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

    def reduce(self, sigma, b, c, projection="orth"):
        """Bitangential Hermite interpolation.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), list of length `r`.
        b
            Right tangential directions, |VectorArray| of length `r` from `self.fom.input_space`.
        c
            Left tangential directions, |VectorArray| of length `r` from `self.fom.output_space`.
        projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with respect to the Euclidean inner
              product
            - `'biorth'`: projection matrices are biorthogolized with respect to the E product
            - `'arnoldi'`: projection matrices are orthogonalized using the Arnoldi process
              (available only for SISO systems).

        Returns
        -------
        rom
            Reduced-order model.
        """
        if projection != "arnoldi":
            return super().reduce(sigma, b, c, projection=projection)

        assert self.fom.input_dim == 1 and self.fom.output_dim == 1
        r = len(sigma)
        assert b in self.fom.B.source and len(b) == r
        assert c in self.fom.C.range and len(c) == r

        # compute projection matrices
        self.V = arnoldi(self.fom.A, self.fom.E, self.fom.B, sigma)
        self.W = arnoldi(self.fom.A, self.fom.E, self.fom.C, sigma, trans=True)

        # find reduced-order model
        self._pg_reductor = self._PGReductor(self.fom, self.W, self.V)
        rom = self._pg_reductor.reduce()
        return rom


class SOBHIReductor(GenericBHIReductor):
    """Bitangential Hermite interpolation for |SecondOrderModels|.

    Parameters
    ----------
    fom
        The full-order |SecondOrderModel| to reduce.
    """

    _PGReductor = SOLTIPGReductor

    def __init__(self, fom):
        assert isinstance(fom, SecondOrderModel)
        super().__init__(fom)
        self._product = fom.M

    def _B_apply(self, s, V):
        return self.fom.B.apply(V)

    def _C_apply_adjoint(self, s, V):
        x = self.fom.Cp.apply_adjoint(V)
        y = self.fom.Cv.apply_adjoint(V)
        return x + y * s.conjugate()

    def _K_apply_inverse(self, s, V):
        s2MpsEpK = s ** 2 * self.fom.M + s * self.fom.E + self.fom.K
        return s2MpsEpK.apply_inverse(V)

    def _K_apply_inverse_adjoint(self, s, V):
        s2MpsEpK = s ** 2 * self.fom.M + s * self.fom.E + self.fom.K
        return s2MpsEpK.apply_inverse_adjoint(V)


class DelayBHIReductor(GenericBHIReductor):
    """Bitangential Hermite interpolation for |LinearDelayModels|.

    Parameters
    ----------
    fom
        The full-order |LinearDelayModel| to reduce.
    """

    _PGReductor = DelayLTIPGReductor

    def __init__(self, fom):
        assert isinstance(fom, LinearDelayModel)
        super().__init__(fom)
        self._product = fom.E

    def _B_apply(self, s, V):
        return self.fom.B.apply(V)

    def _C_apply_adjoint(self, s, V):
        return self.fom.C.apply_adjoint(V)

    def _K_apply_inverse(self, s, V):
        Ks = LincombOperator(
            (self.fom.E, self.fom.A) + self.fom.Ad,
            (s, -1) + tuple(-np.exp(-taui * s) for taui in self.fom.tau),
        )
        return Ks.apply_inverse(V)

    def _K_apply_inverse_adjoint(self, s, V):
        Ks = LincombOperator(
            (self.fom.E, self.fom.A) + self.fom.Ad,
            (s, -1) + tuple(-np.exp(-taui * s) for taui in self.fom.tau),
        )
        return Ks.apply_inverse_adjoint(V)


class TFBHIReductor(BasicInterface):
    """Loewner bitangential Hermite interpolation reductor.

    See [BG12]_.

    Parameters
    ----------
    fom
        The |Model| with `eval_tf` and `eval_dtf` methods.
    """

    def __init__(self, fom):
        self.fom = fom

    def reduce(self, sigma, b, c):
        """Realization-independent tangential Hermite interpolation.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), list of length `r`.
        b
            Right tangential directions, |NumPy array| of shape `(fom.input_dim, r)`.
        c
            Left tangential directions, |NumPy array| of shape `(fom.output_dim, r)`.

        Returns
        -------
        lti
            The reduced-order |LTIModel| interpolating the transfer function of `fom`.
        """
        r = len(sigma)
        assert isinstance(b, np.ndarray) and b.shape == (self.fom.input_dim, r)
        assert isinstance(c, np.ndarray) and c.shape == (self.fom.output_dim, r)

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
        Br = np.empty((r, self.fom.input_dim), dtype=complex)
        Cr = np.empty((self.fom.output_dim, r), dtype=complex)

        Hs = [self.fom.eval_tf(s) for s in sigma]
        dHs = [self.fom.eval_dtf(s) for s in sigma]

        for i in range(r):
            for j in range(r):
                if i != j:
                    Er[i, j] = -c[:, i].dot((Hs[i] - Hs[j]).dot(b[:, j])) / (
                        sigma[i] - sigma[j]
                    )
                    Ar[i, j] = -c[:, i].dot((sigma[i] * Hs[i] - sigma[j] * Hs[j])).dot(
                        b[:, j]
                    ) / (sigma[i] - sigma[j])
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
                indices = np.nonzero(np.isclose(sigma[i + 1 :], sigma[i].conjugate()))[
                    0
                ]
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

        return LTIModel.from_matrices(
            Ar, Br, Cr, D=None, E=Er, cont_time=self.fom.cont_time
        )

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        raise TypeError(
            f"The reconstruct method is not available for {self.__class__.__name__}."
        )
