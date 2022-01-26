# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.krylov import rational_arnoldi
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel, SecondOrderModel, LinearDelayModel
from pymor.models.transfer_function import TransferFunction
from pymor.parameters.base import Mu
from pymor.reductors.basic import (ProjectionBasedReductor, LTIPGReductor, SOLTIPGReductor,
                                   DelayLTIPGReductor)


class GenericBHIReductor(BasicObject):
    r"""Generic bitangential Hermite interpolation reductor.

    This is a generic reductor for reducing any linear `Model` that has a
    transfer function that is a `FactorizedTransferFunction`
    (see :cite:`BG09`).
    The interpolation here is limited to only up to the first derivative.
    Interpolation points are assumed to be pairwise distinct.

    In particular, given:

    - interpolation points :math:`\sigma_i`,
    - right tangential directions :math:`b_i`, and
    - left tangential directions :math:`c_i`,

    for :math:`i = 1, 2, \ldots, r`,
    which are closed under conjugation
    (if :math:`\sigma_i` is real, then so are :math:`b_i` and :math:`c_i`;
    if :math:`\sigma_i` is complex, there is :math:`\sigma_j` such that
    :math:`\sigma_j = \overline{\sigma_i}`,
    :math:`b_j = \overline{b_i}`,
    :math:`c_j = \overline{c_i}`),
    this reductor finds a transfer function :math:`\hat{H}` such that

    .. math::
        H(\sigma_i) b_i & = \hat{H}(\sigma_i) b_i, \\
        c_i^T H(\sigma_i) & = c_i^T \hat{H}(\sigma_i), \\
        c_i^T H'(\sigma_i) b_i & = c_i^T \hat{H}'(\sigma_i) b_i,

    for all :math:`i = 1, 2, \ldots, r`.

    Parameters
    ----------
    fom
        The full-order |Model| to reduce.
    mu
        |Parameter values|.
    """

    _PGReductor = ProjectionBasedReductor

    def __init__(self, fom, mu=None):
        if not isinstance(mu, Mu):
            mu = fom.parameters.parse(mu)
        assert fom.parameters.assert_compatible(mu)
        self.fom = fom
        self.mu = mu
        self.V = None
        self.W = None
        self._pg_reductor = None
        self._product = None

    def _B_apply(self, s, V):
        return self.fom.transfer_function.B(s).apply(V, mu=self.mu)

    def _C_apply_adjoint(self, s, V):
        return self.fom.transfer_function.C(s).apply_adjoint(V, mu=self.mu)

    def _K_apply_inverse(self, s, V):
        return self.fom.transfer_function.K(s).apply_inverse(V, mu=self.mu)

    def _K_apply_inverse_adjoint(self, s, V):
        return self.fom.transfer_function.K(s).apply_inverse_adjoint(V, mu=self.mu)

    def _fom_assemble(self):
        raise NotImplementedError

    def reduce(self, sigma, b, c, projection='orth'):
        """Bitangential Hermite interpolation.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), sequence of
            length `r`.
        b
            Right tangential directions, |NumPy array| of shape
            `(r, fom.dim_input)`.
        c
            Left tangential directions, |NumPy array| of shape
            `(r, fom.dim_output)`.
        projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogolized with
              respect to the E product

        Returns
        -------
        rom
            Reduced-order model.
        """
        r = len(sigma)
        assert b.shape == (r, self.fom.dim_input)
        assert c.shape == (r, self.fom.dim_output)
        assert projection in ('orth', 'biorth')

        # rescale tangential directions (to avoid overflow or underflow)
        b = b / np.linalg.norm(b, axis=0) if b.shape[1] > 1 else np.ones((r, 1))
        c = c / np.linalg.norm(c, axis=0) if c.shape[1] > 1 else np.ones((r, 1))
        b = self.fom.D.source.from_numpy(b)
        c = self.fom.D.range.from_numpy(c)

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
        if projection == 'orth':
            gram_schmidt(self.V, atol=0, rtol=0, copy=False)
            gram_schmidt(self.W, atol=0, rtol=0, copy=False)
        elif projection == 'biorth':
            gram_schmidt_biorth(self.V, self.W, product=self._product, copy=False)

        # find reduced-order model
        self._pg_reductor = self._PGReductor(self._fom_assemble(), self.W, self.V,
                                             projection == 'biorth')
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
    mu
        |Parameter values|.
    """

    _PGReductor = LTIPGReductor

    def __init__(self, fom, mu=None):
        assert isinstance(fom, LTIModel)
        super().__init__(fom, mu=mu)
        self._product = fom.E

    def _fom_assemble(self):
        if self.fom.parametric:
            return self.fom.with_(
                **{op: getattr(self.fom, op).assemble(mu=self.mu)
                   for op in ['A', 'B', 'C', 'D', 'E']}
            )
        return self.fom

    def reduce(self, sigma, b, c, projection='orth'):
        """Bitangential Hermite interpolation.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), sequence of
            length `r`.
        b
            Right tangential directions, |NumPy array| of shape
            `(r, fom.dim_input)`.
        c
            Left tangential directions, |NumPy array| of shape
            `(r, fom.dim_output)`.
        projection
            Projection method:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogolized with
              respect to the E product
            - `'arnoldi'`: projection matrices are orthogonalized using
              the rational Arnoldi process (available only for SISO
              systems).

        Returns
        -------
        rom
            Reduced-order model.
        """
        if projection != 'arnoldi':
            return super().reduce(sigma, b, c, projection=projection)

        assert self.fom.dim_input == 1 and self.fom.dim_output == 1
        r = len(sigma)
        assert b.shape == (r, self.fom.dim_input)
        assert c.shape == (r, self.fom.dim_output)

        # compute projection matrices
        self.V = rational_arnoldi(self.fom.A, self.fom.E, self.fom.B, sigma)
        self.W = rational_arnoldi(self.fom.A, self.fom.E, self.fom.C, sigma, trans=True)

        # find reduced-order model
        self._pg_reductor = self._PGReductor(self._fom_assemble(), self.W, self.V)
        rom = self._pg_reductor.reduce()
        return rom


class SOBHIReductor(GenericBHIReductor):
    """Bitangential Hermite interpolation for |SecondOrderModels|.

    Parameters
    ----------
    fom
        The full-order |SecondOrderModel| to reduce.
    mu
        |Parameter values|.
    """

    _PGReductor = SOLTIPGReductor

    def __init__(self, fom, mu=None):
        assert isinstance(fom, SecondOrderModel)
        super().__init__(fom, mu=mu)
        self._product = fom.M

    def _fom_assemble(self):
        if self.fom.parametric:
            return self.fom.with_(
                **{op: getattr(self.fom, op).assemble(mu=self.mu)
                   for op in ['M', 'E', 'K', 'B', 'Cp', 'Cv', 'D']}
            )
        return self.fom


class DelayBHIReductor(GenericBHIReductor):
    """Bitangential Hermite interpolation for |LinearDelayModels|.

    Parameters
    ----------
    fom
        The full-order |LinearDelayModel| to reduce.
    mu
        |Parameter values|.
    """

    _PGReductor = DelayLTIPGReductor

    def __init__(self, fom, mu=None):
        assert isinstance(fom, LinearDelayModel)
        super().__init__(fom, mu=mu)
        self._product = fom.E

    def _fom_assemble(self):
        if self.fom.parametric:
            return self.fom.with_(
                **{op: getattr(self.fom, op).assemble(mu=self.mu)
                   for op in ['A', 'B', 'C', 'D', 'E']},
                Ad=tuple(op.assemble(mu=self.mu) for op in self.fom.Ad)
            )
        return self.fom


class TFBHIReductor(BasicObject):
    """Loewner bitangential Hermite interpolation reductor.

    See :cite:`BG12`.

    Parameters
    ----------
    fom
        |TransferFunction| or |Model| with a `transfer_function` attribute.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        assert isinstance(fom, TransferFunction) or hasattr(fom, 'transfer_function')
        if not isinstance(fom, TransferFunction):
            fom = fom.transfer_function
        if not isinstance(mu, Mu):
            mu = fom.parameters.parse(mu)
        assert fom.parameters.assert_compatible(mu)
        self.fom = fom
        self.mu = mu

    def reduce(self, sigma, b, c):
        """Realization-independent tangential Hermite interpolation.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), sequence of
            length `r`.
        b
            Right tangential directions, |NumPy array| of shape
            `(r, fom.dim_input)`.
        c
            Left tangential directions, |NumPy array| of shape
            `(r, fom.dim_output)`.

        Returns
        -------
        lti
            The reduced-order |LTIModel| interpolating the transfer
            function of `fom`.
        """
        r = len(sigma)
        assert b.shape == (r, self.fom.dim_input)
        assert c.shape == (r, self.fom.dim_output)

        # rescale tangential directions (to avoid overflow or underflow)
        b = b * (1 / np.linalg.norm(b)) if b.shape[1] > 1 else np.ones((r, 1))
        c = c * (1 / np.linalg.norm(c)) if c.shape[1] > 1 else np.ones((r, 1))

        # matrices of the interpolatory LTI system
        Er = np.empty((r, r), dtype=np.complex_)
        Ar = np.empty((r, r), dtype=np.complex_)
        Br = np.empty((r, self.fom.dim_input), dtype=np.complex_)
        Cr = np.empty((self.fom.dim_output, r), dtype=np.complex_)

        Hs = [self.fom.eval_tf(s, mu=self.mu) for s in sigma]
        dHs = [self.fom.eval_dtf(s, mu=self.mu) for s in sigma]

        for i in range(r):
            for j in range(r):
                if i != j:
                    Er[i, j] = -c[i] @ (Hs[i] - Hs[j]) @ b[j] / (sigma[i] - sigma[j])
                    Ar[i, j] = (-c[i] @ (sigma[i] * Hs[i] - sigma[j] * Hs[j]) @ b[j]
                                / (sigma[i] - sigma[j]))
                else:
                    Er[i, i] = -c[i] @ dHs[i] @ b[i]
                    Ar[i, i] = -c[i] @ (Hs[i] + sigma[i] * dHs[i]) @ b[i]
            Br[i, :] = Hs[i].T @ c[i]
            Cr[:, i] = Hs[i] @ b[i]

        # transform the system to have real matrices
        T = np.zeros((r, r), dtype=np.complex_)
        for i in range(r):
            if sigma[i].imag == 0:
                T[i, i] = 1
            else:
                j = np.argmin(np.abs(sigma - sigma[i].conjugate()))
                if i < j:
                    T[i, i] = 1
                    T[i, j] = 1
                    T[j, i] = -1j
                    T[j, j] = 1j
        Er = (T @ Er @ T.conj().T).real
        Ar = (T @ Ar @ T.conj().T).real
        Br = (T @ Br).real
        Cr = (Cr @ T.conj().T).real

        return LTIModel.from_matrices(Ar, Br, Cr, None, Er, sampling_time=self.fom.sampling_time)

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        raise TypeError(f'The reconstruct method is not available for {self.__class__.__name__}.')
