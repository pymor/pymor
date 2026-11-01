# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.models.transfer_function import TransferFunction


class PoleResidueTransferFunction(TransferFunction):
    r"""Transfer functions in pole-residue form.

    This class describes input-output systems given by a transfer function

    .. math::
        H(s) = \sum_{i = 1}^r \frac{c_i b_i^T}{s - \lambda_i} + D,

    where :math:`\lambda_i \in \mathbb{C}` are the poles and :math:`c_i b_i^T` are the
    residues given by their rank-1 factors :math:`b_i \in \mathbb{C}^m` and
    :math:`c_i \in \mathbb{C}^p`.
    Poles can be repeated, so residues of higher rank can be represented as well.

    An |LTIModel| can be converted to this form using
    :meth:`~pymor.models.iosys.LTIModel.to_pole_residue`.

    Parameters
    ----------
    poles
        1D |NumPy array| of poles of length `r`.
    b
        |NumPy array| of shape `(r, m)` whose rows are the right residue factors.
    c
        |NumPy array| of shape `(r, p)` whose rows are the left residue factors.
    D
        |NumPy array| of shape `(p, m)` or `None` (then D is assumed to be zero).
    sampling_time
        `0` if the system is continuous-time, otherwise a positive number that denotes the
        sampling time (in seconds).
    name
        Name of the system.

    Attributes
    ----------
    order
        The number of poles.
    poles
        The poles.
    b
        The right residue factors.
    c
        The left residue factors.
    D
        The feedthrough term.
    """

    def __init__(self, poles, b, c, D=None, sampling_time=0, name=None):
        poles = np.array(poles)
        b = np.array(b)
        c = np.array(c)
        assert poles.ndim == 1
        assert b.ndim == 2
        assert b.shape[0] == len(poles)
        assert c.ndim == 2
        assert c.shape[0] == len(poles)
        if D is not None:
            D = np.array(D)
            assert D.shape == (c.shape[1], b.shape[1])
        for arr in (poles, b, c, D):
            if arr is not None:
                arr.setflags(write=False)  # make numpy arrays read-only

        def tf(s):
            res = (c.T / (s - poles)) @ b
            return res if D is None else res + D

        def dtf(s):
            return -(c.T / (s - poles)**2) @ b

        super().__init__(b.shape[1], c.shape[1], tf, dtf=dtf, sampling_time=sampling_time, name=name)
        self.poles = poles
        self.b = b
        self.c = c
        self.D = D

    @property
    def order(self):
        return len(self.poles)

    def __str__(self):
        string = (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of poles:   {self.order}\n'
            f'    number of inputs:  {self.dim_input}\n'
            f'    number of outputs: {self.dim_output}\n'
        )
        if self.sampling_time == 0:
            string += '    continuous-time'
        else:
            string += f'    {f"discrete-time (sampling_time={self.sampling_time:.2e}s)"}'
        return string

    def to_lti(self):
        """Return an |LTIModel| with real matrices.

        The poles and residue factors are assumed to be closed under complex
        conjugation.
        Of each pair of complex conjugate poles, only the one with positive
        imaginary part is used.

        Returns
        -------
        lti
            |LTIModel| in real block-diagonal form with the same transfer function.
        """
        from pymor.models.iosys import LTIModel

        A, B, C = [], [], []
        for pole, b, c in zip(self.poles, self.b, self.c, strict=True):
            if pole.imag == 0:
                A.append(pole.real)
                B.append(b.real)
                C.append(c.real[:, np.newaxis])
            elif pole.imag > 0:
                A.append([[pole.real, pole.imag],
                          [-pole.imag, pole.real]])
                B.append(np.vstack([2 * b.real, -2 * b.imag]))
                C.append(np.hstack([c.real[:, np.newaxis], c.imag[:, np.newaxis]]))
        A = spla.block_diag(*A)
        B = np.vstack(B)
        C = np.hstack(C)
        return LTIModel.from_matrices(A, B, C, self.D, sampling_time=self.sampling_time, name=self.name)

    def __add__(self, other):
        if not isinstance(other, PoleResidueTransferFunction):
            return super().__add__(other)

        assert self.sampling_time == other.sampling_time
        assert self.dim_input == other.dim_input
        assert self.dim_output == other.dim_output

        poles = np.concatenate((self.poles, other.poles))
        b = np.vstack((self.b, other.b))
        c = np.vstack((self.c, other.c))
        if self.D is None or other.D is None:
            D = self.D if other.D is None else other.D
        else:
            D = self.D + other.D
        return PoleResidueTransferFunction(poles, b, c, D, sampling_time=self.sampling_time)

    __radd__ = __add__

    def __neg__(self):
        return self.with_(c=-self.c, D=None if self.D is None else -self.D)
