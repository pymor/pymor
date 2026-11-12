# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.models.transfer_function import TransferFunction


class BarycentricRationalTransferFunction(TransferFunction):
    r"""Strictly proper transfer functions in factored right barycentric form.

    With :math:`\Delta(s) = sI - \operatorname{diag}(z_i)`, represent

    .. math::
        H(s) = C\Delta(s)^{-1}B
               \left(I + R\Delta(s)^{-1}B\right)^{-1}.

    The factors can encode matrix-valued and tangential data without expanding
    low-rank weights to full matrix coefficients. Repeated nodes are allowed.
    Only the normalized form with an identity constant in the denominator is
    supported; the zero-constant form used by standard AAA is not supported.

    Evaluation uses the equivalent realization
    :math:`H(s) = C(sI - A)^{-1}B`, with
    :math:`A = \operatorname{diag}(z_i) - BR`. This avoids division by zero at
    support nodes and uses a dense solve in the support dimension. No pole-zero
    cancellation is performed: a singular :math:`sI - A` raises
    `scipy.linalg.LinAlgError`, even if the singularity is removable in :math:`H`.

    Parameters
    ----------
    nodes
        Finite support nodes of shape `(k,)`.
    B
        Weight factor of shape `(k, m)`.
    C
        Numerator factor of shape `(p, k)`.
    R
        Right-direction factor of shape `(m, k)`.
    sampling_time
        Sampling time, with the same convention as |TransferFunction|.
    name
        Name of the transfer function.
    """

    def __init__(self, nodes, B, C, R, sampling_time=0, name=None):
        factors = [np.asarray(a) for a in (nodes, B, C, R)]
        dtype = np.result_type(*factors, np.float64)
        nodes, B, C, R = (np.array(a, dtype=dtype, copy=True) for a in factors)
        if nodes.ndim != 1 or any(a.ndim != 2 for a in (B, C, R)):
            raise ValueError('nodes must be a vector and B, C, R must be matrices.')
        k = len(nodes)
        if B.shape[0] != k or C.shape[1] != k or R.shape != (B.shape[1], k):
            raise ValueError('Incompatible barycentric factor shapes.')
        if not B.shape[1] or not C.shape[0]:
            raise ValueError('Input and output dimensions must be positive.')
        if any(not np.all(np.isfinite(a)) for a in (nodes, B, C, R)):
            raise ValueError('Barycentric factors must be finite.')
        if not np.isfinite(sampling_time) or sampling_time < 0:
            raise ValueError('sampling_time must be finite and nonnegative.')
        for a in (nodes, B, C, R):
            a.flags.writeable = False
        self.nodes, self.B, self.C, self.R = nodes, B, C, R
        self._A = B @ R
        self._A *= -1
        self._A.flat[::k + 1] += nodes
        self._A.flags.writeable = False
        super().__init__(B.shape[1], C.shape[0], self._evaluate, dtf=self._evaluate_derivative,
                         sampling_time=sampling_time, name=name)

    def __setstate__(self, state):
        self.__dict__.update(state)
        for a in (self.nodes, self.B, self.C, self.R, self._A):
            a.flags.writeable = False

    def _evaluate_barycentric(self, s, derivative=False):
        if np.ndim(s):
            raise ValueError('s must be a scalar.')
        if np.isnan(s):
            raise ValueError('s must not be NaN.')
        if not len(self.nodes) or np.isinf(s):
            return np.zeros((self.dim_output, self.dim_input), dtype=np.result_type(self.B, self.C, self._A))
        K = s * np.eye(len(self.nodes)) - self._A
        X = spla.solve(K, self.B)
        return -self.C @ spla.solve(K, X) if derivative else self.C @ X

    def _evaluate(self, s):
        return self._evaluate_barycentric(s)

    def _evaluate_derivative(self, s):
        return self._evaluate_barycentric(s, derivative=True)
