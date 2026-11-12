# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.models.transfer_function import TransferFunction


class BarycentricRationalTransferFunction(TransferFunction):
    r"""Transfer functions in barycentric rational form.

    This class represents a rational transfer function :math:`H` with :math:`m`
    inputs and :math:`p` outputs in the complex frequency variable
    :math:`s \in \mathbb{C}`. Its barycentric representation is specified by
    :math:`n \geq 1` distinct support nodes :math:`z_i \in \mathbb{C}` and
    sample values :math:`F_i \in \mathbb{C}^{p \times m}`, indexed by
    :math:`i = 1, \ldots, n`. Scalar samples correspond to :math:`p = m = 1`.
    The weights are either nonzero scalars :math:`w_i \in \mathbb{C}` or
    nonsingular matrices :math:`W_i \in \mathbb{C}^{p \times p}` for the left
    form and :math:`W_i \in \mathbb{C}^{m \times m}` for the right form.
    At each support node, the rational expressions below are defined by
    continuous extension, so that :math:`H(z_i) = F_i`.

    There exist multiple barycentric representation formulas, see :cite:`GG21` (Section 3).
    For scalar weights, we use the standard barycentric form of :cite:`NST18` (bary-A)

    .. math::
        H(s) = \frac{\sum_i w_i F_i / (s-z_i)}{\sum_i w_i / (s-z_i)}.

    The samples :math:`F_i` can be scalars or matrices. Matrix-valued samples
    still share a scalar denominator and require no matrix solves.

    For matrix weights and ``side='left'``, represent the block-AAA form of
    :cite:`GG21` (Section 3)

    .. math::
        H(s) = \left(\sum_i W_i / (s-z_i)\right)^{-1}
               \left(\sum_i W_i F_i / (s-z_i)\right).

    With ``side='right'``, instead represent

    .. math::
        H(s) = \left(\sum_i F_i W_i / (s-z_i)\right)
               \left(\sum_i W_i / (s-z_i)\right)^{-1}.

    Distinct support nodes and nonzero scalar or nonsingular matrix weights
    ensure interpolation at the nodes. Evaluation and first derivatives use
    nearest-node rescaling to avoid artificial singular terms near support
    nodes. Matrix evaluation solves only in the denominator dimension.
    No pole-zero cancellation is performed away from the support nodes.

    Unlike a strictly proper realization, standard AAA can have a nonzero
    limit at infinity. Infinity is supported when the sum of the weights is
    nonzero (scalar weights) or nonsingular (matrix weights); otherwise
    evaluation at infinity raises :class:`ValueError`.

    Parameters
    ----------
    nodes
        Distinct finite support points, of shape `(n,)`, with `n > 0`.
    values
        Finite samples, of shape `(n,)` for SISO or `(n, p, m)` for `p` outputs
        and `m` inputs. Evaluations always return a matrix, including for SISO.
    weights
        Finite nonzero scalar weights of shape `(n,)`, or nonsingular matrix
        weights of shape `(n, p, p)` for the left form and `(n, m, m)` for the
        right form. A common nonzero scaling of all weights does not change H.
    side
        ``'left'`` (default) or ``'right'``. Only affects matrix weights.
    sampling_time
        Sampling time, with the same convention as |TransferFunction|.
    name
        Name of the transfer function.
    """

    def __init__(self, nodes, values, weights, *, side='left', sampling_time=0, name=None):
        assert side in ('left', 'right'), "side must be 'left' or 'right'."
        sampling_time = float(sampling_time)
        assert np.isfinite(sampling_time), 'sampling_time must be finite.'
        assert sampling_time >= 0, 'sampling_time must be nonnegative.'
        arrays = [np.asarray(a) for a in (nodes, values, weights)]
        dtype = np.result_type(*arrays, np.float64)
        assert dtype.kind in 'fc', 'nodes, values and weights must be numeric arrays.'
        nodes, values, weights = (np.array(a, dtype=dtype, copy=True) for a in arrays)
        assert nodes.ndim == 1, 'nodes must be a vector.'
        assert len(nodes) > 0, 'nodes must be nonempty.'
        n = len(nodes)
        assert values.ndim in (1, 3), 'values must have shape (n,) or (n, p, m).'
        assert values.shape[0] == n, 'There must be one value per support node.'
        assert all(d > 0 for d in values.shape), 'values must have positive dimensions.'
        assert all(np.all(np.isfinite(a)) for a in (nodes, values, weights)), 'Barycentric data must be finite.'
        assert len(np.unique(nodes)) == n, 'Support nodes must be distinct.'
        shaped_values = values.reshape(n, 1, 1) if values.ndim == 1 else values
        p, m = shaped_values.shape[1:]
        size = p if side == 'left' else m
        assert weights.shape in ((n,), (n, size, size)), \
            'Incompatible barycentric weight shape for the chosen side.'
        matrix_weights = weights.ndim == 3
        if not matrix_weights:
            assert np.all(weights != 0), 'Scalar weights must be nonzero.'
        # Common scales avoid product overflow; weights cancel, value scaling is restored.
        scaled_weights, _ = self._normalize(weights, 'weights')
        scaled_values, value_scale = self._normalize(shaped_values, 'values')
        if matrix_weights:
            try:
                for weight in scaled_weights:
                    self._factorize(weight)
            except spla.LinAlgError as error:
                raise ValueError('Matrix weights must be nonsingular.') from error

        super().__init__(m, p, self._evaluate, dtf=self._evaluate_derivative,
                         sampling_time=sampling_time, name=name)
        self.nodes, self.values, self.weights = nodes, values, weights
        self.side = side
        self._values = shaped_values
        self._weights = scaled_weights[:, 0, 0] if matrix_weights and size == 1 else scaled_weights
        self._scaled_values = scaled_values
        self._value_scale = value_scale
        self._matrix_weights = matrix_weights and size > 1
        if self._matrix_weights:
            # Transpose the right form into a left solve; no complex conjugation is involved.
            self._left_values = scaled_values if side == 'left' else scaled_values.swapaxes(1, 2)
            left_weights = scaled_weights if side == 'left' else scaled_weights.swapaxes(1, 2)
            # Column-major blocks make each contraction a LAPACK-ready work array.
            self._flat_weights = np.ascontiguousarray(left_weights.swapaxes(1, 2)).reshape(n, -1)
            self._left_weights = self._flat_weights.reshape(n, size, size).swapaxes(1, 2)
            self._weights = self._left_weights if side == 'left' else self._left_weights.swapaxes(1, 2)
            weighted_values = self._left_weights @ self._left_values
            self._flat_weighted_values = np.ascontiguousarray(weighted_values.swapaxes(1, 2)).reshape(n, -1)
            self._weighted_values = self._flat_weighted_values.reshape(n, -1, size).swapaxes(1, 2)
        else:
            self._flat_values = np.ascontiguousarray(scaled_values).reshape(n, -1)
            if p == m == 1:
                self._flat_values = self._flat_values[:, 0]
            try:
                with np.errstate(under='raise'):
                    multipliers = self._weights if p == m == 1 else self._weights[:, None]
                    self._flat_numerator = multipliers * self._flat_values
            except FloatingPointError:
                # Early multiplication must not erase terms later amplified by qprime.
                self._flat_numerator = None
        for array in self.__dict__.values():
            if isinstance(array, np.ndarray):
                array.flags.writeable = False

    def __setstate__(self, state):
        # NumPy pickling does not preserve the arrays' write protection.
        self.__dict__.update(state)
        for array in self.__dict__.values():
            if isinstance(array, np.ndarray):
                array.flags.writeable = False

    def _evaluate(self, s):
        return self._eval(s)

    def _evaluate_derivative(self, s):
        return self._eval(s, derivative=True)

    def _eval(self, s, derivative=False):
        assert getattr(s, 'ndim', None) == 0 or np.isscalar(s) or np.ndim(s) == 0, 's must be a scalar.'
        at_infinity = not np.isfinite(s)
        if at_infinity:
            assert not np.isnan(s), 's must not be NaN.'
        if len(self.nodes) == 1:
            return np.zeros_like(self._values[0]) if derivative else self._values[0]
        at_node = False
        if at_infinity:
            # All rescaled coefficients tend to one, with derivatives tending to zero.
            q = np.ones(len(self.nodes), dtype=self.nodes.dtype)
            qprime = np.zeros_like(q) if derivative else None
        else:
            delta = s - self.nodes
            j = abs(delta).argmin()
            nearest_delta = delta[j]
            at_node = nearest_delta == 0
            if at_node and not derivative:
                return self._values[j]
            # Replace the nearest denominator before division, including exact/subnormal hits.
            delta[j] = 1
            q = nearest_delta / delta
            q[j] = 1
            qprime = None
            if derivative:
                # q_i' = (z_j - z_i) / (s - z_i)^2; divide twice to avoid squaring overflow.
                qprime = (self.nodes[j] - self.nodes) / delta
                qprime /= delta
                qprime[j] = 0

        if self._matrix_weights:
            size = self._left_weights.shape[1]
            denominator = q.dot(self._flat_weights).reshape(size, size).T
            try:
                factors = self._factorize(denominator, overwrite_a=True)
            except spla.LinAlgError as error:
                if at_infinity:
                    raise ValueError('Singular leading denominator at infinity.') from error
                raise
            if at_node:
                value = self._left_values[j]
            else:
                numerator = q.dot(self._flat_weighted_values).reshape(-1, size).T
                value = self._solve(factors, numerator)
            if derivative:
                # Differentiate D H = N and reuse the LU factors to solve D H' = N' - D' H.
                dN = qprime.dot(self._flat_weighted_values).reshape(-1, size).T
                dD = qprime.dot(self._flat_weights).reshape(size, size).T
                dN -= dD @ value
                value = self._solve(factors, dN)
            value *= self._value_scale
            return value if self.side == 'left' else value.T

        denominator = q.dot(self._weights)
        if denominator == 0:
            if at_infinity:
                raise ValueError('Zero leading denominator at infinity.')
            raise spla.LinAlgError('Singular barycentric denominator.')
        numerator_values = self._flat_values if self._flat_numerator is None else self._flat_numerator
        if at_node:
            value = self._flat_values[j]
        else:
            coefficients = q * self._weights if self._flat_numerator is None else q
            value = coefficients.dot(numerator_values)
            value /= denominator
        if derivative:
            dcoefficients = qprime * self._weights if self._flat_numerator is None else qprime
            dvalue = dcoefficients.dot(numerator_values)
            dvalue -= value * qprime.dot(self._weights)
            dvalue /= denominator
            value = dvalue
        value *= self._value_scale
        return value.reshape(self._values.shape[1:])

    @staticmethod
    def _normalize(array, label):
        # Componentwise scaling avoids overflow in complex magnitudes
        # or division by subnormal scales.
        scale = max(np.max(abs(array.real)), np.max(abs(array.imag)))
        if scale == 0:
            return array.copy(), 1.
        scaled = np.empty_like(array)
        scaled.real = array.real / scale
        if np.iscomplexobj(array):
            scaled.imag = array.imag / scale
        if np.any((array != 0) & (scaled == 0)):
            raise ValueError(f'The relative range of {label} is too large for normalization.')
        return scaled, scale

    @staticmethod
    def _factorize(matrix, overwrite_a=False):
        # LAPACK status avoids warning machinery while retaining exact singularity detection.
        getrf = spla.lapack.zgetrf if matrix.dtype.kind == 'c' else spla.lapack.dgetrf
        lu, pivots, info = getrf(matrix, overwrite_a=overwrite_a)
        if info > 0:
            raise spla.LinAlgError('Singular barycentric denominator.')
        if info < 0:
            raise ValueError(f'Illegal argument {-info} in LAPACK getrf.')
        return lu, pivots

    @staticmethod
    def _solve(factors, rhs):
        # Every right-hand side is a fresh, column-major contraction work array.
        lu, pivots = factors
        getrs = spla.lapack.zgetrs if lu.dtype.kind == 'c' else spla.lapack.dgetrs
        value, info = getrs(lu, pivots, rhs, overwrite_b=True)
        if info != 0:
            raise ValueError(f'Illegal argument {-info} in LAPACK getrs.')
        return value
