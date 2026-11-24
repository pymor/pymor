# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.barycentric import (
    _factorize,
    _normalize,
    evaluate_bary_a,
    evaluate_bary_b,
)
from pymor.models.pole_residue import PoleResidueTransferFunction
from pymor.models.transfer_function import TransferFunction


class BarycentricRationalTransferFunction(TransferFunction):
    r"""Transfer functions in barycentric rational form.

    This class represents a rational transfer function :math:`H` with :math:`m`
    inputs and :math:`p` outputs. For scalar weights, the bary-A form of
    :cite:`GG21` (Section 3) is

    .. math::
        H(s) = \frac{\sum_i w_i F_i / (s-z_i)}{\sum_i w_i / (s-z_i)}.

    For matrix weights and `side='left'`, the bary-B form is

    .. math::
        H(s) = \left(\sum_i W_i/(s-z_i)\right)^{-1}
               \left(\sum_i W_i F_i/(s-z_i)\right).

    For `side='right'`, the numerator uses :math:`F_i W_i` and the denominator
    inverse multiplies from the right. At support nodes, :math:`H(z_i)=F_i`.
    Evaluation and first derivatives use nearest-node rescaling based on
    :cite:`DZ26` (Section 2, Algorithm 1).

    Evaluation at infinity requires a nonzero sum of scalar weights or a
    nonsingular sum of matrix weights. Otherwise, :class:`ValueError` is raised.

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
        `0` for continuous-time systems, otherwise a positive sampling time in seconds.
    name
        Name of the transfer function.

    Attributes
    ----------
    form
        The evaluation form, `'bary-A'` or `'bary-B'`, inferred from the weights.
        Matrix weights of shape `(n, 1, 1)` use `'bary-A'`.
    numerator
        Barycentric numerator as a |PoleResidueTransferFunction|.
    denominator
        Barycentric denominator as a |PoleResidueTransferFunction|.
    """

    def __init__(self, nodes, values, weights, *, side='left', sampling_time=0, name=None):
        sampling_time = float(sampling_time)
        assert np.isfinite(sampling_time), 'sampling_time must be finite.'
        assert sampling_time >= 0, 'sampling_time must be nonnegative.'
        assert side in ('left', 'right'), "side must be 'left' or 'right'."
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

        matrix_values = values.reshape(n, 1, 1) if values.ndim == 1 else values
        p, m = matrix_values.shape[1:]
        denominator_size = p if side == 'left' else m
        assert weights.shape in ((n,), (n, denominator_size, denominator_size)), (
            'Incompatible barycentric weight shape for the chosen side.'
        )
        has_matrix_weights = weights.ndim == 3
        if not has_matrix_weights:
            assert np.all(weights != 0), 'Scalar weights must be nonzero.'

        # normalize weights to prevent overflow. normalization cancels later in the quotient.
        normalized_weights, _ = _normalize(weights, 'weights')
        normalized_values, value_scale = _normalize(matrix_values, 'values')
        if has_matrix_weights:
            try:
                for weight in normalized_weights:
                    _factorize(weight)
            except spla.LinAlgError as error:
                raise ValueError('Matrix weights must be nonsingular.') from error

        # redirect 1x1 matrix weights to scalar case.
        if has_matrix_weights and denominator_size == 1:
            normalized_weights = normalized_weights[:, 0, 0]

        super().__init__(m, p, self._evaluate, dtf=self._evaluate_derivative,
                         sampling_time=sampling_time, name=name)

        # retain original coefficients, independent of normalization and evaluation layout.
        self.nodes, self.values, self.weights = nodes, values, weights
        self.side = side
        self._matrix_values = matrix_values
        self._value_scale = value_scale
        self.form = 'bary-B' if normalized_weights.ndim == 3 else 'bary-A'

        # both kernels receive the same three arrays: original samples, normalized weights,
        # and cached products of normalized weights and normalized samples (or None).
        if self.form == 'bary-B':
            self._eval_values = matrix_values
            if side == 'right':
                # transpose all three inputs to reuse the left-form kernel without conjugation.
                self._eval_values = self._eval_values.swapaxes(1, 2)
                normalized_weights = normalized_weights.swapaxes(1, 2)
                normalized_values = normalized_values.swapaxes(1, 2)

            # prepare column-major blocks once; later transpose/reshape operations are views.
            normalized_weights = np.ascontiguousarray(normalized_weights.swapaxes(1, 2))
            self._eval_weights = normalized_weights.reshape(
                n, denominator_size, denominator_size
            ).swapaxes(1, 2)

            weighted_values = self._eval_weights @ normalized_values
            weighted_values = np.ascontiguousarray(weighted_values.swapaxes(1, 2))
            self._eval_weighted_values = weighted_values.reshape(
                n, -1, denominator_size
            ).swapaxes(1, 2)
        else:
            # scalar weights need division rather than matrix solves.
            self._eval_weights = normalized_weights

            # flatten (n, p, m) to (n, p*m) once for dot contractions over nodes.
            self._eval_values = matrix_values.reshape(n, -1)
            normalized_values = np.ascontiguousarray(normalized_values).reshape(n, -1)
            if p == m == 1:
                # independently of weight type, use a vector for the single SISO sample entry.
                self._eval_values = self._eval_values[:, 0]
                normalized_values = normalized_values[:, 0]

            # cache products unless underflow would lose terms amplified by derivative coefficients.
            try:
                with np.errstate(under='raise'):
                    multipliers = self._eval_weights if p == m == 1 else self._eval_weights[:, None]
                    self._eval_weighted_values = multipliers * normalized_values
            except FloatingPointError:
                self._eval_weighted_values = None

        for array in self.__dict__.values():
            if isinstance(array, np.ndarray):
                array.flags.writeable = False

    def __add__(self, other):
        if (isinstance(other, BarycentricRationalTransferFunction)
                and self.side == other.side
                and np.array_equal(self.nodes, other.nodes)
                and np.array_equal(self.weights, other.weights)):
            assert self.sampling_time == other.sampling_time
            assert self.dim_input == other.dim_input
            assert self.dim_output == other.dim_output
            return self.with_(values=(self._matrix_values + other._matrix_values).reshape(self.values.shape))
        return super().__add__(other)

    __radd__ = __add__

    def __neg__(self):
        return self.with_(values=-self.values)

    @property
    def numerator(self):
        r"""Barycentric numerator as a |PoleResidueTransferFunction|.

        The residues are :math:`w_i F_i` for scalar weights, :math:`W_i F_i`
        for the left form, and :math:`F_i W_i` for the right form. The support
        nodes are repeated as needed to represent the residues by rank-one factors.
        The original weights and sampling time are used. The feedthrough is zero.
        """
        if self.weights.ndim == 1:
            p, m = self._matrix_values.shape[1:]
            size = min(p, m)
            weighted_identity = self.weights[:, None, None] * np.eye(size, dtype=self.weights.dtype)
            if m <= p:
                left, right = self._matrix_values, weighted_identity
            else:
                left, right = weighted_identity, self._matrix_values
        elif self.side == 'left':
            left, right = self.weights, self._matrix_values
        else:
            left, right = self._matrix_values, self.weights
        return self._pole_residue_sum(left, right, 'numerator')

    @property
    def denominator(self):
        r"""Barycentric denominator as a |PoleResidueTransferFunction|.

        For scalar weights, this is :math:`D(s)=\sum_i w_i/(s-z_i)`, a SISO
        transfer function. For matrix weights, :math:`D(s)=\sum_i W_i/(s-z_i)`
        has shape `(p, p)` for the left form or `(m, m)` for the right form.
        The original weights and sampling time are used. The feedthrough is zero.
        """
        residues = self.weights[:, None, None] if self.weights.ndim == 1 else self.weights
        identity = np.broadcast_to(np.eye(residues.shape[1], dtype=residues.dtype), residues.shape)
        return self._pole_residue_sum(residues, identity, 'denominator')

    def _pole_residue_sum(self, left, right, suffix):
        """Wrap factored sum data without forming full residue products."""
        n, p, k = left.shape
        # repeat each support for the rank-one terms of left @ right; no conjugation.
        poles = np.repeat(self.nodes, k)
        b = right.reshape(n * k, -1)
        c = left.swapaxes(1, 2).reshape(n * k, p)
        return PoleResidueTransferFunction(poles, b, c, sampling_time=self.sampling_time,
                                           name=f'{self.name}_{suffix}')

    def __setstate__(self, state):
        # pickling with NumPy does not preserve the arrays' write protection.
        self.__dict__.update(state)
        for array in self.__dict__.values():
            if isinstance(array, np.ndarray):
                array.flags.writeable = False

    def _evaluate(self, s):
        """Evaluate the transfer function using the cached numerical data."""
        return self._eval(s)

    def _evaluate_derivative(self, s):
        """Evaluate the first derivative using the same cached numerical data."""
        return self._eval(s, derivative=True)

    def _eval(self, s, derivative=False):
        """Dispatch by weight type, transpose the right form and restore the output matrix shape."""
        if self.form == 'bary-B':
            value = evaluate_bary_b(
                s,
                self.nodes,
                self._eval_values,
                self._eval_weights,
                weighted_values=self._eval_weighted_values,
                value_scale=self._value_scale,
                derivative=derivative,
            )
            return value if self.side == 'left' else value.T

        value = evaluate_bary_a(
            s,
            self.nodes,
            self._eval_values,
            self._eval_weights,
            weighted_values=self._eval_weighted_values,
            value_scale=self._value_scale,
            derivative=derivative,
        )
        # restore the public (p, m) result shape after the flat scalar-weight contraction.
        return value.reshape(self._matrix_values.shape[1:])
