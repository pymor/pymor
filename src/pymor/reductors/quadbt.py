# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.loewner import (
    _real_transformation,
    complete_conjugate_pairs,
    loewner_quadruple,
    sample_transfer_function,
)
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.cache import CacheableObject, cached
from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction


class QuadBTReductor(CacheableObject):
    r"""Quadrature-based balanced truncation from transfer function samples.

    Implements the frequency-domain formulation in :cite:`GGB22` (Algorithm 2).
    The underlying system is assumed to be stable and strictly proper after subtracting
    the supplied feedthrough.

    The left and right quadratures approximate the observability and reachability
    Gramians, respectively. Supply explicit weights or omit them to use trapezoidal
    quadrature on each node set independently. Use :meth:`from_model` to sample a model
    or transfer function. No automatic partitioning is performed.

    Parameters
    ----------
    left_nodes
        One-dimensional array of left quadrature nodes: complex values on the imaginary
        axis for continuous-time systems or on the unit circle for discrete-time systems.
    right_nodes
        One-dimensional array of right quadrature nodes, independent of `left_nodes`.
    left_values
        Transfer function samples at `left_nodes`, of shape `(n_left,)` for SISO data
        or `(n_left, dim_output, dim_input)` for matrix-valued data. Includes feedthrough.
    right_values
        Transfer function samples at `right_nodes`, with the same convention as `left_values`.
    left_weights
        Nonnegative quadrature weights of shape `(n_left,)`, including the measure
        normalisation: :math:`d\omega/(2\pi)` in continuous time or
        :math:`d\theta/(2\pi)` in discrete time. These are weights, not their square roots.
        If `None`, use nonuniform trapezoidal weights on the sorted frequencies in continuous
        time, or periodic trapezoidal weights on sorted angles in discrete time. Weights
        remain aligned with the original node ordering. Computed after conjugate completion.
    right_weights
        Nonnegative quadrature weights of shape `(n_right,)`, with the same convention.
    derivatives
        Optional transfer function derivatives with respect to the complex argument at
        `left_nodes`, with the same shape as `left_values`. Required at coincident
        left and right nodes; entries at other nodes are ignored.
    sampling_time
        Zero for continuous-time systems, otherwise the positive sampling time in seconds.
        Discrete-time nodes are :math:`z=e^{i\theta}`, with `theta` in radians per sample.
    feedthrough
        Known feedthrough matrix of shape `(dim_output, dim_input)`, or a scalar for SISO
        data. Subtracted from the samples and restored in the ROM. `None` means zero.
    real
        If `True`, use unitary conjugate-pair transformations to obtain a real ROM.
        Each completed node set must be conjugate-closed, with conjugate sample data and
        equal weights within every pair. The feedthrough must be real. Roundoff-sized
        discrepancies in conjugate nodes and discrete-time endpoints are canonicalised.
    conjugate
        If `True`, append missing conjugate nodes and conjugated samples, derivatives and
        supplied weights. This assumes a real underlying system. Supplied weights are
        per-node weights for the full contour; they are copied, not halved. Conjugate nodes
        and discrete-time endpoints are canonicalised as for `real=True`.
        Defaults to `False`; without completion, supply the full contour yourself.

    Notes
    -----
    The continuous-time default integrates only between the smallest and largest supplied
    frequencies. It does not estimate the omitted tails and is not the paper's ExpTrap rule.
    The discrete-time default includes the periodic wraparound interval; do not repeat the
    endpoint. Uniform full-circle grids have weights `1 / n`, without a sampling-time factor.

    Quadrature accuracy is the caller's responsibility. Approximate balancing does not
    in general guarantee stability or the classical balanced-truncation error bound.
    """

    cache_region = 'memory'

    def __init__(self, left_nodes, right_nodes, left_values, right_values, left_weights=None, right_weights=None,
                 *, derivatives=None, sampling_time=0, feedthrough=None, real=True, conjugate=False):
        sampling_time = float(sampling_time)
        if not np.isfinite(sampling_time) or sampling_time < 0:
            raise ValueError('sampling_time must be finite and nonnegative.')
        left_nodes, left_values, left_weights, derivatives = self._prepare_data(
            left_nodes, left_values, left_weights, derivatives, 'left', sampling_time, conjugate, real,
        )
        right_nodes, right_values, right_weights, _ = self._prepare_data(
            right_nodes, right_values, right_weights, None, 'right', sampling_time, conjugate, real,
        )
        if left_values.shape[1:] != right_values.shape[1:]:
            raise ValueError('Left and right samples must have matching input and output dimensions.')
        self.dim_output, self.dim_input = left_values.shape[1:]
        if feedthrough is not None:
            feedthrough = np.array(feedthrough, copy=True)
            if feedthrough.ndim == 0 and self.dim_output == self.dim_input == 1:
                feedthrough = feedthrough.reshape(1, 1)
            if feedthrough.shape != (self.dim_output, self.dim_input) or not np.all(np.isfinite(feedthrough)):
                raise ValueError('feedthrough must be a finite matrix of shape (dim_output, dim_input).')
            if real:
                if np.any(feedthrough.imag != 0):
                    raise ValueError('feedthrough must be real when real=True.')
                feedthrough = feedthrough.real
        self.__auto_init(locals())

    @classmethod
    def from_model(cls, fom, left_nodes, right_nodes, *, left_weights=None, right_weights=None,
                   derivatives=None, feedthrough=None, real=True, conjugate=True):
        """Sample a model or transfer function and construct a QuadBT reductor.

        Parameters
        ----------
        fom
            Nonparametric |TransferFunction| or model with a `transfer_function` attribute.
            The sampling time is taken from this transfer function. For an |LTIModel|,
            the feedthrough is inferred from its `D` operator unless explicitly supplied.
            For other inputs, unknown feedthrough is assumed to be zero; supply it explicitly
            when the transfer function is not strictly proper.
        left_nodes, right_nodes
            Complex sampling nodes, as in the constructor, not angular frequencies.
        left_weights, right_weights
            Optional explicit quadrature weights, as in the constructor. If omitted,
            trapezoidal rules appropriate to the model's sampling time are used.
        derivatives
            Optional derivatives at the supplied left nodes. If omitted, evaluate the
            transfer function derivative only at nodes that overlap the right grid after
            conjugate completion. Overlapping grids require derivative access or explicit data.
        feedthrough
            Optional known feedthrough, as in the constructor.
        real
            Whether to construct a real ROM, as in the constructor.
        conjugate
            Whether to complete conjugate data. Defaults to `True`, assuming a real system.
            Set both `conjugate=False` and `real=False` for complex systems.

        Returns
        -------
        reductor
            The sampled :class:`QuadBTReductor`.
        """
        tf = fom.transfer_function if hasattr(fom, 'transfer_function') else fom
        if not isinstance(tf, TransferFunction):
            raise TypeError('fom must be a TransferFunction or a model with a transfer_function.')
        if tf.parametric:
            raise ValueError('from_model requires a nonparametric transfer function.')
        sampling_time = tf.sampling_time
        left_nodes = cls._nodes(left_nodes, 'left', sampling_time, conjugate or real)
        right_nodes = cls._nodes(right_nodes, 'right', sampling_time, conjugate or real)
        left_values = sample_transfer_function(left_nodes, tf)
        right_values = sample_transfer_function(right_nodes, tf)
        if derivatives is None:
            overlap = np.isin(left_nodes, right_nodes)
            if conjugate:
                overlap |= np.isin(left_nodes.conj(), right_nodes)
            if np.any(overlap):
                if tf.dtf is None:
                    raise ValueError('Overlapping quadrature nodes require transfer function derivatives.')
                derivatives = np.zeros(left_values.shape, dtype=complex)
                for i in np.flatnonzero(overlap):
                    derivatives[i] = tf.eval_dtf(left_nodes[i])
        if feedthrough is None and isinstance(fom, LTIModel):
            feedthrough = to_matrix(fom.D, format='dense')
        return cls(left_nodes, right_nodes, left_values, right_values, left_weights, right_weights,
                   derivatives=derivatives, sampling_time=sampling_time, feedthrough=feedthrough,
                   real=real, conjugate=conjugate)

    @staticmethod
    def _nodes(nodes, name, sampling_time, canonicalize):
        nodes = np.array(nodes, dtype=complex, copy=True)
        if nodes.ndim != 1 or len(nodes) == 0 or not np.all(np.isfinite(nodes)):
            raise ValueError(f'{name}_nodes must be a nonempty, finite one-dimensional array.')
        if canonicalize:
            tolerance = 100 * np.finfo(float).eps
            if sampling_time > 0:
                for endpoint in (-1, 1):
                    nodes[np.isclose(nodes, endpoint, rtol=tolerance, atol=tolerance)] = endpoint
            for i, node in enumerate(nodes):
                matches = np.flatnonzero(np.isclose(nodes[i + 1:], node.conj(), rtol=tolerance, atol=tolerance))
                nodes[i + 1 + matches] = node.conj()
        if len(np.unique(nodes)) != len(nodes):
            raise ValueError(f'{name}_nodes must be distinct within each quadrature rule.')
        return nodes

    @classmethod
    def _prepare_data(cls, nodes, values, weights, derivatives, name, sampling_time, conjugate, real):
        nodes = cls._nodes(nodes, name, sampling_time, conjugate or real)
        values = cls._samples(values, len(nodes), f'{name}_values')
        if derivatives is not None:
            derivatives = cls._samples(derivatives, len(nodes), 'derivatives')
            if derivatives.shape != values.shape:
                raise ValueError('derivatives must have the same shape as left_values.')
        automatic_weights = weights is None
        weights = np.ones(len(nodes)) if automatic_weights else weights
        nodes, weights = cls._quadrature_rule(nodes, weights, name, False)
        if conjugate:
            extra = () if derivatives is None else (derivatives,)
            nodes, values, weights, *extra = complete_conjugate_pairs(nodes, values, weights, *extra)
            derivatives = extra[0] if extra else None
        if automatic_weights:
            weights = cls._trapezoidal_weights(nodes, sampling_time)
        nodes, weights = cls._quadrature_rule(nodes, weights, name, real)
        return nodes, values, weights, derivatives

    @staticmethod
    def _trapezoidal_weights(nodes, sampling_time):
        if sampling_time == 0:
            if len(nodes) < 2 or np.any(nodes.real != 0):
                raise ValueError('Continuous-time trapezoidal quadrature requires at least two imaginary-axis nodes.')
            order = np.argsort(nodes.imag)
            gaps = np.diff(nodes.imag[order])
            sorted_weights = np.r_[gaps[0], gaps[:-1] + gaps[1:], gaps[-1]] / (4 * np.pi)
        else:
            if not np.allclose(np.abs(nodes), 1, rtol=1e-12, atol=0):
                raise ValueError('Discrete-time trapezoidal quadrature requires unit-circle nodes.')
            order = np.argsort(np.angle(nodes))
            angles = np.angle(nodes[order])
            gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
            if np.any(gaps <= 100 * np.finfo(float).eps):
                raise ValueError('Periodic quadrature nodes must be distinct; do not repeat the endpoint.')
            sorted_weights = (gaps + np.roll(gaps, 1)) / (4 * np.pi)
        weights = np.empty(len(nodes))
        weights[order] = sorted_weights
        return weights

    @staticmethod
    def _quadrature_rule(nodes, weights, name, real):
        weights = np.array(weights, copy=True)
        if weights.shape != nodes.shape or not np.isrealobj(weights) \
                or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError(f'{name}_weights must be finite, real, nonnegative and aligned with nodes.')
        weights = weights.astype(float)
        if real:
            transformation = _real_transformation(nodes)
            # Weighting commutes with realification only for equal conjugate-pair weights.
            if not np.allclose(transformation * weights, weights[:, np.newaxis] * transformation,
                               rtol=1e-12, atol=0):
                raise ValueError(f'{name}_weights must be equal at conjugate nodes when real=True.')
        return nodes, weights

    @staticmethod
    def _samples(values, size, name):
        values = np.array(values, copy=True)
        if values.ndim == 1:
            values = values[:, np.newaxis, np.newaxis]
        if values.ndim != 3 or values.shape[0] != size or 0 in values.shape \
                or not np.all(np.isfinite(values)):
            raise ValueError(f'{name} must contain finite samples of shape (n,) or (n, dim_output, dim_input).')
        return values

    @cached
    def quadrature_matrices(self):
        r"""Return the weighted data matrices :math:`(K, M, B_d, C_d)`.

        For weighted resolvent factors :math:`O` and :math:`R`, these matrices equal
        :math:`OER`, :math:`OAR`, :math:`OB` and :math:`CR`, respectively. They are
        constructed without accessing the underlying system matrices. When `real=True`,
        unitary transformations give their real counterparts.
        """
        D = 0 if self.feedthrough is None else self.feedthrough
        L, Ls, V, W = loewner_quadruple(
            self.left_nodes, self.right_nodes, self.left_values - D, self.right_values - D,
            derivatives=self.derivatives, real=self.real,
        )
        wl = np.repeat(np.sqrt(self.left_weights), self.dim_output)[:, np.newaxis]
        wr = np.repeat(np.sqrt(self.right_weights), self.dim_input)[np.newaxis, :]
        return -wl * L * wr, -wl * Ls * wr, wl * V, W * wr

    @cached
    def _svd(self):
        return spla.svd(self.quadrature_matrices()[0], full_matrices=False)

    def reduce(self, r, rank_tol=1e-12):
        """Construct a reduced model using a truncated SVD of the weighted Loewner matrix.

        Parameters
        ----------
        r
            Positive target order, not exceeding the numerical rank of the weighted
            Loewner matrix. Requests exceeding that rank raise a `ValueError`.
        rank_tol
            Relative singular-value threshold in `[0, 1)`. Singular values at or below
            `rank_tol` times the largest singular value are discarded. This is a numerical
            rank tolerance, not a model-reduction error tolerance.

        Returns
        -------
        rom
            Reduced |LTIModel| with identity mass matrix, the supplied feedthrough and
            the supplied sampling time.
        """
        if not isinstance(r, int | np.integer) or isinstance(r, bool | np.bool_) or r <= 0:
            raise ValueError('r must be a positive integer.')
        if not np.isscalar(rank_tol) or not np.isrealobj(rank_tol) \
                or not np.isfinite(rank_tol) or not 0 <= rank_tol < 1:
            raise ValueError('rank_tol must be finite and in [0, 1).')
        U, sv, Vh = self._svd()
        rank = np.count_nonzero(sv > rank_tol * sv[0])
        if r > rank:
            raise ValueError(f'r={r} exceeds the numerical rank {rank} of the weighted Loewner matrix.')
        _, M, B, C = self.quadrature_matrices()
        scale = 1 / np.sqrt(sv[:r])
        X = scale[:, np.newaxis] * U[:, :r].conj().T
        Y = Vh[:r].conj().T * scale
        return LTIModel.from_matrices(X @ M @ Y, X @ B, C @ Y,
                                      D=self.feedthrough, sampling_time=self.sampling_time)
