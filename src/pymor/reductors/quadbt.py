# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.loewner import _real_transformation, loewner_quadruple
from pymor.core.cache import CacheableObject, cached
from pymor.models.iosys import LTIModel


class QuadBTReductor(CacheableObject):
    r"""Quadrature-based balanced truncation from transfer function samples.

    Implements the frequency-domain formulation in :cite:`GGB22` (Algorithm 2).
    The underlying system is assumed to be stable and strictly proper after subtracting
    the supplied feedthrough.

    The left and right quadratures approximate the observability and reachability
    Gramians, respectively. Nodes and weights are supplied explicitly; no sampling,
    partitioning or conjugate completion is performed. Both quadratures must include
    the full contour, not just positive frequencies.

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
        Each node set must already be conjugate-closed, with conjugate sample data and
        equal weights within every pair. The feedthrough must be real.

    Notes
    -----
    Quadrature accuracy is the caller's responsibility. Approximate balancing does not
    in general guarantee stability or the classical balanced-truncation error bound.
    """

    cache_region = 'memory'

    def __init__(self, left_nodes, right_nodes, left_values, right_values, left_weights, right_weights,
                 *, derivatives=None, sampling_time=0, feedthrough=None, real=True):
        left_nodes, left_weights = self._quadrature_rule(left_nodes, left_weights, 'left', real)
        right_nodes, right_weights = self._quadrature_rule(right_nodes, right_weights, 'right', real)
        left_values = self._samples(left_values, len(left_nodes), 'left_values')
        right_values = self._samples(right_values, len(right_nodes), 'right_values')
        if left_values.shape[1:] != right_values.shape[1:]:
            raise ValueError('Left and right samples must have matching input and output dimensions.')
        self.dim_output, self.dim_input = left_values.shape[1:]
        if derivatives is not None:
            derivatives = self._samples(derivatives, len(left_nodes), 'derivatives')
            if derivatives.shape != left_values.shape:
                raise ValueError('derivatives must have the same shape as left_values.')
        sampling_time = float(sampling_time)
        if not np.isfinite(sampling_time) or sampling_time < 0:
            raise ValueError('sampling_time must be finite and nonnegative.')
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

    @staticmethod
    def _quadrature_rule(nodes, weights, name, real):
        nodes = np.array(nodes, dtype=complex, copy=True)
        weights = np.array(weights, copy=True)
        if nodes.ndim != 1 or len(nodes) == 0 or not np.all(np.isfinite(nodes)):
            raise ValueError(f'{name}_nodes must be a nonempty, finite one-dimensional array.')
        if len(np.unique(nodes)) != len(nodes):
            raise ValueError(f'{name}_nodes must be distinct within each quadrature rule.')
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
