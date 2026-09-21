# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import product

import numpy as np

from pymor.core.exceptions import AccuracyError
from pymor.core.logger import getLogger
from pymor.models.transfer_function import TransferFunction
from pymor.tools.random import new_rng

logger = getLogger('pymor.algorithms.loewner')


def sample_transfer_function(sampling_values, fom, *, derivative=False):
    """Sample a |TransferFunction| or its derivative on a Cartesian grid.

    Parameters
    ----------
    sampling_values
        A one-dimensional |NumPy array| or a sequence of such arrays. The first array contains
        Laplace-variable values; subsequent arrays contain parameter values.
    fom
        A |TransferFunction| or a model with a `transfer_function` attribute.
    derivative
        If `True`, sample the derivative with respect to the complex frequency argument.

    Returns
    -------
    samples
        Sample data of shape `tuple(map(len, sampling_values)) + (dim_output, dim_input)`.
        A single frequency array produces shape `(n, dim_output, dim_input)`.
    """
    fom = fom.transfer_function if hasattr(fom, 'transfer_function') else fom
    assert isinstance(fom, TransferFunction), 'fom must be a TransferFunction or a model with a transfer_function.'

    if isinstance(sampling_values, np.ndarray):
        sampling_values = (sampling_values,)
    else:
        sampling_values = tuple(sampling_values)
    assert len(sampling_values) == fom.parameters.dim + 1, \
        'sampling_values must contain the Laplace variable and one array per parameter.'
    assert all(values.ndim == 1 for values in sampling_values), 'sampling_values must contain one-dimensional arrays.'
    assert all(len(values) > 0 for values in sampling_values), 'sampling_values must contain non-empty arrays.'

    sample_shape = tuple(len(s) for s in sampling_values)
    evaluate = fom.eval_dtf if derivative else fom.eval_tf
    with logger.block(f'Sampling {"transfer function derivative" if derivative else "transfer function"} ...'):
        samples = [
            evaluate(values[0], mu=fom.parameters.parse(values[1:]))
            for values in product(*sampling_values)
        ]
    return np.array(samples).reshape(sample_shape + (fom.dim_output, fom.dim_input))


def complete_conjugate_pairs(nodes, *data):
    """Complete nodes and associated data with complex conjugate pairs.

    For each node whose complex conjugate is missing, append its conjugate and the
    elementwise conjugates of the corresponding entries in each array in `data`.
    Existing entries retain their order; new entries are appended in input-node order.
    For example, complete samples and weights together with::

        nodes, samples, weights = complete_conjugate_pairs(nodes, samples, weights)

    .. note::
        Nodes are compared using exact equality. Already supplied conjugate data are not
        checked for consistency. Real nodes are not duplicated, and input arrays are not modified.

    Parameters
    ----------
    nodes
        Nonempty one-dimensional |NumPy array| of sampling nodes.
    data
        Zero or more |NumPy arrays| passed as positional arguments, e.g. samples, derivatives,
        quadrature weights or tangential directions. Each array must have shape `(len(nodes), ...)`;
        trailing dimensions are preserved.

    Returns
    -------
    completed_nodes
        One-dimensional |NumPy array| containing the original and appended nodes.
    completed_data
        Completed |NumPy arrays| in input order, returned as separate entries in
        `(completed_nodes, *completed_data)`, not a nested tuple. With no data arguments,
        the return value is `(completed_nodes,)`.
    """
    nodes = _as_nodes(nodes, 'nodes')
    data = tuple(np.asarray(values) for values in data)
    assert all(values.shape[:1] == (len(nodes),) for values in data), 'Data must be aligned with nodes.'

    num_nodes = len(nodes)
    for i, node in enumerate(nodes):
        if node.conj() not in nodes:
            nodes = np.append(nodes, node.conj())
            data = tuple(np.concatenate((values, values[i:i + 1].conj())) for values in data)

    if len(nodes) != num_nodes:
        logger.info(f'Added {len(nodes) - num_nodes} complex conjugates to the data.')
    return nodes, *data


def partition_frequencies(nodes, samples, partitioning='even-odd', ordering='regular', force_real=True):
    """Partition frequency samples into left and right Loewner data sets.

    Parameters
    ----------
    nodes
        One-dimensional |NumPy array| of complex sampling nodes of shape `(n,)`.
    samples
        |NumPy array| of sampled values of shape `(n, ...)`, aligned with `nodes`.
        Used to determine sample norms when `ordering` is `'magnitude'`.
    partitioning
        Partitioning rule or a tuple `(left_indices, right_indices)` of index arrays.
        Available rules are:

        - `'even-odd'`: assign alternating entries to the left and right sets.
        - `'half-half'`: assign the first half to the left set and the remainder to the
          right set. For an odd number of entries, the left set receives one extra entry.

        Explicit index arrays are returned without validation or reordering; `ordering`
        and `force_real` are ignored in this case.
    ordering
        Ordering applied before a partitioning rule:

        - `'regular'`: preserve the input order; no sorting by frequency is performed.
        - `'magnitude'`: order by increasing sample norm.
        - `'random'`: use a reproducible random ordering with seed zero.
    force_real
        If `True`, keep complex conjugate nodes in the same set. The nodes and samples
        must already be closed under conjugation. Real nodes and nodes in the upper
        half-plane are ordered and partitioned separately; the corresponding lower
        half-plane nodes are then appended to each set. If `False`, all nodes are
        ordered and partitioned together.

    Returns
    -------
    left_indices
        One-dimensional |NumPy array| of indices into `nodes` and `samples` for the left set.
    right_indices
        One-dimensional |NumPy array| of indices into `nodes` and `samples` for the right set.
    """
    if not isinstance(partitioning, str):
        logger.info('Using supplied frequency partition ...')
        return tuple(np.asarray(indices) for indices in partitioning)

    assert partitioning in ('even-odd', 'half-half'), f'Unknown partitioning: {partitioning}.'
    assert ordering in ('magnitude', 'random', 'regular'), f'Unknown ordering: {ordering}.'

    logger.info(f"Partitioning frequency samples using '{partitioning}' ...")
    if force_real:
        positive_imaginary = np.flatnonzero(nodes.imag > 0)
        real = np.flatnonzero(nodes.imag == 0)
        if ordering == 'magnitude':
            positive_imaginary = positive_imaginary[np.argsort(
                [np.linalg.norm(samples[i]) for i in positive_imaginary]
            )]
            real = real[np.argsort([np.linalg.norm(samples[i]) for i in real])]
        elif ordering == 'random':
            rng = new_rng(seed_seq=0)
            rng.shuffle(positive_imaginary)
            rng.shuffle(real)

        if partitioning == 'even-odd':
            left = np.concatenate((real[::2], positive_imaginary[::2]))
            right = np.concatenate((real[1::2], positive_imaginary[1::2]))
        else:
            real = np.array_split(real, 2)
            positive_imaginary = np.array_split(positive_imaginary, 2)
            left = np.concatenate((real[0], positive_imaginary[0]))
            right = np.concatenate((real[1], positive_imaginary[1]))

        left_conjugates = np.concatenate([np.flatnonzero(nodes == nodes[i].conj()) for i in left if nodes[i].imag]) \
            if np.any(nodes[left].imag) else np.array([], dtype=int)
        right_conjugates = np.concatenate([np.flatnonzero(nodes == nodes[i].conj()) for i in right if nodes[i].imag]) \
            if np.any(nodes[right].imag) else np.array([], dtype=int)
        return np.concatenate((left, left_conjugates)), np.concatenate((right, right_conjugates))

    if ordering == 'magnitude':
        indices = np.argsort([np.linalg.norm(sample) for sample in samples])
    elif ordering == 'random':
        indices = new_rng(seed_seq=0).permutation(len(nodes))
    else:
        indices = np.arange(len(nodes))

    if partitioning == 'even-odd':
        return indices[::2], indices[1::2]
    return tuple(np.array_split(indices, 2))


def loewner_matrix(left_nodes, right_nodes, left_terms, right_terms, derivative_terms=None):
    r"""Construct a Loewner matrix from pairwise terms.

    The returned array contains the divided differences

    .. math::
        \mathbb{L}_{ij} = \frac{A_{ij} - B_{ij}}{\mu_i - \lambda_j}.

    Scalar, matrix-valued and tangentially projected terms are supported through broadcasting.
    Use :func:`loewner_quadruple` to assemble pairwise terms directly from transfer function data.

    .. note::
        Coincident nodes are detected using exact equality. Nearly coincident nodes are treated
        by divided differences and may suffer from cancellation.

    Parameters
    ----------
    left_nodes
        Nonempty one-dimensional |NumPy array| of left interpolation nodes :math:`\mu_i`,
        of shape `(n_left,)`.
    right_nodes
        Nonempty one-dimensional |NumPy array| of right interpolation nodes :math:`\lambda_j`,
        of shape `(n_right,)`.
    left_terms
        |NumPy array| containing :math:`A_{ij}`. Together with `right_terms`, must broadcast
        to shape `(n_left, n_right, ...)`. For scalar function samples, a column of left
        values of shape `(n_left, 1)` can be supplied.
    right_terms
        |NumPy array| containing :math:`B_{ij}`, broadcastable with `left_terms` as above.
        For scalar function samples, a row of right values of shape `(1, n_right)` can be
        supplied.
    derivative_terms
        Optional |NumPy array| of derivative terms with respect to the complex argument,
        broadcastable to the pairwise term shape. Required when a left and right node
        coincide. At these entries, the left and right terms must agree and the supplied
        derivative replaces the divided difference. Other derivative entries are ignored.

    Returns
    -------
    L
        |NumPy array| of shape `(n_left, n_right, ...)` containing the Loewner matrix.
        Trailing dimensions of the broadcast terms are retained, not flattened into blocks.
    """
    logger.info('Constructing Loewner matrix ...')
    left_nodes = _as_nodes(left_nodes, 'left_nodes')
    right_nodes = _as_nodes(right_nodes, 'right_nodes')
    dtype_args = (left_nodes, right_nodes, left_terms, right_terms, float)
    if derivative_terms is not None:
        dtype_args += (derivative_terms,)
    dtype = np.result_type(*dtype_args)

    if left_terms.shape == (len(left_nodes),) and right_terms.shape == (len(right_nodes),):
        left_terms, right_terms = left_terms[:, np.newaxis], right_terms[np.newaxis]

    L = np.asarray(left_terms, dtype=dtype) - np.asarray(right_terms, dtype=dtype)
    assert L.shape[:2] == (len(left_nodes), len(right_nodes)), \
        'Pair terms must broadcast to shape (len(left_nodes), len(right_nodes), ...).'

    denominator = left_nodes[:, np.newaxis] - right_nodes[np.newaxis, :]
    collision = denominator == 0
    trailing_axes = (1,) * (L.ndim - 2)
    denominator = denominator.reshape(denominator.shape + trailing_axes)

    if np.any(collision):
        assert derivative_terms is not None, 'Coincident left and right nodes require derivative_terms.'
        collision = np.broadcast_to(collision.reshape(collision.shape + trailing_axes), L.shape)
        assert np.allclose(L[collision], 0), 'Left and right terms differ at coincident nodes.'
        derivative_terms = np.broadcast_to(np.asarray(derivative_terms, dtype=dtype), L.shape)
        np.divide(L, denominator, out=L, where=~collision)
        L[collision] = derivative_terms[collision]
    else:
        L /= denominator

    return L


def loewner_matrices(left_nodes, right_nodes, left_terms, right_terms, derivative_terms=None):
    r"""Construct Loewner and shifted Loewner matrices from pairwise terms.

    For distinct left and right nodes, the entries are

    .. math::
        \mathbb{L}_{ij} = \frac{A_{ij} - B_{ij}}{\mu_i - \lambda_j},
        \qquad
        (\mathbb{L}_s)_{ij} =
            \frac{\mu_i A_{ij} - \lambda_j B_{ij}}{\mu_i - \lambda_j}.

    The shifted matrix is computed as
    :math:`(\mathbb{L}_s)_{ij} = \mu_i\mathbb{L}_{ij} + B_{ij}`. At coincident nodes,
    this gives the Hermite value :math:`\mu_i D_{ij} + B_{ij}`, where :math:`D_{ij}`
    is the supplied derivative term.

    Parameters
    ----------
    left_nodes
        Nonempty one-dimensional |NumPy array| of left interpolation nodes :math:`\mu_i`,
        of shape `(n_left,)`.
    right_nodes
        Nonempty one-dimensional |NumPy array| of right interpolation nodes :math:`\lambda_j`,
        of shape `(n_right,)`.
    left_terms
        |NumPy array| containing :math:`A_{ij}`. Together with `right_terms`, must broadcast
        to shape `(n_left, n_right, ...)`.
    right_terms
        |NumPy array| containing :math:`B_{ij}`, broadcastable with `left_terms` as above.
    derivative_terms
        Optional |NumPy array| containing :math:`D_{ij}`, broadcastable to the pairwise
        term shape. Required at coincident nodes, where the left and right terms must agree.
        See :func:`loewner_matrix` for the derivative and coincidence conventions.

    Returns
    -------
    L
        Loewner matrix as a |NumPy array| of shape `(n_left, n_right, ...)`.
    Ls
        Shifted Loewner matrix as a |NumPy array| of the same shape as `L`.
    """
    L = loewner_matrix(left_nodes, right_nodes, left_terms, right_terms, derivative_terms=derivative_terms)
    logger.info('Constructing shifted Loewner matrix ...')
    left_nodes = np.asarray(left_nodes)
    left_nodes = left_nodes.reshape((len(left_nodes), 1) + (1,) * (L.ndim - 2))
    Ls = left_nodes * L
    Ls += right_terms
    return L, Ls


def loewner_matrix_nd(sampling_values, samples, interpolation_indices):
    """Construct a higher-dimensional Loewner matrix for Cartesian scalar data.

    Assemble the Loewner matrix used in the parametric AAA algorithm of :cite:`CRBG23`.
    The interpolation nodes form a Cartesian subgrid of the sampling grid. For one variable,
    this reduces to :func:`loewner_matrix` with interpolation nodes on the right and all
    remaining nodes on the left.

    Parameters
    ----------
    sampling_values
        Nonempty list or tuple of nonempty one-dimensional |NumPy arrays|, one per variable.
        The first variable is typically the complex frequency; subsequent variables are
        parameters. Interpolation and non-interpolation node values must not coincide in
        any variable.
    samples
        |NumPy array| of scalar samples with shape ``tuple(map(len, sampling_values))``.
        `samples[i, j, ...]` corresponds to the nodes
        `(sampling_values[0][i], sampling_values[1][j], ...)`. Matrix-valued samples must
        be projected to scalar data before calling this function.
    interpolation_indices
        List or tuple of nonempty one-dimensional integer |NumPy arrays|, one per variable.
        Each array selects interpolation nodes from the corresponding `sampling_values`
        array and must contain distinct, valid indices. For one variable, at least one
        sampling node must remain outside the interpolation set.

    Returns
    -------
    L
        Two-dimensional |NumPy array| of shape `(N - K, K)`, where `N` and `K` are the
        numbers of points in the full sampling grid and the Cartesian interpolation grid,
        respectively. Columns follow the supplied interpolation-index order. Rows correspond
        to all sampling-grid points outside the Cartesian interpolation grid. Both grids are
        flattened in NumPy C order, with the last variable varying fastest.

    """
    assert isinstance(sampling_values, (list, tuple)), 'sampling_values must be a list or tuple.'
    assert len(sampling_values) > 0, 'sampling_values must not be empty.'
    assert isinstance(interpolation_indices, (list, tuple)), 'interpolation_indices must be a list or tuple.'
    assert len(interpolation_indices) == len(sampling_values), \
        'interpolation_indices must contain one array per sampling dimension.'

    sampling_values = tuple(_as_nodes(nodes, f'sampling_values[{i}]')
                            for i, nodes in enumerate(sampling_values))
    samples = np.asarray(samples)
    expected_shape = tuple(len(nodes) for nodes in sampling_values)
    assert samples.shape == expected_shape, f'samples must be a scalar tensor of shape {expected_shape}.'
    interpolation_indices = tuple(
        _interpolation_indices(indices, len(nodes), i)
        for i, (nodes, indices) in enumerate(zip(sampling_values, interpolation_indices, strict=True))
    )

    if len(sampling_values) == 1:
        right = interpolation_indices[0]
        left_mask = np.ones(len(sampling_values[0]), dtype=bool)
        left_mask[right] = False
        left = np.flatnonzero(left_mask)
        return loewner_matrix(
            sampling_values[0][left], sampling_values[0][right],
            samples[left, np.newaxis], samples[np.newaxis, right],
        )

    logger.info('Constructing multidimensional Loewner matrix ...')
    dtype = np.result_type(samples, *sampling_values, float)
    samples = samples.astype(dtype, copy=False)
    cauchy = np.ones((1, 1), dtype=dtype)
    interpolation_rows = np.ones(1, dtype=bool)
    for nodes, indices in zip(sampling_values, interpolation_indices, strict=True):
        cauchy = np.kron(cauchy, _modified_cauchy_matrix(nodes, indices, dtype))
        mask = np.zeros(len(nodes), dtype=bool)
        mask[indices] = True
        interpolation_rows = np.kron(interpolation_rows, mask)

    interpolation_samples = samples[np.ix_(*interpolation_indices)].reshape(-1)
    keep = ~interpolation_rows
    cauchy = cauchy[keep]
    L = samples.reshape(-1)[keep, np.newaxis] - interpolation_samples
    L *= cauchy
    return L


def loewner_quadruple(left_nodes, right_nodes, left_values, right_values, *,
                      left_directions=None, right_directions=None, derivatives=None, force_real=False):
    r"""Construct a Loewner quadruple from partitioned transfer function samples.

    Assemble the Loewner matrix, shifted Loewner matrix and left and right interpolation
    data as in :cite:`ALI17`. Supports SISO data, full-block MIMO data and tangential MIMO
    data. No sampling, partitioning or conjugate completion is performed.

    .. note::
        In the tangential case, before realification,

        .. math::
            V_i = \ell_i^T H(\mu_i), \qquad W_j = H(\lambda_j)r_j,
            \qquad \mathbb{L}_{ij} =
            \frac{V_i r_j - \ell_i^T W_j}{\mu_i - \lambda_j}.

        Directions are neither normalised nor conjugated internally. For full-block MIMO data,
        rows are ordered by left node then output, and columns by right node then input.
        For a square quadruple, the descriptor-system sign convention is :math:`E=-L`,
        :math:`A=-L_s`, :math:`B=V` and :math:`C=W`.

    Parameters
    ----------
    left_nodes
        Nonempty one-dimensional |NumPy array| of left interpolation nodes :math:`\mu_i`,
        of shape `(n_left,)`.
    right_nodes
        Nonempty one-dimensional |NumPy array| of right interpolation nodes :math:`\lambda_j`,
        of shape `(n_right,)`.
    left_values
        |NumPy array| containing transfer function values at `left_nodes`. Shape `(n_left,)`
        for SISO data or `(n_left, p, m)` for a system with `p` outputs and `m` inputs. With
        tangential directions, already projected values `V` of shape `(n_left, m)` are also
        accepted.
    right_values
        |NumPy array| containing transfer function values at `right_nodes`. Shape `(n_right,)`
        for SISO data or `(n_right, p, m)` for MIMO data. With tangential directions, already
        projected values `W` of shape `(p, n_right)` are also accepted.
    left_directions
        Optional |NumPy array| of left tangential directions of shape `(n_left, p)`.
        Requires `right_directions`. Each row stores a direction :math:`\ell_i^T`, applied
        without complex conjugation.
    right_directions
        Optional |NumPy array| of right tangential directions of shape `(n_right, m)`.
        Requires `left_directions`. Each row stores a direction :math:`r_j^T`, used as a
        column when multiplying a transfer function value.
    derivatives
        Optional |NumPy array| of unprojected transfer function derivatives with respect to
        the complex argument at `left_nodes`, with the shape `(n_left, p, m)` for MIMO data.
        Required when a left and right node coincide. Entries at other nodes are ignored.
        Tangential projections of these derivatives are performed internally.
    force_real
        If `True`, transform the quadruple to real arrays using unitary conjugate-pair
        transformations. Each node set must already contain unique complex conjugate pairs,
        and values, used derivatives and directions at conjugate nodes must be conjugates.
        Values and directions at real nodes must be real. No conjugate data are appended.

    Returns
    -------
    L
        Loewner matrix as a |NumPy array|. Shape `(n_left * p, n_right * m)` for full-block
        MIMO data, or `(n_left, n_right)` for SISO or tangential data.
    Ls
        Shifted Loewner matrix as a |NumPy array| of the same shape as `L`.
    V
        Left interpolation data as a |NumPy array|. Shape `(n_left * p, m)` for full-block
        MIMO data, `(n_left, m)` for tangential data, or `(n_left, 1)` for SISO data.
    W
        Right interpolation data as a |NumPy array|. Shape `(p, n_right * m)` for full-block
        MIMO data, `(p, n_right)` for tangential data, or `(1, n_right)` for SISO data.

    Raises
    ------
    AccuracyError
        If `force_real=True` cannot produce real matrices up to roundoff.
    """
    left_nodes = _as_nodes(left_nodes, 'left_nodes')
    right_nodes = _as_nodes(right_nodes, 'right_nodes')
    left_values = np.asarray(left_values)
    right_values = np.asarray(right_values)
    tangential = left_directions is not None or right_directions is not None
    full_mimo = left_values.ndim == 3 and not tangential
    if tangential:
        assert left_directions is not None, 'left_directions are required when right_directions are given.'
        assert right_directions is not None, 'right_directions are required when left_directions are given.'
        left_directions = np.asarray(left_directions)
        right_directions = np.asarray(right_directions)
        assert left_directions.ndim == right_directions.ndim == 2, 'Tangential directions must be matrices.'
        assert left_directions.shape[0] == len(left_nodes), 'left_directions has the wrong shape.'
        assert right_directions.shape[0] == len(right_nodes), 'right_directions has the wrong shape.'
        dim_output = left_directions.shape[1]
        dim_input = right_directions.shape[1]
        if left_values.ndim == right_values.ndim == 2:
            assert left_values.shape == (len(left_nodes), dim_input), 'left_values has the wrong shape.'
            assert right_values.shape == (dim_output, len(right_nodes)), 'right_values has the wrong shape.'
            logger.info('Using already tangentially sampled data ...')
            V, W = left_values, right_values
        else:
            assert left_values.shape == (len(left_nodes), dim_output, dim_input), 'left_values has the wrong shape.'
            assert right_values.shape == (len(right_nodes), dim_output, dim_input), 'right_values has the wrong shape.'
            logger.info('Projecting tangential transfer function data ...')
            V = np.einsum('ip,ipm->im', left_directions, left_values)
            W = np.einsum('jpm,jm->pj', right_values, right_directions)
        left_terms = V @ right_directions.T
        right_terms = left_directions @ W
        if derivatives is not None:
            derivatives = np.asarray(derivatives)
            assert derivatives.shape == (len(left_nodes), dim_output, dim_input), 'derivatives has the wrong shape.'
            derivative_terms = np.einsum('ip,ipm->im', left_directions, derivatives) @ right_directions.T
        else:
            derivative_terms = None
    else:
        assert left_values.shape[:1] == (len(left_nodes),), 'left_values must be aligned with left_nodes.'
        assert right_values.shape[:1] == (len(right_nodes),), 'right_values must be aligned with right_nodes.'
        assert left_values.shape[1:] == right_values.shape[1:], \
            'Left and right sample values must have matching trailing dimensions.'
        assert left_values.ndim in (1, 3), 'Sample values must be SISO arrays or sample-major matrices.'
        if derivatives is not None:
            derivatives = np.asarray(derivatives)
            assert derivatives.shape == left_values.shape, 'derivatives must have the same shape as left_values.'
        left_terms = left_values[:, np.newaxis, ...]
        right_terms = right_values[np.newaxis, ...]
        derivative_terms = None if derivatives is None else derivatives[:, np.newaxis, ...]
        if left_values.ndim == 1:
            logger.info('Using SISO transfer function data ...')
            V = left_values[:, np.newaxis]
            W = right_values[np.newaxis, :]
        else:
            logger.info('Using MIMO transfer function data ...')
            V = left_values[:, np.newaxis, ...]
            W = right_values[np.newaxis, ...]

    with logger.block('Constructing Loewner matrices ...'):
        L, Ls = loewner_matrices(left_nodes, right_nodes, left_terms, right_terms, derivative_terms=derivative_terms)

    if force_real:
        logger.info('Keeping it real ...')
        TL = _real_transformation(left_nodes)
        TR = _real_transformation(right_nodes)
        if full_mimo:
            L = np.tensordot(TL, L, axes=(1, 0))
            L = np.transpose(np.tensordot(L, TR.conj().T, axes=(1, 0)), (0, 3, 1, 2))
            Ls = np.tensordot(TL, Ls, axes=(1, 0))
            Ls = np.transpose(np.tensordot(Ls, TR.conj().T, axes=(1, 0)), (0, 3, 1, 2))
            V = np.tensordot(TL, V, axes=(1, 0))
            W = np.transpose(np.tensordot(W, TR.conj().T, axes=(1, 0)), (0, 3, 1, 2))
        else:
            L = TL @ L @ TR.conj().T
            Ls = TL @ Ls @ TR.conj().T
            V = TL @ V
            W = W @ TR.conj().T
        L = _real_array(L, 'L')
        Ls = _real_array(Ls, 'Ls')
        V = _real_array(V, 'V')
        W = _real_array(W, 'W')

    if full_mimo:
        dim_output, dim_input = left_values.shape[1:]
        L = np.transpose(L, (0, 2, 1, 3)).reshape(len(left_nodes) * dim_output, len(right_nodes) * dim_input)
        Ls = np.transpose(Ls, (0, 2, 1, 3)).reshape(len(left_nodes) * dim_output, len(right_nodes) * dim_input)
        V = V[:, 0].reshape(len(left_nodes) * dim_output, dim_input)
        W = np.transpose(W[0], (1, 0, 2)).reshape(dim_output, len(right_nodes) * dim_input)

    return L, Ls, V, W


def _as_nodes(nodes, name):
    nodes = np.asarray(nodes)
    assert nodes.ndim == 1, f'{name} must be one-dimensional.'
    assert len(nodes) > 0, f'{name} must not be empty.'
    return nodes


def _interpolation_indices(indices, size, dimension):
    indices = np.asarray(indices)
    assert indices.ndim == 1, f'interpolation_indices[{dimension}] must be one-dimensional.'
    assert np.issubdtype(indices.dtype, np.integer), f'interpolation_indices[{dimension}] must contain integers.'
    assert len(indices) > 0, f'interpolation_indices[{dimension}] must not be empty.'
    assert np.all(indices >= 0), f'interpolation_indices[{dimension}] contains a negative index.'
    assert np.all(indices < size), f'interpolation_indices[{dimension}] contains an out-of-bounds index.'
    assert len(np.unique(indices)) == len(indices), f'interpolation_indices[{dimension}] contains duplicate indices.'
    return indices


def _modified_cauchy_matrix(nodes, interpolation_indices, dtype):
    least_squares = np.ones(len(nodes), dtype=bool)
    least_squares[interpolation_indices] = False
    denominator = nodes[least_squares, np.newaxis] - nodes[interpolation_indices]
    assert np.all(denominator != 0), 'Interpolation and least-squares nodes must have distinct values.'

    C = np.zeros((len(nodes), len(interpolation_indices)), dtype=dtype)
    C[interpolation_indices] = np.eye(len(interpolation_indices))
    C[least_squares] = 1 / denominator
    return C


def _real_transformation(nodes):
    transformation = np.zeros((len(nodes), len(nodes)), dtype=np.complex128)
    visited = np.zeros(len(nodes), dtype=bool)
    dtype = nodes.real.dtype
    precision = np.finfo(dtype).eps if np.issubdtype(dtype, np.inexact) else np.finfo(float).eps
    tolerance = 100 * precision
    for i, node in enumerate(nodes):
        if visited[i]:
            continue
        if np.imag(node) == 0:
            transformation[i, i] = 1
            visited[i] = True
            continue
        matches = np.flatnonzero(np.isclose(nodes, np.conj(node), rtol=tolerance, atol=tolerance))
        matches = matches[matches != i]
        assert len(matches) == 1, 'Nodes must contain unique complex conjugate pairs when force_real=True.'
        j = matches[0]
        assert not visited[j], 'Nodes must contain unique complex conjugate pairs when force_real=True.'
        scale = 1 / np.sqrt(2)
        transformation[i, i] = scale
        transformation[i, j] = scale
        transformation[j, i] = -1j * scale
        transformation[j, j] = 1j * scale
        visited[i] = visited[j] = True
    return transformation


def _real_array(array, name):
    tolerance = 1000 * np.finfo(array.real.dtype).eps * max(1, np.max(np.abs(array)))
    if np.max(np.abs(array.imag)) > tolerance:
        raise AccuracyError(f'{name} is not real after conjugate transformation.')
    return array.real
