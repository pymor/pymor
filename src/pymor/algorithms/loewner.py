# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import product

import numpy as np

from pymor.models.transfer_function import TransferFunction
from pymor.tools.random import new_rng

def _nodes(nodes, name):
    nodes = np.asarray(nodes)
    if nodes.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional.')
    if len(nodes) == 0:
        raise ValueError(f'{name} must not be empty.')
    return nodes


def sample_transfer_function(sampling_values, samples_or_fom):
    """Sample a transfer function on a Cartesian grid when needed.

    Parameters
    ----------
    sampling_values
        A one-dimensional |NumPy array| or a sequence of such arrays. The first array contains
        Laplace-variable values; subsequent arrays contain parameter values.
    samples_or_fom
        Sample data, a |TransferFunction|, or a model with a `transfer_function` attribute.

    Returns
    -------
    samples
        Sample data. For transfer-function input, the shape is
        ``tuple(map(len, sampling_values)) + (dim_output, dim_input)``.
    """
    fom = samples_or_fom.transfer_function if hasattr(samples_or_fom, 'transfer_function') else samples_or_fom
    if not isinstance(fom, TransferFunction):
        return np.asarray(samples_or_fom)

    if isinstance(sampling_values, np.ndarray):
        sampling_values = (sampling_values,)
    else:
        sampling_values = tuple(sampling_values)
    if len(sampling_values) != fom.parameters.dim + 1:
        raise ValueError('sampling_values must contain the Laplace variable and one array per parameter.')
    if any(values.ndim != 1 or len(values) == 0 for values in sampling_values):
        raise ValueError('sampling_values must contain non-empty one-dimensional arrays.')

    sample_shape = tuple(map(len, sampling_values))
    samples = [
        fom.eval_tf(values[0], mu=fom.parameters.parse(values[1:]))
        for values in product(*sampling_values)
    ]
    return np.array(samples).reshape(sample_shape + (fom.dim_output, fom.dim_input))


def complete_conjugate_pairs(nodes, samples, *data):
    """Append missing conjugate nodes and their conjugate-aligned data."""
    nodes = _nodes(nodes, 'nodes')
    data = (np.asarray(samples), *(np.asarray(values) for values in data))
    if any(values.shape[:1] != (len(nodes),) for values in data):
        raise ValueError('Data must be aligned with nodes.')

    for i, node in enumerate(nodes):
        if node.conj() not in nodes:
            nodes = np.append(nodes, node.conj())
            data = tuple(np.concatenate((values, values[i:i + 1].conj())) for values in data)

    return nodes, *data


def partition_frequencies(nodes, samples, partitioning='even-odd', ordering='regular', conjugate=True):
    """Partition frequency samples into left and right Loewner data sets.

    Complex-conjugate nodes are assigned to the same set when `conjugate` is `True`. In that
    case, `nodes` must already contain the corresponding conjugate samples.
    """
    if not isinstance(partitioning, str):
        return tuple(np.asarray(indices) for indices in partitioning)

    if conjugate:
        positive_imaginary = np.flatnonzero(nodes.imag > 0)
        real = np.flatnonzero(nodes.imag == 0)
        if ordering == 'magnitude':
            positive_imaginary = positive_imaginary[np.argsort([np.linalg.norm(samples[i])
                                                                for i in positive_imaginary])]
            real = real[np.argsort([np.linalg.norm(samples[i]) for i in real])]
        elif ordering == 'random':
            rng = new_rng(0)
            rng.shuffle(positive_imaginary)
            rng.shuffle(real)

        if partitioning == 'even-odd':
            left = np.concatenate((real[::2], positive_imaginary[::2]))
            right = np.concatenate((real[1::2], positive_imaginary[1::2]))
        elif partitioning == 'half-half':
            real = np.array_split(real, 2)
            positive_imaginary = np.array_split(positive_imaginary, 2)
            left = np.concatenate((real[0], positive_imaginary[0]))
            right = np.concatenate((real[1], positive_imaginary[1]))
        else:
            raise ValueError(f'Unknown partitioning: {partitioning}.')

        left_conjugates = np.concatenate([np.flatnonzero(nodes == nodes[i].conj()) for i in left if nodes[i].imag]) \
            if np.any(nodes[left].imag) else np.array([], dtype=int)
        right_conjugates = np.concatenate([np.flatnonzero(nodes == nodes[i].conj()) for i in right if nodes[i].imag]) \
            if np.any(nodes[right].imag) else np.array([], dtype=int)
        return np.concatenate((left, left_conjugates)), np.concatenate((right, right_conjugates))

    if ordering == 'magnitude':
        indices = np.argsort([np.linalg.norm(sample) for sample in samples])
    elif ordering == 'random':
        indices = new_rng(0).permutation(len(nodes))
    elif ordering == 'regular':
        indices = np.arange(len(nodes))
    else:
        raise ValueError(f'Unknown ordering: {ordering}.')

    if partitioning == 'even-odd':
        return indices[::2], indices[1::2]
    if partitioning == 'half-half':
        return tuple(np.array_split(indices, 2))
    raise ValueError(f'Unknown partitioning: {partitioning}.')


def loewner_matrix(left_nodes, right_nodes, left_terms, right_terms, derivative_terms=None):
    r"""Construct a Loewner matrix from pairwise terms.

    The returned matrix contains the divided differences

    .. math::
        \mathbb{L}_{ij} = \frac{A_{ij} - B_{ij}}{\mu_i - \lambda_j}.

    `left_terms` and `right_terms` must broadcast to an array whose first two dimensions
    are `(len(left_nodes), len(right_nodes))`. This permits scalar, matrix-valued and
    tangentially projected data without additional construction modes.

    Parameters
    ----------
    left_nodes
        One-dimensional array of left interpolation nodes :math:`\mu_i`.
    right_nodes
        One-dimensional array of right interpolation nodes :math:`\lambda_j`.
    left_terms
        Array containing :math:`A_{ij}` or values broadcastable to it.
    right_terms
        Array containing :math:`B_{ij}` or values broadcastable to it.
    derivative_terms
        Optional array of derivatives, broadcastable to the pairwise term shape. At entries
        with identical left and right nodes, these values replace the undefined divided differences.

    Returns
    -------
    L
        Loewner matrix. Trailing dimensions of the broadcast pair terms are retained.
    """
    left_nodes = _nodes(left_nodes, 'left_nodes')
    right_nodes = _nodes(right_nodes, 'right_nodes')
    dtype_args = (left_nodes, right_nodes, left_terms, right_terms, float)
    if derivative_terms is not None:
        dtype_args += (derivative_terms,)
    dtype = np.result_type(*dtype_args)

    try:
        L = np.asarray(left_terms, dtype=dtype) - np.asarray(right_terms, dtype=dtype)
    except ValueError as error:
        raise ValueError('left_terms and right_terms are not broadcastable.') from error
    if L.ndim < 2 or L.shape[:2] != (len(left_nodes), len(right_nodes)):
        raise ValueError('Pair terms must broadcast to shape (len(left_nodes), len(right_nodes), ...).')

    denominator = left_nodes[:, np.newaxis] - right_nodes[np.newaxis, :]
    collision = denominator == 0
    trailing_axes = (1,) * (L.ndim - 2)
    denominator = denominator.reshape(denominator.shape + trailing_axes)

    if np.any(collision):
        if derivative_terms is None:
            raise ValueError('Coincident left and right nodes require derivative_terms.')
        collision = np.broadcast_to(collision.reshape(collision.shape + trailing_axes), L.shape)
        if not np.allclose(L[collision], 0):
            raise ValueError('Left and right terms differ at coincident nodes.')
        try:
            derivative_terms = np.broadcast_to(np.asarray(derivative_terms, dtype=dtype), L.shape)
        except ValueError as error:
            raise ValueError('derivative_terms are not broadcastable to the pairwise term shape.') from error
        np.divide(L, denominator, out=L, where=~collision)
        L[collision] = derivative_terms[collision]
    else:
        L /= denominator

    return L


def loewner_matrices(left_nodes, right_nodes, left_terms, right_terms, derivative_terms=None):
    r"""Construct Loewner and shifted Loewner matrices in one pass.

    Parameters are the same as for :func:`loewner_matrix`. The shifted matrix is computed
    from :math:`\mathbb{L}_s = \operatorname{diag}(\mu)\mathbb{L} + B`, which also gives
    the correct Hermite limit at coincident nodes.

    Returns
    -------
    L
        Loewner matrix.
    Ls
        Shifted Loewner matrix.
    """
    L = loewner_matrix(left_nodes, right_nodes, left_terms, right_terms, derivative_terms)
    left_nodes = np.asarray(left_nodes)
    left_nodes = left_nodes.reshape((len(left_nodes), 1) + (1,) * (L.ndim - 2))
    Ls = left_nodes * L
    try:
        Ls += right_terms
    except ValueError as error:
        raise ValueError('right_terms are not broadcastable to the Loewner matrix shape.') from error
    return L, Ls


def _interpolation_indices(indices, size, dimension):
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f'interpolation_indices[{dimension}] must be a one-dimensional integer array.')
    if len(indices) == 0:
        raise ValueError(f'interpolation_indices[{dimension}] must not be empty.')
    if np.any(indices < 0) or np.any(indices >= size):
        raise ValueError(f'interpolation_indices[{dimension}] contains an out-of-bounds index.')
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f'interpolation_indices[{dimension}] contains duplicate indices.')
    return indices


def _modified_cauchy_matrix(nodes, interpolation_indices, dtype):
    least_squares = np.ones(len(nodes), dtype=bool)
    least_squares[interpolation_indices] = False
    denominator = nodes[least_squares, np.newaxis] - nodes[interpolation_indices]
    if np.any(denominator == 0):
        raise ValueError('Interpolation and least-squares nodes must have distinct values.')

    C = np.zeros((len(nodes), len(interpolation_indices)), dtype=dtype)
    C[interpolation_indices] = np.eye(len(interpolation_indices))
    C[least_squares] = 1 / denominator
    return C


def loewner_matrix_nd(sampling_values, samples, interpolation_indices):
    """Construct a higher-dimensional Loewner matrix for Cartesian scalar data.

    Parameters
    ----------
    sampling_values
        Sequence of one-dimensional sampling-node arrays, one per variable.
    samples
        Scalar sample tensor with shape ``tuple(map(len, sampling_values))``.
    interpolation_indices
        Sequence of index arrays selecting interpolation nodes for each variable.

    Returns
    -------
    L
        Two-dimensional higher-dimensional Loewner matrix. Columns correspond to the
        Cartesian interpolation grid. Rows correspond to all other sampled grid points.
    """
    if not isinstance(sampling_values, (list, tuple)) or len(sampling_values) == 0:
        raise ValueError('sampling_values must be a non-empty sequence.')
    if not isinstance(interpolation_indices, (list, tuple)) \
            or len(interpolation_indices) != len(sampling_values):
        raise ValueError('interpolation_indices must contain one array per sampling dimension.')

    sampling_values = tuple(_nodes(nodes, f'sampling_values[{i}]')
                            for i, nodes in enumerate(sampling_values))
    samples = np.asarray(samples)
    expected_shape = tuple(len(nodes) for nodes in sampling_values)
    if samples.ndim != len(sampling_values) or samples.shape != expected_shape:
        raise ValueError(f'samples must be a scalar tensor of shape {expected_shape}.')
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
        if len(matches) != 1:
            raise ValueError('Nodes must contain unique complex conjugate pairs when real=True.')
        j = matches[0]
        if visited[j]:
            raise ValueError('Nodes must contain unique complex conjugate pairs when real=True.')
        scale = 1 / np.sqrt(2)
        transformation[i, i] = scale
        transformation[i, j] = scale
        transformation[j, i] = -1j * scale
        transformation[j, j] = 1j * scale
        visited[i] = visited[j] = True
    return transformation


def _real_array(array, name):
    if not np.iscomplexobj(array):
        return array
    tolerance = 1000 * np.finfo(array.real.dtype).eps * max(1, np.max(np.abs(array)))
    if np.max(np.abs(array.imag)) > tolerance:
        raise ValueError(f'{name} is not real after conjugate transformation.')
    return array.real


def loewner_quadruple(left_nodes, right_nodes, left_values, right_values, *,
                       left_directions=None, right_directions=None, derivatives=None, real=False):
    """Construct a Loewner quadruple from already partitioned sample data.

    Parameters
    ----------
    left_nodes
        One-dimensional array of left interpolation nodes.
    right_nodes
        One-dimensional array of right interpolation nodes.
    left_values
        SISO values of shape `(n_left,)` or MIMO values of shape `(n_left, dim_output, dim_input)`.
    right_values
        SISO values of shape `(n_right,)` or MIMO values of shape
        `(n_right, dim_output, dim_input)`.
    left_directions
        Optional left tangential directions of shape `(n_left, dim_output)`.
    right_directions
        Optional right tangential directions of shape `(n_right, dim_input)`.
    derivatives
        Optional unprojected derivatives at `left_nodes`, with the same shape as `left_values`.
        Derivatives are required for coincident left and right nodes.
    real
        If `True`, transform conjugate-closed data to real matrices using unitary pair transforms.
        Values, derivatives and directions at conjugate nodes must be conjugates.

    Returns
    -------
    L
        Loewner matrix.
    Ls
        Shifted Loewner matrix.
    V
        Left interpolation data.
    W
        Right interpolation data.
    """
    left_nodes = _nodes(left_nodes, 'left_nodes')
    right_nodes = _nodes(right_nodes, 'right_nodes')
    left_values = np.asarray(left_values)
    right_values = np.asarray(right_values)
    if left_values.shape[:1] != (len(left_nodes),) or right_values.shape[:1] != (len(right_nodes),):
        raise ValueError('Sample values must be aligned with their nodes.')
    if left_values.ndim != right_values.ndim or left_values.shape[1:] != right_values.shape[1:]:
        raise ValueError('Left and right sample values must have matching trailing dimensions.')
    if left_values.ndim not in (1, 3):
        raise ValueError('Sample values must be SISO arrays or sample-major matrices.')
    if derivatives is not None:
        derivatives = np.asarray(derivatives)
        if derivatives.shape != left_values.shape:
            raise ValueError('derivatives must have the same shape as left_values.')

    tangential = left_directions is not None or right_directions is not None
    full_mimo = left_values.ndim == 3 and not tangential
    if tangential:
        if left_directions is None or right_directions is None:
            raise ValueError('Both left_directions and right_directions are required.')
        if left_values.ndim != 3:
            raise ValueError('Tangential directions require matrix-valued samples.')
        dim_output, dim_input = left_values.shape[1:]
        left_directions = np.asarray(left_directions)
        right_directions = np.asarray(right_directions)
        if left_directions.shape != (len(left_nodes), dim_output):
            raise ValueError('left_directions has the wrong shape.')
        if right_directions.shape != (len(right_nodes), dim_input):
            raise ValueError('right_directions has the wrong shape.')
        V = np.einsum('ip,ipm->im', left_directions, left_values)
        W = np.einsum('jpm,jm->pj', right_values, right_directions)
        left_terms = V @ right_directions.T
        right_terms = left_directions @ W
        derivative_terms = None if derivatives is None else (
            np.einsum('ip,ipm->im', left_directions, derivatives) @ right_directions.T
        )
    else:
        left_terms = left_values[:, np.newaxis, ...]
        right_terms = right_values[np.newaxis, ...]
        derivative_terms = None if derivatives is None else derivatives[:, np.newaxis, ...]
        if left_values.ndim == 1:
            V = left_values[:, np.newaxis]
            W = right_values[np.newaxis, :]
        else:
            V = left_values[:, np.newaxis, ...]
            W = right_values[np.newaxis, ...]

    L, Ls = loewner_matrices(left_nodes, right_nodes, left_terms, right_terms, derivative_terms)

    if real:
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
        L = np.transpose(L, (0, 2, 1, 3)).reshape(len(left_nodes) * dim_output,
                                                    len(right_nodes) * dim_input)
        Ls = np.transpose(Ls, (0, 2, 1, 3)).reshape(len(left_nodes) * dim_output,
                                                     len(right_nodes) * dim_input)
        V = V[:, 0].reshape(len(left_nodes) * dim_output, dim_input)
        W = np.transpose(W[0], (1, 0, 2)).reshape(dim_output, len(right_nodes) * dim_input)

    return L, Ls, V, W
