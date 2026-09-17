# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import itertools

import numpy as np
import pytest

from pymor.algorithms.loewner import (
    _sample_transfer_function,
    complete_conjugate_pairs,
    loewner_matrices,
    loewner_matrix,
    loewner_matrix_nd,
    loewner_quadruple,
    partition_frequencies,
)
from pymor.core.exceptions import AccuracyError
from pymor.models.transfer_function import TransferFunction

pytestmark = pytest.mark.builtin


def _loewner_matrix_nd_reference(sampling_values, samples, interpolation_indices):
    interpolation_sets = tuple(set(indices) for indices in interpolation_indices)
    interpolation_grid = tuple(itertools.product(*interpolation_indices))
    rows = []
    for row in np.ndindex(samples.shape):
        if all(index in indices for index, indices in zip(row, interpolation_sets, strict=True)):
            continue
        entries = []
        for column in interpolation_grid:
            factor = 1
            for dimension, (row_index, column_index) in enumerate(zip(row, column, strict=True)):
                if row_index in interpolation_sets[dimension]:
                    if row_index != column_index:
                        factor = 0
                        break
                else:
                    factor /= (sampling_values[dimension][row_index]
                               - sampling_values[dimension][column_index])
            entries.append((samples[row] - samples[column]) * factor)
        rows.append(entries)
    return np.array(rows)


@pytest.mark.parametrize('derivative', [False, True])
def test_sample_transfer_function_on_parametric_grid(derivative):
    fom = TransferFunction(
        1,
        1,
        lambda s, mu: np.array([[1 / (s + mu['mu'][0])]]),
        dtf=lambda s, mu: np.array([[-1 / (s + mu['mu'][0])**2]]),
        parameters={'mu': 1},
    )
    sampling_values = [np.array([1j, 2j]), np.array([1., 3.])]

    samples = _sample_transfer_function(sampling_values, fom, derivative=derivative)

    assert samples.shape == (2, 2, 1, 1)
    for i, s in enumerate(sampling_values[0]):
        for j, mu in enumerate(sampling_values[1]):
            expected = -1 / (s + mu)**2 if derivative else 1 / (s + mu)
            assert np.allclose(samples[i, j], [[expected]])


@pytest.mark.parametrize('num_data', [0, 1, 2])
def test_complete_conjugate_pairs(num_data):
    nodes = np.array([1j, 2j, -1j])
    samples = np.array([1 + 2j, 3 + 4j, 1 - 2j])
    weights = np.array([2., 3., 2.])
    data = (samples, weights)[:num_data]

    nodes, *completed_data = complete_conjugate_pairs(nodes, *data)

    assert np.array_equal(nodes, [1j, 2j, -1j, -2j])
    expected_data = ([1 + 2j, 3 + 4j, 1 - 2j, 3 - 4j], [2., 3., 2., 3.])[:num_data]
    for values, expected in zip(completed_data, expected_data, strict=True):
        assert np.array_equal(values, expected)


@pytest.mark.parametrize(('ordering', 'partitioning', 'left', 'right'), [
    ('regular', 'even-odd', [0, 2, 4], [1, 3]),
    ('regular', 'half-half', [0, 1, 2], [3, 4]),
    ('random', 'even-odd', [2, 3, 1], [4, 0]),
    ('random', 'half-half', [2, 4, 3], [0, 1]),
])
def test_partition_frequencies_without_conjugates(ordering, partitioning, left, right):
    nodes = 1j * np.arange(1, 6)
    samples = 1 / (nodes + 1)

    left_indices, right_indices = partition_frequencies(
        nodes, samples, partitioning=partitioning, ordering=ordering, force_real=False,
    )

    assert np.array_equal(left_indices, left)
    assert np.array_equal(right_indices, right)


def test_loewner_input_validation():
    nodes = np.array([1., 2., 3.])
    samples = 1 / nodes

    with pytest.raises(AssertionError, match='one-dimensional'):
        complete_conjugate_pairs(nodes[:, np.newaxis], samples)
    with pytest.raises(AssertionError, match='aligned'):
        complete_conjugate_pairs(nodes, samples[:-1])
    with pytest.raises(AssertionError, match='scalar tensor'):
        loewner_matrix_nd([nodes], samples[:, np.newaxis], [np.array([0])])
    with pytest.raises(AssertionError, match='out-of-bounds'):
        loewner_matrix_nd([nodes], samples, [np.array([3])])
    with pytest.raises(AssertionError, match='duplicate'):
        loewner_matrix_nd([nodes], samples, [np.array([0, 0])])


def test_loewner_matrix_and_shifted_matrix():
    left_nodes = np.array([1j, 2j])
    right_nodes = np.array([3j, 4j, 5j])
    left_values = np.array([2 + 1j, 3 - 2j])[:, np.newaxis]
    right_values = np.array([1 - 1j, 4 + 2j, 2])

    L, Ls = loewner_matrices(left_nodes, right_nodes, left_values, right_values)
    denominator = left_nodes[:, np.newaxis] - right_nodes
    assert np.allclose(L, (left_values - right_values) / denominator)
    assert np.allclose(
        Ls,
        ((left_nodes * left_values[:, 0])[:, np.newaxis] - right_nodes * right_values) / denominator,
    )
    assert np.allclose(loewner_matrix(left_nodes, right_nodes, left_values, right_values), L)


def test_loewner_matrix_hermite_entries():
    nodes = np.array([1., 2.])
    values = np.array([3., 5.])
    derivatives = np.array([7., 11.])[:, np.newaxis]

    L, Ls = loewner_matrices(nodes, nodes, values[:, np.newaxis], values, derivative_terms=derivatives)
    assert np.allclose(np.diag(L), derivatives[:, 0])
    assert np.allclose(np.diag(Ls), values + nodes * derivatives[:, 0])
    with pytest.raises(AssertionError, match='require derivative_terms'):
        loewner_matrix(nodes, nodes, values[:, np.newaxis], values)


def test_loewner_matrix_nd_matches_reference(rng):
    sampling_values = [rng.random(3), rng.random(4), rng.random(5)]
    samples = rng.random((3, 4, 5))
    interpolation_indices = [np.array([0, 2]), np.array([1]), np.array([0, 3])]

    L = loewner_matrix_nd(sampling_values, samples, interpolation_indices)
    reference = _loewner_matrix_nd_reference(sampling_values, samples, interpolation_indices)
    assert L.shape == (56, 4)
    assert np.allclose(L, reference)


def test_loewner_matrix_nd_one_dimensional(rng):
    nodes = rng.random(8)
    samples = rng.random(8)
    right = np.array([1, 4, 6])
    left = np.array([0, 2, 3, 5, 7])

    L = loewner_matrix_nd([nodes], samples, [right])
    reference = loewner_matrix(nodes[left], nodes[right], samples[left, np.newaxis], samples[np.newaxis, right])
    assert np.allclose(L, reference)


def test_loewner_quadruple_full_mimo(rng):
    left_nodes = 1j * np.arange(1, 4)
    right_nodes = 1j * np.arange(4, 8)
    left_values = rng.random((3, 2, 3))
    right_values = rng.random((4, 2, 3))

    L, Ls, V, W = loewner_quadruple(left_nodes, right_nodes, left_values, right_values)
    denominator = left_nodes[:, np.newaxis, np.newaxis, np.newaxis] \
        - right_nodes[np.newaxis, :, np.newaxis, np.newaxis]
    reference_L = (left_values[:, np.newaxis] - right_values[np.newaxis]) / denominator
    reference_Ls = (
        (left_nodes[:, np.newaxis, np.newaxis] * left_values)[:, np.newaxis]
        - (right_nodes[:, np.newaxis, np.newaxis] * right_values)[np.newaxis]
    ) / denominator

    assert L.shape == Ls.shape == (6, 12)
    assert V.shape == (6, 3)
    assert W.shape == (2, 12)
    assert np.allclose(L, np.transpose(reference_L, (0, 2, 1, 3)).reshape(6, 12))
    assert np.allclose(Ls, np.transpose(reference_Ls, (0, 2, 1, 3)).reshape(6, 12))
    assert np.allclose(V, left_values.reshape(6, 3))
    assert np.allclose(W, np.transpose(right_values, (1, 0, 2)).reshape(2, 12))


def test_loewner_quadruple_tangential_hermite(rng):
    nodes = np.array([1., 2., 3.])
    values = rng.random((3, 2, 2))
    derivatives = rng.random((3, 2, 2))
    left_directions = rng.random((3, 2))
    right_directions = rng.random((3, 2))

    L, Ls, V, W = loewner_quadruple(
        nodes, nodes, values, values,
        left_directions=left_directions,
        right_directions=right_directions,
        derivatives=derivatives,
    )

    reference_L = np.empty((3, 3))
    reference_Ls = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                reference_L[i, j] = left_directions[i] @ derivatives[i] @ right_directions[j]
                reference_Ls[i, j] = left_directions[i] @ (
                    values[i] + nodes[i] * derivatives[i]
                ) @ right_directions[j]
            else:
                reference_L[i, j] = left_directions[i] @ (values[i] - values[j]) \
                    @ right_directions[j] / (nodes[i] - nodes[j])
                reference_Ls[i, j] = left_directions[i] @ (
                    nodes[i] * values[i] - nodes[j] * values[j]
                ) @ right_directions[j] / (nodes[i] - nodes[j])

    assert np.allclose(L, reference_L)
    assert np.allclose(Ls, reference_Ls)
    assert np.allclose(V, np.einsum('ip,ipm->im', left_directions, values))
    assert np.allclose(W, np.einsum('ipm,im->pi', values, right_directions))


def test_loewner_quadruple_unitary_realification():
    left_nodes = np.array([0, 1j, -1j])
    right_nodes = np.array([2, 2 + 2j, 2 - 2j])

    def values(nodes):
        return np.array([[[1 / (node + 3), 2 / (node + 4)],
                          [3 / (node + 5), 4 / (node + 6)]] for node in nodes])

    L, Ls, *_ = loewner_quadruple(left_nodes, right_nodes, values(left_nodes), values(right_nodes))
    real_quadruple = loewner_quadruple(
        left_nodes, right_nodes, values(left_nodes), values(right_nodes), force_real=True
    )
    real_L, real_Ls, *_ = real_quadruple

    assert all(not np.iscomplexobj(matrix) for matrix in real_quadruple)
    assert np.allclose(np.linalg.svd(L, compute_uv=False), np.linalg.svd(real_L, compute_uv=False))
    assert np.allclose(np.linalg.svd(Ls, compute_uv=False), np.linalg.svd(real_Ls, compute_uv=False))


def test_loewner_quadruple_rejects_nonreal_transformation():
    left_nodes = np.array([1j, -1j])
    right_nodes = np.array([2j, -2j])
    with pytest.raises(AccuracyError, match='not real'):
        loewner_quadruple(left_nodes, right_nodes, np.array([1, 2j]), np.array([3, 4j]), force_real=True)
