# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import pickle

import numpy as np
import pytest
import scipy.linalg as spla

from pymor.models.barycentric import BarycentricRationalTransferFunction
from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction
from pymortests.models.iosys_add_sub_mul import (
    assert_tf_add,
    assert_tf_mul,
    assert_tf_sub,
    get_model,
    name_list,
)

pytestmark = pytest.mark.builtin


def test_evaluation_at_and_away_from_support_nodes():
    C = np.array([[2., 1.], [1., -1.], [0., 2.]])
    tf = BarycentricRationalTransferFunction([0., 1.], np.diag([1., 3.]), C, np.eye(2))
    for s in [0., 1., 1e-10, 1 + 1e-10, 2j]:
        expected = C @ np.diag([1 / (s + 1), 3 / (s + 2)])
        assert np.allclose(tf.eval_tf(s), expected)


@pytest.mark.parametrize('parametric', [False, True])
@pytest.mark.parametrize('sampling_time', [0, 0.1])
@pytest.mark.parametrize('barycentric_first', [False, True])
@pytest.mark.parametrize('name', name_list)
def test_arithmetic(name, barycentric_first, sampling_time, parametric):
    tf = BarycentricRationalTransferFunction([0.], [[1.]], [[2.]], [[1.]], sampling_time=sampling_time)
    other = get_model(name, sampling_time, parametric)
    m1, m2 = (tf, other) if barycentric_first else (other, tf)
    for result, check in ((m1 + m2, assert_tf_add), (m1 - m2, assert_tf_sub), (m1 * m2, assert_tf_mul)):
        assert type(result) is TransferFunction
        assert result.sampling_time == sampling_time
        check(m1, m2, result, parametric)


@pytest.mark.parametrize('barycentric_first', [False, True])
def test_rectangular_composition(barycentric_first):
    tf = BarycentricRationalTransferFunction([0.], [[1., 2.]], np.ones((3, 1)), [[1.], [0.]])
    m, p = (4, 2) if barycentric_first else (3, 4)
    other = LTIModel.from_matrices(np.array([[-2.]]), np.ones((1, m)), np.ones((p, 1)))
    m1, m2 = (tf, other) if barycentric_first else (other, tf)
    result = m1 * m2
    assert result.dim_input == m2.dim_input
    assert result.dim_output == m1.dim_output
    assert_tf_mul(m1, m2, result, False)


@pytest.mark.parametrize('matrix_weights', [False, True])
def test_full_block_interpolation(matrix_weights, rng):
    nodes = np.array([1., 2., 3.])
    values = rng.normal(size=(3, 3, 2))
    weights = rng.normal(size=(3, 2, 2)) if matrix_weights else np.array([1., -2., 4.])[:, None, None] * np.eye(2)
    tf = BarycentricRationalTransferFunction(
        np.repeat(nodes, 2), weights.reshape(6, 2), values.transpose(1, 0, 2).reshape(3, 6), np.tile(np.eye(2), (1, 3))
    )
    for s, value in zip(nodes, values, strict=True):
        assert np.allclose(tf.eval_tf(s), value)
    for s in [0.5, 2j]:
        numerator = sum(F @ W / (s - z) for z, F, W in zip(nodes, values, weights, strict=True))
        denominator = np.eye(2) + sum(W / (s - z) for z, W in zip(nodes, weights, strict=True))
        assert np.allclose(tf.eval_tf(s) @ denominator, numerator)


def test_tangential_interpolation_and_derivatives(rng):
    nodes = np.array([1., 2., 3.])
    values = rng.normal(size=(3, 3, 2))
    directions = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    weights = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    C = np.einsum('npm,nm->pn', values, directions)
    tf = BarycentricRationalTransferFunction(nodes, weights, C, directions.T)
    for z, F, r in zip(nodes, values, directions, strict=True):
        assert np.allclose(tf.eval_tf(z) @ r, F @ r)
        assert not np.allclose(tf.eval_tf(z), F)
    for s in [*nodes, 0.3 + 1j]:
        difference = (tf.eval_tf(s + 1e-5) - tf.eval_tf(s - 1e-5)) / 2e-5
        assert np.allclose(tf.eval_dtf(s), difference, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
def test_frequency_scaling(scale):
    tf = BarycentricRationalTransferFunction(
        scale * np.array([0., 1.]), scale * np.array([[-1.], [4.]]), [[1., 0.25]], [[1., 1.]]
    )
    for s in [0., 1., 0.3j]:
        assert np.allclose(tf.eval_tf(scale * s), [[1 / (s + 1)**2]])
        assert np.allclose(scale * tf.eval_dtf(scale * s), [[-2 / (s + 1)**3]])


@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int64, np.uint64])
def test_integer_factors(dtype):
    weight = np.iinfo(dtype).max // 2
    tf = BarycentricRationalTransferFunction(
        np.array([0], dtype=dtype), np.array([[weight]], dtype=dtype),
        np.array([[1]], dtype=dtype), np.array([[4]], dtype=dtype)
    )
    assert np.allclose(tf.eval_tf(0), [[0.25]])
    np.testing.assert_allclose(tf.eval_dtf(0), [[-1 / (16 * weight)]], atol=0)


def test_zero_and_infinity():
    zero = BarycentricRationalTransferFunction([], np.empty((0, 2)), np.empty((3, 0)), np.empty((2, 0)))
    tf = BarycentricRationalTransferFunction([0.], [[1.]], [[2.]], [[1.]])
    for s in [0., 1j, np.inf]:
        assert np.array_equal(zero.eval_tf(s), np.zeros((3, 2)))
        assert np.array_equal(zero.eval_dtf(s), np.zeros((3, 2)))
    assert np.array_equal(tf.eval_tf(np.inf), np.zeros((1, 1)))
    assert np.array_equal(tf.eval_dtf(np.inf), np.zeros((1, 1)))


def test_copy_and_pickle():
    nodes, B, C, R = np.array([0.]), np.array([[1.]]), np.array([[2.]]), np.array([[1.]])
    tf = BarycentricRationalTransferFunction(nodes, B, C, R, sampling_time=0.1, name='example')
    for original, stored in zip((nodes, B, C, R), (tf.nodes, tf.B, tf.C, tf.R), strict=True):
        original[:] = 10
        assert not stored.flags.writeable
    assert np.allclose(tf.eval_tf(1), [[1.]])
    assert np.allclose(tf.with_(C=np.array([[4.]])).eval_tf(1), [[2.]])
    restored = pickle.loads(pickle.dumps(tf))
    assert restored.sampling_time == 0.1
    assert restored.name == 'example'
    for factor in (restored.nodes, restored.B, restored.C, restored.R):
        assert not factor.flags.writeable
    assert np.allclose(restored.eval_tf(1), [[1.]])
    assert np.allclose(restored.eval_dtf(1), [[-0.5]])


@pytest.mark.parametrize('args', [
    ([[0.]], [[1.]], [[1.]], [[1.]]),
    ([0.], [1.], [[1.]], [[1.]]),
    ([0.], [[1., 2.]], [[1.]], [[1.]]),
    ([0.], [[]], [[1.]], []),
    ([np.inf], [[1.]], [[1.]], [[1.]]),
    ([0.], [[np.nan]], [[1.]], [[1.]]),
])
def test_validation(args):
    with pytest.raises(ValueError):
        BarycentricRationalTransferFunction(*args)


@pytest.mark.parametrize('sampling_time', [-1, np.inf, np.nan])
def test_invalid_sampling_time(sampling_time):
    with pytest.raises(ValueError, match='sampling_time'):
        BarycentricRationalTransferFunction([0.], [[1.]], [[1.]], [[1.]], sampling_time=sampling_time)


@pytest.mark.parametrize('s', [np.nan, [1., 2.]])
def test_invalid_evaluation_argument(s):
    tf = BarycentricRationalTransferFunction([0.], [[1.]], [[1.]], [[1.]])
    with pytest.raises(ValueError):
        tf.eval_tf(s)
    with pytest.raises(ValueError):
        tf.eval_dtf(s)


def test_evaluation_at_a_pole():
    tf = BarycentricRationalTransferFunction([0.], [[1.]], [[1.]], [[1.]])
    with pytest.raises(spla.LinAlgError):
        tf.eval_tf(-1.)
