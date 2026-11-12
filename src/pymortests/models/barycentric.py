# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import copy
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


def test_sampled_scalar_evaluation():
    tf = BarycentricRationalTransferFunction([0, 1], [1, 0.5], [-1, 2])
    for s in [0, 1, 1e-15, 1 + 1e-15, 2j]:
        assert np.allclose(tf.eval_tf(s), [[1 / (s + 1)]])


@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
def test_sampled_derivative_near_nodes(scale):
    tf = BarycentricRationalTransferFunction(scale * np.array([0., 1.]), [1, 0.5], [-1, 2])
    for s in [0., 1., 1e-16, 1 + 1e-15, 0.3j]:
        np.testing.assert_allclose(scale * tf.eval_dtf(scale * s), [[-1 / (s + 1)**2]], atol=1e-14)


@pytest.mark.parametrize(('p', 'm'), [(1, 3), (3, 1), (2, 3)])
def test_sampled_matrix_values_with_scalar_weights(p, m, rng):
    residue = rng.normal(size=(p, m)) + 1j * rng.normal(size=(p, m))
    feedthrough = rng.normal(size=(p, m))
    values = np.stack([feedthrough + residue, feedthrough + residue / 2])
    tf = BarycentricRationalTransferFunction([0, 1], values, [-1, 2])
    assert (tf.dim_output, tf.dim_input) == (p, m)
    for s in [0, 1, 1e-16, 1 + 1e-15, 2j]:
        np.testing.assert_allclose(tf.eval_tf(s), feedthrough + residue / (s + 1))
        np.testing.assert_allclose(tf.eval_dtf(s), -residue / (s + 1)**2)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize(('p', 'm'), [(2, 3), (3, 2), (2, 2)])
def test_sampled_matrix_weights(side, p, m, rng):
    nodes = np.array([-1., 0.5, 2.])
    values = rng.normal(size=(3, p, m)) + 1j * rng.normal(size=(3, p, m))
    size = p if side == 'left' else m
    weights = np.eye(size) + 0.2 * (rng.normal(size=(3, size, size)) + 1j * rng.normal(size=(3, size, size)))
    tf = BarycentricRationalTransferFunction(nodes, values, weights, side=side)
    coefficients = weights @ values if side == 'left' else values @ weights
    for s in [*nodes, -1 + 1e-15, 0.5 + 1e-15j, 0.3j]:
        basis = np.array([np.prod(s - np.delete(nodes, i)) for i in range(3)])
        dbasis = np.array([sum(s - np.delete(nodes, i)) for i in range(3)])
        N = np.einsum('i,ijk->jk', basis, coefficients)
        D = np.einsum('i,ijk->jk', basis, weights)
        dN = np.einsum('i,ijk->jk', dbasis, coefficients)
        dD = np.einsum('i,ijk->jk', dbasis, weights)
        if side == 'left':
            expected = spla.solve(D, N)
            derivative = spla.solve(D, dN - dD @ expected)
        else:
            expected = spla.solve(D.T, N.T).T
            derivative = spla.solve(D.T, (dN - expected @ dD).T).T
        np.testing.assert_allclose(tf.eval_tf(s), expected, atol=1e-12)
        np.testing.assert_allclose(tf.eval_dtf(s), derivative, atol=1e-12)
    for node, value in zip(nodes, values, strict=True):
        np.testing.assert_allclose(tf.eval_tf(node), value)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_scalar_and_identity_weights_agree(side, rng):
    values = rng.normal(size=(3, 2, 3)) + 1j * rng.normal(size=(3, 2, 3))
    weights = np.array([1 + 1j, -2, 3j])
    size = 2 if side == 'left' else 3
    scalar = BarycentricRationalTransferFunction([0, 1, 2], values, weights)
    block = BarycentricRationalTransferFunction([0, 1, 2], values, weights[:, None, None] * np.eye(size), side=side)
    for s in [0, 1 + 1e-15j, 2, 3j]:
        np.testing.assert_allclose(block.eval_tf(s), scalar.eval_tf(s))
        np.testing.assert_allclose(block.eval_dtf(s), scalar.eval_dtf(s))


@pytest.mark.parametrize('parametric', [False, True])
@pytest.mark.parametrize('sampling_time', [0, 0.1])
@pytest.mark.parametrize('barycentric_first', [False, True])
@pytest.mark.parametrize('name', name_list)
def test_arithmetic(name, barycentric_first, sampling_time, parametric):
    tf = BarycentricRationalTransferFunction([0, 1], [2, 1], [-1, 2], sampling_time=sampling_time)
    other = get_model(name, sampling_time, parametric)
    m1, m2 = (tf, other) if barycentric_first else (other, tf)
    for result, check in ((m1 + m2, assert_tf_add), (m1 - m2, assert_tf_sub), (m1 * m2, assert_tf_mul)):
        assert type(result) is TransferFunction
        assert result.sampling_time == sampling_time
        check(m1, m2, result, parametric)
    negated = -tf
    assert type(negated) is TransferFunction
    np.testing.assert_allclose(negated.eval_tf(2j), -tf.eval_tf(2j))
    np.testing.assert_allclose(negated.eval_dtf(2j), -tf.eval_dtf(2j))


@pytest.mark.parametrize('barycentric_first', [False, True])
def test_rectangular_composition(barycentric_first):
    tf = BarycentricRationalTransferFunction([0, 1], [np.ones((3, 2)), 0.5 * np.ones((3, 2))], [-1, 2])
    m, p = (4, 2) if barycentric_first else (3, 4)
    other = LTIModel.from_matrices(np.array([[-2.]]), np.ones((1, m)), np.ones((p, 1)))
    m1, m2 = (tf, other) if barycentric_first else (other, tf)
    result = m1 * m2
    assert result.dim_input == m2.dim_input
    assert result.dim_output == m1.dim_output
    assert_tf_mul(m1, m2, result, False)


@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int64, np.uint64])
def test_integer_data(dtype):
    weight = np.iinfo(dtype).max // 2
    tf = BarycentricRationalTransferFunction(
        np.array([0, 1], dtype=dtype), np.array([2, 3], dtype=dtype), np.array([weight, weight], dtype=dtype)
    )
    np.testing.assert_allclose(tf.eval_tf(2), [[8 / 3]])
    np.testing.assert_allclose(tf.eval_dtf(2), [[-1 / 9]])


@pytest.mark.parametrize('block', [False, True])
@pytest.mark.parametrize('scale', [1e-300, 1, 1e300 * (1 + 1j)])
def test_weight_scaling(block, scale):
    weights = scale * np.array([-0.5, 1])
    if block:
        weights = weights[:, None, None] * np.eye(1)
    tf = BarycentricRationalTransferFunction([0, 1], [1e100, 0.5e100], weights)
    for s in [0, 1, 1e-16, 2j]:
        np.testing.assert_allclose(tf.eval_tf(s) / 1e100, [[1 / (s + 1)]])
        np.testing.assert_allclose(tf.eval_dtf(s) / 1e100, [[-1 / (s + 1)**2]])


@pytest.mark.parametrize('block', [False, True])
def test_large_constant_values(block):
    values = np.full((1, 2, 1), 1e308)
    weights = np.array([[[1., 1.], [0., 1.]]]) if block else [1.]
    with np.errstate(over='raise', invalid='raise'):
        tf = BarycentricRationalTransferFunction([0.], values, weights)
        np.testing.assert_allclose(tf.eval_tf(1.), values[0])
        np.testing.assert_allclose(tf.eval_dtf(1.), np.zeros((2, 1)))


@pytest.mark.parametrize(('values', 'weights'), [([1., 2.], [1e-300, 1e300]), ([1e-300, 1e300], [1., 1.])])
def test_unrepresentable_relative_ranges(values, weights):
    with pytest.raises(ValueError, match='relative range'):
        BarycentricRationalTransferFunction([0., 1.], values, weights)


@pytest.mark.parametrize('block', [False, True])
def test_copy_with_and_pickle(block):
    nodes = np.array([0., 1.])
    values = np.array([2., 1.])
    weights = np.array([-1., 2.])
    if block:
        weights = weights[:, None, None]
    tf = BarycentricRationalTransferFunction(nodes, values, weights, side='right', sampling_time=0.1, name='example')
    for original, stored in zip((nodes, values, weights), (tf.nodes, tf.values, tf.weights), strict=True):
        original[:] = 10
        assert not stored.flags.writeable
    np.testing.assert_allclose(tf.eval_tf(1), [[1.]])
    np.testing.assert_allclose(tf.with_(values=np.array([4., 2.])).eval_tf(1), [[2.]])
    assert copy.copy(tf) is tf
    assert copy.deepcopy(tf) is tf
    restored = pickle.loads(pickle.dumps(tf))
    assert restored.sampling_time == 0.1
    assert restored.name == 'example'
    assert restored.side == 'right'
    for array in (restored.nodes, restored.values, restored.weights):
        assert not array.flags.writeable
    np.testing.assert_allclose(restored.eval_tf(0.5), [[4 / 3]])
    np.testing.assert_allclose(restored.eval_dtf(0.5), [[-8 / 9]])


def test_rectangular_right_pickle_and_with(rng):
    values = rng.normal(size=(3, 2, 3)) + 1j * rng.normal(size=(3, 2, 3))
    weights = np.eye(3) + 0.1 * rng.normal(size=(3, 3, 3))
    tf = BarycentricRationalTransferFunction([0, 1, 2], values, weights, side='right')
    restored = pickle.loads(pickle.dumps(tf))
    changed = restored.with_(values=2 * restored.values)
    for s in [0, 1 + 1e-15j, 0.3j]:
        np.testing.assert_allclose(restored.eval_tf(s), tf.eval_tf(s))
        np.testing.assert_allclose(restored.eval_dtf(s), tf.eval_dtf(s))
        np.testing.assert_allclose(changed.eval_tf(s), 2 * tf.eval_tf(s))
        np.testing.assert_allclose(changed.eval_dtf(s), 2 * tf.eval_dtf(s))
    assert not restored.weights.flags.writeable
    assert not changed.values.flags.writeable


def test_complex_data_with_subnormal_common_weight_scale():
    tf = BarycentricRationalTransferFunction([0, 1], [1 + 2j, 0.5 + 1j], 1e-320 * np.array([-1, 2]))
    np.testing.assert_allclose(tf.eval_tf(1j), [[(1 + 2j) / (1 + 1j)]])
    np.testing.assert_allclose(tf.eval_dtf(1j), [[-(1 + 2j) / (1 + 1j)**2]])


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('block', [False, True])
def test_infinity_and_constant(side, block, rng):
    values = rng.normal(size=(2, 2, 3))
    weights = np.array([-1., 2.])
    size = 2 if side == 'left' else 3
    if block:
        weights = weights[:, None, None] * np.eye(size)
    tf = BarycentricRationalTransferFunction([0, 1], values, weights, side=side)
    for s in [np.inf, -np.inf, complex(0, np.inf)]:
        np.testing.assert_allclose(tf.eval_tf(s), -values[0] + 2 * values[1])
        np.testing.assert_allclose(tf.eval_dtf(s), np.zeros((2, 3)))
    constant = BarycentricRationalTransferFunction([2], values[:1], weights[:1], side=side)
    for s in [2, 2 + 1e-16j, 3j, np.inf]:
        np.testing.assert_allclose(constant.eval_tf(s), values[0])
        np.testing.assert_allclose(constant.eval_dtf(s), np.zeros((2, 3)))


@pytest.mark.parametrize('block', [False, True])
def test_singular_denominators(block):
    weights = np.array([-1., 2.])
    if block:
        weights = weights[:, None, None]
    tf = BarycentricRationalTransferFunction([0, 1], [1, 0.5], weights)
    with pytest.raises(spla.LinAlgError):
        tf.eval_tf(-1)
    with pytest.raises(spla.LinAlgError):
        tf.eval_dtf(-1)
    weights = np.array([1., -1.])
    if block:
        weights = weights[:, None, None]
    improper = BarycentricRationalTransferFunction([0, 1], [0, 1], weights)
    with pytest.raises(ValueError, match='infinity'):
        improper.eval_tf(np.inf)
    with pytest.raises(ValueError, match='infinity'):
        improper.eval_dtf(np.inf)


@pytest.mark.parametrize(('nodes', 'values', 'weights'), [
    ([], [], []),
    ([[0]], [1], [1]),
    ([0, 0], [1, 2], [1, 1]),
    ([0], [[1]], [1]),
    ([0], [1, 2], [1]),
    ([0], np.empty((1, 0, 2)), [1]),
    ([0], [1], [0]),
    ([0], [1], [1, 2]),
    ([0], [1], [[1]]),
    ([0], np.ones((1, 2, 3)), np.ones((1, 3, 3))),
    ([np.inf], [1], [1]),
    ([0], [np.nan], [1]),
    ([0], [1], [np.inf]),
])
def test_validation(nodes, values, weights):
    with pytest.raises(AssertionError):
        BarycentricRationalTransferFunction(nodes, values, weights)


@pytest.mark.parametrize(('values', 'weights'), [
    ([1], np.zeros((1, 1, 1))),
    (np.ones((1, 2, 3)), np.ones((1, 2, 2))),
])
def test_singular_matrix_weight(values, weights):
    with pytest.raises(ValueError, match='nonsingular'):
        BarycentricRationalTransferFunction([0], values, weights)


@pytest.mark.parametrize('sampling_time', [-1, np.inf, np.nan])
def test_invalid_sampling_time(sampling_time):
    with pytest.raises(AssertionError, match='sampling_time'):
        BarycentricRationalTransferFunction([0], [1], [1], sampling_time=sampling_time)


def test_invalid_side():
    with pytest.raises(AssertionError, match='side'):
        BarycentricRationalTransferFunction([0], [1], [1], side='invalid')


@pytest.mark.parametrize('s', [np.nan, [1., 2.]])
def test_invalid_evaluation_argument(s):
    tf = BarycentricRationalTransferFunction([0], [1], [1])
    with pytest.raises(AssertionError, match='s must'):
        tf.eval_tf(s)
    with pytest.raises(AssertionError, match='s must'):
        tf.eval_dtf(s)


@pytest.mark.parametrize('scale', [1e-160, 1., 1e200])
@pytest.mark.parametrize('kind', ['scalar', 'left', 'right'])
def test_extreme_frequency_scales_and_nearest_indices(scale, kind):
    residue = np.arange(1., 7.).reshape(2, 3) + 0.2j
    values = np.stack([residue, residue / 2])
    weights = np.array([-1., 2.])
    side = 'left' if kind == 'scalar' else kind
    if kind != 'scalar':
        size = values.shape[1 if side == 'left' else 2]
        matrix = np.eye(size, dtype=complex)
        matrix[0, -1] = 0.2j
        weights = weights[:, None, None] * matrix
    tf = BarycentricRationalTransferFunction(scale * np.array([0., 1.]), values, weights, side=side)
    points = [0., 1., np.nextafter(1., 2.), 1e-320j, 1 + 1e-320j, 0.3j]
    with np.errstate(over='raise', invalid='raise'):
        for s in points:
            np.testing.assert_allclose(tf.tf(scale * s), residue / (s + 1), atol=1e-13)
            np.testing.assert_allclose(scale * tf.dtf(scale * s), -residue / (s + 1)**2, atol=1e-13)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('block', [False, True])
@pytest.mark.parametrize('layout', ['C', 'F', 'negative'])
def test_contraction_layout_and_state(side, block, layout, rng):
    nodes = np.array([-2., -1., 0., 1.])
    values = rng.normal(size=(4, 2, 3)) + 1j * rng.normal(size=(4, 2, 3))
    size = 2 if side == 'left' else 3
    weights = (np.eye(size) + 0.1 * rng.normal(size=(4, size, size))) if block else np.arange(1., 5.)
    reference = BarycentricRationalTransferFunction(nodes, values, weights, side=side)

    def arrange(array):
        return array[::-1].copy()[::-1] if layout == 'negative' else np.array(array, order=layout)

    tf = BarycentricRationalTransferFunction(*(arrange(a) for a in (nodes, values, weights)), side=side)
    stored = {key: array.copy() for key, array in tf.__dict__.items() if isinstance(array, np.ndarray)}
    restored = pickle.loads(pickle.dumps(tf))
    changed = tf.with_(values=2 * values)
    for s in [*nodes, 0.3j, 1 + 1e-15j, np.array(0.2)]:
        for derivative in (False, True):
            expected = reference._eval(s, derivative)
            for model in (tf, restored):
                np.testing.assert_allclose(model._eval(s, derivative), expected, atol=1e-12)
            np.testing.assert_allclose(changed._eval(s, derivative), 2 * expected, atol=1e-12)
    for key, array in stored.items():
        np.testing.assert_array_equal(getattr(tf, key), array)
        assert not getattr(tf, key).flags.writeable
        assert not getattr(restored, key).flags.writeable


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('s', [0., 0.2, 0.3j])
def test_derivative_factorization_reuse(side, s, monkeypatch):
    size = 2 if side == 'left' else 3
    tf = BarycentricRationalTransferFunction(
        [0., 1., 2.], np.arange(18.).reshape(3, 2, 3), np.tile(np.eye(size), (3, 1, 1)), side=side
    )
    calls = [0, 0]
    factorize, solve = tf._factorize, tf._solve

    def counted_factorize(*args, **kwargs):
        calls[0] += 1
        return factorize(*args, **kwargs)

    def counted_solve(*args, **kwargs):
        calls[1] += 1
        return solve(*args, **kwargs)

    monkeypatch.setattr(BarycentricRationalTransferFunction, '_factorize', staticmethod(counted_factorize))
    monkeypatch.setattr(BarycentricRationalTransferFunction, '_solve', staticmethod(counted_solve))
    tf._eval(s, derivative=True)
    assert calls == [1, 1 if s == 0 else 2]


@pytest.mark.parametrize('small', [1e-170, 1e-200])
@pytest.mark.parametrize('block', [False, True])
@pytest.mark.parametrize('phase', [1., 1. + 1.j])
def test_scalar_numerator_cache_preserves_underflowed_terms(small, block, phase):
    large = 1e150 if small == 1e-170 else 1e200
    weights = phase * np.array([1., small, 1.])
    if block:
        weights = weights[:, None, None]
    tf = BarycentricRationalTransferFunction([0., small, large], [0., 1., 2 * large], weights)
    assert tf._flat_numerator is None
    np.testing.assert_allclose(tf.dtf(0.), [[-3.]], rtol=1e-14)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('block', [False, True])
@pytest.mark.parametrize('entry', [0., complex(1.3e308, 1.3e308)])
def test_single_support_extreme_constant(side, block, entry):
    values = np.full((1, 2, 3), entry)
    size = 2 if side == 'left' else 3
    weights = np.eye(size)[None] if block else [1.]
    with np.errstate(over='raise', invalid='raise'):
        tf = BarycentricRationalTransferFunction([0.], values, weights, side=side)
        for s in [0., 0.3j, np.inf, complex(np.inf, 1.)]:
            np.testing.assert_array_equal(tf.tf(s), values[0])
            np.testing.assert_array_equal(tf.dtf(s), np.zeros((2, 3)))


@pytest.mark.parametrize('block', [False, True])
def test_constant_samples_do_not_cancel_singular_denominator(block):
    values = np.ones((2, 2, 3))
    weights = np.array([-1., 2.])
    if block:
        weights = weights[:, None, None] * np.eye(2)
    tf = BarycentricRationalTransferFunction([0., 1.], values, weights)
    for derivative in (False, True):
        with pytest.raises(spla.LinAlgError):
            tf._eval(-1., derivative)


@pytest.mark.parametrize('complex_data', [False, True])
def test_lapack_error_status(complex_data, monkeypatch):
    dtype = complex if complex_data else float
    matrix = np.eye(2, dtype=dtype)
    factors = BarycentricRationalTransferFunction._factorize(matrix)
    prefix = 'z' if complex_data else 'd'
    monkeypatch.setattr(spla.lapack, prefix + 'getrf', lambda *args, **kwargs: (matrix, np.array([0, 1]), -1))
    with pytest.raises(ValueError, match=r'argument 1.*getrf'):
        BarycentricRationalTransferFunction._factorize(matrix)
    monkeypatch.setattr(spla.lapack, prefix + 'getrs', lambda *args, **kwargs: (matrix, -3))
    with pytest.raises(ValueError, match=r'argument 3.*getrs'):
        BarycentricRationalTransferFunction._solve(factors, matrix)
