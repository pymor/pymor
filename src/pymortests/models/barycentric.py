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


def test_scalar_evaluation():
    tf = BarycentricRationalTransferFunction([0, 1], [1, 0.5], [-1, 2])
    for s in [0, 1, 1e-15, 1 + 1e-15j, 2j]:
        np.testing.assert_allclose(tf.eval_tf(s), [[1 / (s + 1)]], atol=1e-14)
        np.testing.assert_allclose(tf.eval_dtf(s), [[-1 / (s + 1)**2]], atol=1e-14)


def test_matrix_samples_with_scalar_weights(rng):
    residue = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
    feedthrough = rng.normal(size=(2, 3))
    values = np.stack([feedthrough + residue, feedthrough + residue / 2])
    tf = BarycentricRationalTransferFunction([0, 1], values, [-1, 2])
    assert (tf.dim_output, tf.dim_input) == (2, 3)
    for s in [0, 1, 1e-15j, 2j]:
        assert np.allclose(tf.eval_tf(s), feedthrough + residue / (s + 1))
        assert np.allclose(tf.eval_dtf(s), -residue / (s + 1)**2)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_matrix_weights(side, rng):
    nodes = np.array([-1., 0.5, 2.])
    values = rng.normal(size=(3, 2, 3)) + 1j * rng.normal(size=(3, 2, 3))
    size = 2 if side == 'left' else 3
    weights = np.eye(size) + 0.2 * (rng.normal(size=(3, size, size)) + 1j * rng.normal(size=(3, size, size)))
    tf = BarycentricRationalTransferFunction(nodes, values, weights, side=side)
    coefficients = weights @ values if side == 'left' else values @ weights
    for s in [*nodes, -1 + 1e-15, 0.5 + 1e-15j, 0.3j]:
        # multiply by the common nodal polynomial for a reference without singular support terms.
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
        np.testing.assert_array_equal(tf.eval_tf(node), value)


@pytest.mark.parametrize(('matrix_weights', 'side', 'shape', 'expected'), [
    (False, 'left', (1, 1), 'bary-A'),
    (False, 'right', (2, 3), 'bary-A'),
    (True, 'right', (1, 1), 'bary-A'),
    (True, 'left', (1, 3), 'bary-A'),
    (True, 'right', (3, 1), 'bary-A'),
    (True, 'left', (2, 3), 'bary-B'),
    (True, 'right', (2, 3), 'bary-B'),
])
def test_form(matrix_weights, side, shape, expected):
    values = np.arange(2 * np.prod(shape), dtype=float).reshape(2, *shape) + 1
    weights = np.array([-1 + 0.2j, 2 - 0.1j])
    if matrix_weights:
        weights = weights[:, None, None] * np.eye(shape[0 if side == 'left' else 1])
    tf = BarycentricRationalTransferFunction([0, 1], values, weights, side=side)
    assert tf.form == expected
    assert tf.side == side
    np.testing.assert_array_equal(tf.weights, weights)
    scalar = tf.with_(weights=np.array([-1 + 0.2j, 2 - 0.1j]))
    assert scalar.form == 'bary-A'
    for s in [0, 1 + 1e-15j, 0.3j]:
        assert np.allclose(scalar.eval_tf(s), tf.eval_tf(s))
        assert np.allclose(scalar.eval_dtf(s), tf.eval_dtf(s))


@pytest.mark.parametrize('barycentric_first', [False, True])
@pytest.mark.parametrize('name', name_list)
def test_mixed_arithmetic(name, barycentric_first):
    tf = BarycentricRationalTransferFunction([0, 1], [2, 1], [-1, 2], sampling_time=0.1)
    other = get_model(name, 0.1, True)
    m1, m2 = (tf, other) if barycentric_first else (other, tf)
    for result, check in ((m1 + m2, assert_tf_add), (m1 - m2, assert_tf_sub), (m1 * m2, assert_tf_mul)):
        assert type(result) is TransferFunction
        assert result.sampling_time == 0.1
        check(m1, m2, result, True)


@pytest.mark.parametrize(('form', 'side', 'shape'), [
    ('bary-A', 'left', (1, 1)),
    ('bary-A', 'right', (2, 3)),
    ('bary-B', 'left', (2, 3)),
    ('bary-B', 'right', (3, 2)),
])
def test_add_sub_neg(form, side, shape, rng):
    values = rng.normal(size=(3, *shape)) + 1j * rng.normal(size=(3, *shape))
    other_values = rng.normal(size=values.shape) + 1j * rng.normal(size=values.shape)
    weights = np.array([1., 2., 3.])
    if form == 'bary-B':
        size = shape[0 if side == 'left' else 1]
        weights = weights[:, None, None] * (np.eye(size) + 0.1j * rng.normal(size=(3, size, size)))
    a = BarycentricRationalTransferFunction([0, 1, 2], values, weights, side=side, sampling_time=0.1)
    # mix vector and 1x1-matrix storage to catch accidental broadcasting in SISO addition.
    if shape == (1, 1):
        other_values = other_values[:, 0, 0]
    b = BarycentricRationalTransferFunction(a.nodes, other_values, weights, side=side, sampling_time=0.1)
    for result, left, right in ((a + b, 1, 1), (b + a, 1, 1), (a - b, 1, -1), (b - a, -1, 1), (-a, -1, 0)):
        assert type(result) is BarycentricRationalTransferFunction
        assert result.form == form
        assert result.side == side
        assert result.sampling_time == 0.1
        np.testing.assert_array_equal(result.nodes, a.nodes)
        np.testing.assert_array_equal(result.weights, a.weights)
        for s in [0, 1e-15j, 0.3j]:
            np.testing.assert_allclose(result.eval_tf(s), left * a.eval_tf(s) + right * b.eval_tf(s), atol=1e-13)
            np.testing.assert_allclose(result.eval_dtf(s), left * a.eval_dtf(s) + right * b.eval_dtf(s), atol=1e-13)


@pytest.mark.parametrize(('change', 'value'), [('nodes', [0, 2]), ('weights', [-1, 3]), ('side', 'right')])
def test_arithmetic_fallback(change, value):
    a = BarycentricRationalTransferFunction([0, 1], [2, 1], [-1, 2])
    b = a.with_(**{change: value})
    for result, check in ((a + b, assert_tf_add), (a - b, assert_tf_sub), (a * b, assert_tf_mul)):
        assert type(result) is TransferFunction
        check(a, b, result, False)


@pytest.mark.parametrize('kwargs', [
    {'sampling_time': 0.1}, {'values': np.ones((2, 2, 1))}, {'values': np.ones((2, 1, 3))},
])
def test_incompatible_arithmetic(kwargs):
    a = BarycentricRationalTransferFunction([0, 1], np.ones((2, 2, 3)), [-1, 2])
    b = a.with_(**kwargs)
    with pytest.raises(AssertionError):
        a + b
    with pytest.raises(AssertionError):
        a - b


@pytest.mark.parametrize('barycentric_first', [False, True])
def test_rectangular_composition(barycentric_first):
    tf = BarycentricRationalTransferFunction([0, 1], [np.ones((3, 2)), 0.5 * np.ones((3, 2))], [-1, 2])
    m, p = (4, 2) if barycentric_first else (3, 4)
    other = LTIModel.from_matrices(np.array([[-2.]]), np.ones((1, m)), np.ones((p, 1)))
    m1, m2 = (tf, other) if barycentric_first else (other, tf)
    result = m1 * m2
    assert (result.dim_output, result.dim_input) == (m1.dim_output, m2.dim_input)
    assert_tf_mul(m1, m2, result, False)


@pytest.mark.parametrize(('form', 'side', 'shape', 'layout'), [
    ('bary-A', 'left', (1, 1), 'C'),
    ('bary-A', 'right', (2, 3), 'F'),
    ('bary-B', 'left', (2, 3), 'F'),
    ('bary-B', 'right', (2, 3), 'negative'),
])
def test_copy_with_and_pickle(form, side, shape, layout, rng):
    residue = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    nodes = np.array([0., 1.])
    values = np.stack([residue, residue / 2])
    if shape == (1, 1):
        values = values[:, 0, 0]
    weights = np.array([-1., 2.])
    if form == 'bary-B':
        size = shape[0 if side == 'left' else 1]
        weights = weights[:, None, None] * (np.eye(size) + 0.1j * rng.normal(size=(size, size)))

    def arrange(array):
        return array[::-1].copy()[::-1] if layout == 'negative' else np.array(array, order=layout)

    nodes, values, weights = map(arrange, (nodes, values, weights))
    tf = BarycentricRationalTransferFunction(nodes, values, weights, side=side, sampling_time=0.1, name='example')
    for original, stored in zip((nodes, values, weights), (tf.nodes, tf.values, tf.weights), strict=True):
        saved = stored.copy()
        original[:] = 10
        np.testing.assert_array_equal(stored, saved)
    assert copy.copy(tf) is tf
    assert copy.deepcopy(tf) is tf
    restored = pickle.loads(pickle.dumps(tf))
    changed = restored.with_(values=2 * restored.values)
    for model, factor in [(tf, 1), (restored, 1), (changed, 2)]:
        assert model.form == form
        assert model.side == side
        assert model.sampling_time == 0.1
        assert model.name == 'example'
        for array in (model.nodes, model.values, model.weights):
            assert not array.flags.writeable
        for s in [0, 1 + 1e-15j, 0.3j, np.array(0.2)]:
            np.testing.assert_allclose(model.eval_tf(s), factor * residue / (s + 1), atol=1e-13)
            np.testing.assert_allclose(model.eval_dtf(s), -factor * residue / (s + 1)**2, atol=1e-13)


@pytest.mark.parametrize('dtype', [np.int8, np.uint64])
def test_integer_data(dtype):
    weight = np.iinfo(dtype).max // 2
    tf = BarycentricRationalTransferFunction(
        np.array([0, 1], dtype=dtype), np.array([2, 3], dtype=dtype), np.array([weight, weight], dtype=dtype)
    )
    assert np.allclose(tf.eval_tf(2), [[8 / 3]])
    assert np.allclose(tf.eval_dtf(2), [[-1 / 9]])


@pytest.mark.parametrize('form', ['bary-A', 'bary-B'])
@pytest.mark.parametrize('scale', [1e-300, 1e300 * (1 + 1j)])
def test_normalization(form, scale):
    residue = np.arange(1., 7.).reshape(2, 3)
    values = 1e100 * np.stack([residue, residue / 2])
    weights = scale * np.array([-0.5, 1])
    if form == 'bary-B':
        weights = weights[:, None, None] * np.array([[1., 0.2j], [0., 1.]])
    tf = BarycentricRationalTransferFunction([0, 1], values, weights)
    for s in [0, 1, 1e-15j, 2j]:
        assert np.allclose(tf.eval_tf(s) / 1e100, residue / (s + 1))
        assert np.allclose(tf.eval_dtf(s) / 1e100, -residue / (s + 1)**2)


def test_subnormal_weight_scale():
    tf = BarycentricRationalTransferFunction([0, 1], [1 + 2j, 0.5 + 1j], 1e-320 * np.array([-1, 2]))
    assert np.allclose(tf.eval_tf(1j), [[(1 + 2j) / (1 + 1j)]])
    assert np.allclose(tf.eval_dtf(1j), [[-(1 + 2j) / (1 + 1j)**2]])


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('matrix_weights', [False, True])
def test_infinity_and_constant(side, matrix_weights, rng):
    values = rng.normal(size=(2, 2, 3))
    weights = np.array([-1., 2.])
    if matrix_weights:
        weights = weights[:, None, None] * np.eye(2 if side == 'left' else 3)
    tf = BarycentricRationalTransferFunction([0, 1], values, weights, side=side)
    for s in [np.inf, -np.inf, complex(0, np.inf)]:
        assert np.allclose(tf.eval_tf(s), -values[0] + 2 * values[1])
        assert np.allclose(tf.eval_dtf(s), np.zeros((2, 3)))
    constant = BarycentricRationalTransferFunction([2], values[:1], weights[:1], side=side)
    for s in [2, 2 + 1e-16j, 3j, np.inf]:
        assert np.allclose(constant.eval_tf(s), values[0])
        assert np.allclose(constant.eval_dtf(s), np.zeros((2, 3)))


@pytest.mark.parametrize('matrix_weights', [False, True])
@pytest.mark.parametrize('constant', [False, True])
def test_singular_denominators(matrix_weights, constant):
    values = np.ones((2, 2, 3))
    if not constant:
        values[1] /= 2
    weights = np.array([-1., 2.])
    if matrix_weights:
        weights = weights[:, None, None] * np.eye(2)
    tf = BarycentricRationalTransferFunction([0, 1], values, weights)
    improper = tf.with_(weights=np.array([1., -1.])[:, None, None] * np.eye(2) if matrix_weights else [1., -1.])
    with pytest.raises(spla.LinAlgError):
        tf.eval_tf(-1)
    with pytest.raises(spla.LinAlgError):
        tf.eval_dtf(-1)
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


@pytest.mark.parametrize('kwargs', [{'sampling_time': -1}, {'sampling_time': np.inf},
                                  {'sampling_time': np.nan}, {'side': 'invalid'}])
def test_invalid_options(kwargs):
    with pytest.raises(AssertionError):
        BarycentricRationalTransferFunction([0], [1], [1], **kwargs)


@pytest.mark.parametrize(('values', 'weights'), [
    ([1], np.zeros((1, 1, 1))),
    (np.ones((1, 2, 3)), np.ones((1, 2, 2))),
])
def test_singular_matrix_weight(values, weights):
    with pytest.raises(ValueError, match='nonsingular'):
        BarycentricRationalTransferFunction([0], values, weights)


@pytest.mark.parametrize(('values', 'weights'), [([1., 2.], [1e-300, 1e300]), ([1e-300, 1e300], [1., 1.])])
def test_unrepresentable_relative_ranges(values, weights):
    with pytest.raises(ValueError, match='relative range'):
        BarycentricRationalTransferFunction([0, 1], values, weights)


@pytest.mark.parametrize('s', [np.nan, [1., 2.]])
def test_invalid_evaluation_argument(s):
    tf = BarycentricRationalTransferFunction([0], [1], [1])
    with pytest.raises(AssertionError, match='s must'):
        tf.eval_tf(s)
    with pytest.raises(AssertionError, match='s must'):
        tf.eval_dtf(s)


@pytest.mark.parametrize('scale', [1e-160, 1e200])
@pytest.mark.parametrize('kind', ['scalar', 'left', 'right'])
def test_extreme_frequency_scales(scale, kind):
    residue = np.arange(1., 7.).reshape(2, 3) + 0.2j
    values = np.stack([residue, residue / 2])
    weights = np.array([-1., 2.])
    side = 'left' if kind == 'scalar' else kind
    if kind != 'scalar':
        matrix = np.eye(2 if side == 'left' else 3, dtype=complex)
        matrix[0, -1] = 0.2j
        weights = weights[:, None, None] * matrix
    tf = BarycentricRationalTransferFunction(scale * np.array([0., 1.]), values, weights, side=side)
    with np.errstate(over='raise', invalid='raise'):
        for s in [0., 1., np.nextafter(1., 2.), 1e-320j, 1 + 1e-320j, 0.3j]:
            np.testing.assert_allclose(tf.eval_tf(scale * s), residue / (s + 1), atol=1e-13)
            np.testing.assert_allclose(scale * tf.eval_dtf(scale * s), -residue / (s + 1)**2, atol=1e-13)


@pytest.mark.parametrize(('small', 'matrix_weights', 'phase'), [(1e-170, False, 1.), (1e-200, True, 1. + 1.j)])
def test_numerator_cache_underflow_fallback(small, matrix_weights, phase):
    large = 1e150 if small == 1e-170 else 1e200
    weights = phase * np.array([1., small, 1.])
    if matrix_weights:
        weights = weights[:, None, None]
    tf = BarycentricRationalTransferFunction([0., small, large], [0., 1., 2 * large], weights)
    np.testing.assert_allclose(tf.eval_dtf(0.), [[-3.]], rtol=1e-14)


@pytest.mark.parametrize(('side', 'matrix_weights', 'entry'), [
    ('left', False, 0.),
    ('right', False, complex(1.3e308, 1.3e308)),
    ('left', True, complex(1.3e308, 1.3e308)),
    ('right', True, 1e308),
])
def test_single_support_extreme_constant(side, matrix_weights, entry):
    values = np.full((1, 2, 3), entry)
    weights = np.eye(2 if side == 'left' else 3)[None] if matrix_weights else [1.]
    with np.errstate(over='raise', invalid='raise'):
        tf = BarycentricRationalTransferFunction([0.], values, weights, side=side)
        for s in [0., 0.3j, np.inf, complex(np.inf, 1.)]:
            np.testing.assert_array_equal(tf.eval_tf(s), values[0])
            np.testing.assert_array_equal(tf.eval_dtf(s), np.zeros((2, 3)))
