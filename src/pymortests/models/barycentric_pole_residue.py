# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.linalg as spla

from pymor.models.barycentric import BarycentricRationalTransferFunction
from pymor.models.pole_residue import PoleResidueTransferFunction
from pymor.models.transfer_function import TransferFunction
from pymortests.models.iosys_add_sub_mul import assert_tf_add, assert_tf_mul, assert_tf_sub

pytestmark = pytest.mark.builtin


@pytest.mark.parametrize(('side', 'matrix_weights', 'shape', 'count', 'sampling_time'), [
    ('left', False, (1, 1), 1, 0.),
    ('left', False, (2, 3), 3, 0.1),
    ('right', False, (3, 2), 3, 0.),
    ('left', True, (1, 3), 3, 0.),
    ('right', True, (3, 1), 1, 0.1),
    ('left', True, (3, 2), 3, 0.1),
    ('right', True, (2, 3), 3, 0.),
])
def test_numerator_and_denominator(side, matrix_weights, shape, count, sampling_time, rng):
    nodes = np.array([-1 + 0.2j, 0.4 - 1j, 2 + 0.3j])[:count]
    values = rng.normal(size=(count, *shape)) + 1j * rng.normal(size=(count, *shape))
    weights = 8j * (1 + rng.random(count))
    if matrix_weights:
        size = shape[0 if side == 'left' else 1]
        weights = weights[:, None, None] * (np.eye(size) + 0.1 * (
            rng.normal(size=(count, size, size)) + 1j * rng.normal(size=(count, size, size))))
        residues = weights @ values if side == 'left' else values @ weights
        denominator_residues = weights
    else:
        residues = weights[:, None, None] * values
        denominator_residues = weights[:, None, None]
    tf = BarycentricRationalTransferFunction(nodes, values, weights, side=side,
                                           sampling_time=sampling_time, name='example')
    numerator, denominator = tf.numerator, tf.denominator
    for factor, label in [(numerator, 'numerator'), (denominator, 'denominator')]:
        assert type(factor) is PoleResidueTransferFunction
        assert factor.sampling_time == sampling_time
        assert factor.name == f'example_{label}'
        assert factor.D is None
        assert np.array_equal(np.unique(factor.poles), np.unique(nodes))
    assert (numerator.dim_output, numerator.dim_input) == shape
    assert denominator.dim_input == denominator.dim_output == denominator_residues.shape[1]
    for s in (0.3j, 0.8 + 0.6j, -0.7):
        delta = (s - nodes)[:, None, None]
        N, D = numerator.eval_tf(s), denominator.eval_tf(s)
        assert np.allclose(N, np.sum(residues / delta, axis=0))
        assert np.allclose(D, np.sum(denominator_residues / delta, axis=0))
        assert np.allclose(numerator.eval_dtf(s), -np.sum(residues / delta**2, axis=0))
        assert np.allclose(denominator.eval_dtf(s), -np.sum(denominator_residues / delta**2, axis=0))
        if not matrix_weights:
            H = N / D[0, 0]
        elif side == 'left':
            H = spla.solve(D, N)
        else:
            H = spla.solve(D.T, N.T).T
        assert np.allclose(H, tf.eval_tf(s))


@pytest.mark.parametrize(('side', 'matrix_weights'), [('left', False), ('left', True), ('right', True)])
def test_factoring_without_residue_product_overflow(side, matrix_weights):
    values = np.full((2, 2, 3), 1e200)
    weights = np.array([1e200, 2e200])
    if matrix_weights:
        weights = weights[:, None, None] * np.eye(2 if side == 'left' else 3)
    tf = BarycentricRationalTransferFunction([0, 1], values, weights, side=side)
    with np.errstate(over='raise', invalid='raise'):
        for factor in (tf.numerator, tf.denominator):
            assert np.all(np.isfinite(factor.b))
            assert np.all(np.isfinite(factor.c))


def test_zero_numerator():
    tf = BarycentricRationalTransferFunction([0, 1], [0, 0], [-1, 2])
    assert np.array_equal(tf.numerator.eval_tf(0.3j), np.zeros((1, 1)))
    assert np.allclose(tf.denominator.eval_tf(0.3j), [[-1 / 0.3j + 2 / (0.3j - 1)]])


@pytest.mark.parametrize('sampling_time', [0, 0.1])
@pytest.mark.parametrize('barycentric_first', [False, True])
def test_mixed_arithmetic(sampling_time, barycentric_first):
    values = np.stack([np.ones((2, 2)) / 3, np.ones((2, 2)) / 4])
    bary = BarycentricRationalTransferFunction([0, 1], values, [-3, 4], sampling_time=sampling_time)
    pole = PoleResidueTransferFunction([-1, -2], [[1, 2], [2, 1]], [[2, 1], [1, 3]],
                                      D=np.eye(2), sampling_time=sampling_time)
    a, b = (bary, pole) if barycentric_first else (pole, bary)
    for result, check in ((a + b, assert_tf_add), (a - b, assert_tf_sub), (a * b, assert_tf_mul)):
        assert type(result) is TransferFunction
        assert result.sampling_time == sampling_time
        check(a, b, result, False)


@pytest.mark.parametrize('barycentric_first', [False, True])
def test_rectangular_mixed_composition(barycentric_first):
    values = np.stack([np.ones((3, 2)) / 3, np.ones((3, 2)) / 4])
    bary = BarycentricRationalTransferFunction([0, 1], values, [-3, 4])
    p, m = (2, 4) if barycentric_first else (4, 3)
    pole = PoleResidueTransferFunction([-2], np.ones((1, m)), np.ones((1, p)))
    a, b = (bary, pole) if barycentric_first else (pole, bary)
    result = a * b
    assert (result.dim_output, result.dim_input) == (a.dim_output, b.dim_input)
    assert_tf_mul(a, b, result, False)


def test_mixed_h2_inner_product():
    values = np.stack([np.ones((2, 2)) / 3, np.ones((2, 2)) / 4])
    bary = BarycentricRationalTransferFunction([0, 1], values, [-3, 4])
    pole = PoleResidueTransferFunction([-2], [[1, 2]], [[2, 1]])
    assert np.isclose(bary.h2_inner(pole), 9 / 5)
    assert np.isclose(pole.h2_inner(bary), 9 / 5)
