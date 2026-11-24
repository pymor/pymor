# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.linalg as spla

from pymor.algorithms import barycentric
from pymor.algorithms.barycentric import evaluate_bary_a, evaluate_bary_b

pytestmark = pytest.mark.builtin


@pytest.mark.parametrize('matrix', [False, True])
@pytest.mark.parametrize(('cached', 'scale'), [(False, 1.), (True, 7.), (True, 1e-300), (False, 1e300)])
def test_evaluation(matrix, cached, scale):
    nodes = np.array([0., 1.])
    residue = np.array([[1., 2., 3.], [4., 5., 6.]])
    values = residue[None] / (nodes[:, None, None] + 1)
    weights = np.array([-1., 2.])
    if matrix:
        weights = weights[:, None, None] * np.array([[1., 0.2], [0., 1.]])
        evaluate = evaluate_bary_b
        weighted_values = weights @ barycentric._scale_values(values, scale) if cached else None
    else:
        values = values.reshape(2, -1)
        evaluate = evaluate_bary_a
        weighted_values = weights[:, None] * barycentric._scale_values(values, scale) if cached else None
    arrays = [nodes, values, weights] + ([] if weighted_values is None else [weighted_values])
    originals = [array.copy() for array in arrays]
    for array in arrays:
        array.flags.writeable = False
    for derivative in (False, True):
        for s in (0., 1., 1e-15j, 1 + 1e-15j, 0.3j, np.inf):
            expected = np.zeros_like(residue) if not np.isfinite(s) else (
                -residue / (s + 1)**2 if derivative else residue / (s + 1)
            )
            actual = evaluate(s, nodes, values, weights, weighted_values=weighted_values,
                              value_scale=scale, derivative=derivative)
            np.testing.assert_allclose(actual.reshape(2, 3), expected, atol=1e-13)
            if not derivative and s in nodes:
                np.testing.assert_array_equal(actual, values[np.flatnonzero(s == nodes)[0]])
    for array, original in zip(arrays, originals, strict=True):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize('s', [0., 0.3j])
def test_derivative_factorization_reuse(s, monkeypatch):
    nodes = np.array([0., 1., 2.])
    values = np.arange(18.).reshape(3, 2, 3)
    weights = np.tile(np.eye(2), (3, 1, 1))
    calls = [0, 0]
    factorize, solve = barycentric._factorize, barycentric._solve

    def counted_factorize(*args, **kwargs):
        calls[0] += 1
        return factorize(*args, **kwargs)

    def counted_solve(*args, **kwargs):
        calls[1] += 1
        return solve(*args, **kwargs)

    monkeypatch.setattr(barycentric, '_factorize', counted_factorize)
    monkeypatch.setattr(barycentric, '_solve', counted_solve)
    evaluate_bary_b(s, nodes, values, weights, derivative=True)
    assert calls == [1, 1 if s == 0 else 2]


def test_late_multiplication_preserves_underflowed_terms():
    small, large = 1e-200, 1e200
    weights = (1 + 1j) * np.array([1., small, 1.])
    nodes = np.array([0., small, large])
    values = np.array([0., 1., 2 * large])
    weights, _ = barycentric._normalize(weights, 'weights')
    _, scale = barycentric._normalize(values, 'values')
    result = evaluate_bary_a(0., nodes, values, weights, value_scale=scale, derivative=True)
    np.testing.assert_allclose(result, -3., rtol=1e-14)


@pytest.mark.parametrize('evaluate', [evaluate_bary_a, evaluate_bary_b])
@pytest.mark.parametrize('s', [np.nan, [1., 2.]])
def test_invalid_point(evaluate, s):
    with pytest.raises(AssertionError, match='s must'):
        evaluate(s, np.array([0.]), np.array([1.]), np.array([1.]))


@pytest.mark.parametrize('complex_data', [False, True])
def test_lapack_error_status(complex_data, monkeypatch):
    dtype = complex if complex_data else float
    matrix = np.eye(2, dtype=dtype)
    factors = barycentric._factorize(matrix)
    prefix = 'z' if complex_data else 'd'
    monkeypatch.setattr(spla.lapack, prefix + 'getrf', lambda *args, **kwargs: (matrix, np.array([0, 1]), -1))
    with pytest.raises(ValueError, match=r'argument 1.*getrf'):
        barycentric._factorize(matrix)
    monkeypatch.setattr(spla.lapack, prefix + 'getrs', lambda *args, **kwargs: (matrix, -3))
    with pytest.raises(ValueError, match=r'argument 3.*getrs'):
        barycentric._solve(factors, matrix)
