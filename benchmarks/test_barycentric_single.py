# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Run explicitly alongside the single-point benchmark campaign."""

import numpy as np
import pytest

from benchmarks.barycentric_single import _cases, _error, _fixture, _measure, _reference, _run_case
from pymor.models.barycentric import BarycentricRationalTransferFunction


def test_fixture_coverage():
    cases = _cases()
    assert {(c['data'], c['query']) for c in cases} == {
        ('real', 'real'), ('real', 'complex'), ('complex', 'real'), ('complex', 'complex')
    }
    assert {c['form'] for c in cases} == {'scalar', 'left', 'right'}
    assert {c['layout'] for c in cases} == {'C', 'F', 'negative'}
    assert max(c['supports'] for c in cases) == 512
    assert max(max(c['shape']) for c in cases) == 32


@pytest.mark.parametrize('form', ['scalar', 'left', 'right'])
def test_independent_reference_identity(form):
    nodes = np.array([0., 1.])
    residue = np.arange(1., 7.).reshape(2, 3) + 0.2j
    values = np.stack([residue, residue / 2])
    weights = np.array([-1., 2.])
    if form != 'scalar':
        size = 2 if form == 'left' else 3
        matrix = np.eye(size, dtype=complex)
        matrix[0, -1] = 0.3j
        weights = weights[:, None, None] * matrix
    for s in [0., 1., 0.3j]:
        value, derivative = _reference(nodes, values, weights, form, s)
        np.testing.assert_allclose(value, residue / (s + 1))
        np.testing.assert_allclose(derivative, -residue / (s + 1)**2)


@pytest.mark.parametrize('form', ['scalar', 'left', 'right'])
def test_fixture_layout_and_real_near_grid(form):
    case = {'supports': 8, 'shape': (2, 3), 'form': form, 'data': 'real', 'query': 'real', 'layout': 'C'}
    arrays, groups = _fixture(case, 123, 8)
    assert not np.iscomplexobj(groups['near_real'])
    assert np.iscomplexobj(groups['near'])
    assert np.all(groups['near_real'] != groups['support'])
    for layout in ['F', 'negative']:
        changed, _ = _fixture({**case, 'layout': layout}, 123, 8)
        for actual, expected in zip(changed, arrays, strict=True):
            np.testing.assert_array_equal(actual, expected)
        if layout == 'F':
            assert changed[1].flags.f_contiguous
        else:
            assert changed[1].strides[0] < 0


def test_timing_order_alternates():
    calls = []
    functions = [lambda s: calls.append(0), lambda s: calls.append(1)]
    result = _measure(functions, np.array([0.]), 3)
    assert calls == [0, 1, 0, 1, 1, 0, 0, 1]
    assert len(result['before_samples_us']) == 3
    assert len(result['after_samples_us']) == 3


@pytest.mark.parametrize('actual', [np.nan, np.inf, 1.])
def test_error_rejects_nonfinite_and_disagreement(actual):
    with pytest.raises(AssertionError):
        _error(np.array([actual]), np.array([0.]))


def test_case_runner_self_comparison():
    case = {'supports': 2, 'shape': (1, 1), 'form': 'scalar',
            'data': 'real', 'query': 'complex', 'layout': 'C'}
    result = _run_case(case, 0, BarycentricRationalTransferFunction, 4, 1)
    assert set(result['groups']) == {'ordinary', 'support', 'near', 'near_real'}
    assert set(result['scalar_input_types']) == {'python_scalar', 'zero_dimensional_array'}
    for group in result['groups'].values():
        for operation in ('value', 'derivative'):
            assert group[operation]['max_scaled_change'] == 0
