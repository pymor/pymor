# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Run explicitly; benchmark-only optional dependencies are not library dependencies."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from benchmarks.barycentric_landscape import (
    Case,
    _accuracy_points,
    _check_scipy_adapter,
    _extract_chebfun,
    _load_baseline,
    _metrics,
    _reference,
    _run_python,
    _scipy_model,
    _time,
    _write_report,
)
from pymor.models.barycentric import BarycentricRationalTransferFunction


def test_scipy_adapter_matches_public_aaa():
    _check_scipy_adapter()


@pytest.mark.parametrize('matrix', [False, True])
def test_reference_and_evaluators(matrix):
    pytest.importorskip('mpmath')
    values = np.array([1., 0.5])
    if matrix:
        values = values[:, None, None] * np.array([[1., 2.], [3., 4.]])
    case = Case('one_pole', np.array([0., 1.]), values, np.array([-1., 2.]), np.linspace(0, 1, 32), '')
    points = np.array([0., 1., 1e-15j, 1 + 1e-15j, 0.3j])
    value, derivative = _reference(case, points, 80)
    expected = 1 / (1 + points)
    factors = np.array([1., 2., 3., 4.]) if matrix else np.array([1.])
    np.testing.assert_allclose(value, expected[:, None] * factors)
    np.testing.assert_allclose(derivative, -expected[:, None]**2 * factors)
    scipy_model = _scipy_model(case)
    pymor = BarycentricRationalTransferFunction(case.nodes, case.values, case.weights)
    np.testing.assert_allclose(np.asarray(scipy_model(points)).reshape(value.shape), value)
    np.testing.assert_allclose(np.array([pymor.tf(s) for s in points]).reshape(value.shape), value)
    np.testing.assert_allclose(np.array([pymor.dtf(s) for s in points]).reshape(value.shape), derivative)


def test_runner_records_both_accuracy_modes_and_missing_derivatives():
    pytest.importorskip('baryrat')
    pytest.importorskip('mpmath')
    grid = np.array([0., 0.25, 0.5, 1.])
    case = Case('one_pole', np.array([0., 1.]), np.array([1., 0.5]), np.array([-1., 2.]), grid, '')
    value, derivative = _reference(case, grid, 80)
    rows = _run_python(case, {'ordinary': grid}, {'ordinary': {'value': value, 'derivative': derivative}},
                       grid, grid, 1)
    assert len(rows) == 6
    for row in rows:
        if row['backend'] == 'scipy_aaa' and row['operation'] == 'derivative':
            assert row['point']['status'] == 'unsupported'
            assert row['batch']['status'] == 'unsupported'
            assert row['accuracy'] == {}
        else:
            for mode in ('accuracy', 'accuracy_point'):
                assert row[mode]['ordinary']['max_scaled_error'] < 1e-12
            assert row['point']['points'] == len(grid)
            assert row['batch']['points'] == len(grid)


def test_accuracy_grid_includes_near_but_not_exact_points():
    case = Case('test', np.linspace(-1, 1, 128), np.ones(128), np.ones(128), np.linspace(-1, 1, 128), '')
    groups = _accuracy_points(case)
    assert not np.any(groups['near_supports'][:, None] == case.nodes[None, :])
    assert np.all(np.isin(groups['supports'], case.nodes))


def test_metrics_do_not_hide_nonfinite_results():
    assert _metrics([np.nan], np.array([[1.]])) == {'status': 'nonfinite', 'nonfinite': 1}
    assert _metrics([1 + 1e-9], np.array([[1.]]))['max_scaled_error'] > 9e-10


def test_timer_reports_exception_instead_of_zero_time():
    def fail(x):
        raise ValueError('unsupported')

    result = _time(fail, np.array([0., 1.]), 1, False)
    assert result['status'] == 'error'
    assert 'unsupported' in result['message']
    assert 'microseconds_per_point' not in result


def test_extract_keeps_kernel_algorithm_unchanged(tmp_path):
    source = ('function r = reval(x)\nr=x;\nend % End of REVAL.\n'
              'function dRx = diffbary(x)\ndRx=x;\nend % End of DIFFBARY.\n')
    source_dir, target = tmp_path / 'source', tmp_path / 'target'
    source_dir.mkdir()
    target.mkdir()
    (source_dir / 'aaa.m').write_text(source)
    # Isolate the external commit lookup from source extraction.
    import subprocess
    from unittest.mock import patch

    with patch.object(subprocess, 'check_output', return_value='test-commit\n'):
        metadata = _extract_chebfun(source_dir, target)
    assert metadata['commit'] == 'test-commit'
    output = (target / 'pymor_chebfun_reval.m').read_text()
    assert 'function r = pymor_chebfun_reval(x)\nr=x;\nend % End of REVAL.' in output
    assert 'reval' in metadata['kernel_sha256']


def test_load_frozen_pymor_source(tmp_path):
    source = tmp_path / 'baseline.py'
    source.write_text(Path(inspect.getfile(BarycentricRationalTransferFunction)).read_text())
    baseline = _load_baseline(source)
    assert baseline is not BarycentricRationalTransferFunction
    model = baseline([0, 1], [1, 0.5], [-1, 2])
    np.testing.assert_allclose(model.tf(0.3j), [[1 / (1 + 0.3j)]])


def test_runner_includes_and_evaluates_old_pymor(tmp_path):
    pytest.importorskip('baryrat')
    pytest.importorskip('mpmath')
    calls = []

    class Baseline(BarycentricRationalTransferFunction):
        def _evaluate(self, s):
            calls.append(s)
            return super()._evaluate(s)

    grid = np.array([0., 0.25, 0.5, 1.])
    case = Case('one_pole', np.array([0., 1.]), np.array([1., 0.5]), np.array([-1., 2.]), grid, '')
    value, derivative = _reference(case, grid, 80)
    rows = _run_python(case, {'ordinary': grid}, {'ordinary': {'value': value, 'derivative': derivative}},
                       grid, grid, 1, baseline=Baseline)
    assert calls
    assert len(rows) == 8
    assert {row['backend'] for row in rows} == {'pymor_old', 'pymor', 'baryrat', 'scipy_aaa'}
    for row in rows:
        if row['backend'] in ('pymor_old', 'pymor'):
            for mode in ('accuracy', 'accuracy_point'):
                assert row[mode]['ordinary']['max_scaled_error'] < 1e-12
    result = {'protocol': {'reference_decimal_digits': 80, 'reference_crosscheck_decimal_digits': 120},
              'environment': {}, 'source': {'pymor_old': {'path': 'baseline.py', 'sha256': 'old'},
                                           'pymor_class_sha256': 'optimized'},
              'cases': [{'name': case.name, 'description': '', 'supports': 2, 'shape': [1, 1]}], 'rows': rows}
    output = tmp_path / 'results.json'
    _write_report(result, output)
    text = output.with_suffix('.md').read_text()
    assert '| pyMOR old |' in text
    assert '| pyMOR optimized |' in text
    assert 'old timings were not copied' in text
