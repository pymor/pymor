# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Single-point sampled barycentric evaluation: frozen source versus current source.

Run with ``uv run --no-sync python -m benchmarks.barycentric_single
--baseline-source /path/to/before.py``. No fitting, construction, cache hits,
batching or reference calculations are included in the timings.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg as spla
from threadpoolctl import threadpool_info, threadpool_limits

from pymor.models.barycentric import BarycentricRationalTransferFunction


def _cases():
    cases = []

    def add(n, shape, form, layout='C'):
        for data, query in [('real', 'real'), ('real', 'complex'), ('complex', 'complex')]:
            cases.append({'supports': n, 'shape': shape, 'form': form,
                          'data': data, 'query': query, 'layout': layout})

    for n in [1, 2, 8, 32, 128, 512]:
        shapes = [(1, 1), (2, 3), (4, 4), (8, 8)]
        if n in [2, 32, 128]:
            shapes += [(1, 4), (4, 1)]
        for shape in shapes:
            add(n, shape, 'scalar')
        for shape in [(1, 1), (2, 3), (3, 2), (4, 4)]:
            for side in ['left', 'right']:
                add(n, shape, side)
    for n in [8, 128]:
        for size in [8, 32]:
            for side in ['left', 'right']:
                add(n, (size, size), side)
    for layout in ['F', 'negative']:
        for form in ['scalar', 'left', 'right']:
            add(32, (2, 3), form, layout)
    cases += [{**case, 'query': 'real'} for case in cases if case['data'] == 'complex']
    return cases


def _fixture(case, seed, count):
    rng = np.random.default_rng(seed)
    n, (p, m) = case['supports'], case['shape']
    nodes = np.linspace(-2., -0.2, n)
    values = rng.normal(size=(n, p, m))
    if case['data'] == 'complex':
        nodes = nodes + 1j * np.linspace(-0.15, 0.15, n)
        values = values + 1j * rng.normal(size=values.shape)
    if case['form'] == 'scalar':
        weights = rng.uniform(0.25, 1.25, n)
    else:
        size = p if case['form'] == 'left' else m
        factors = rng.normal(size=(n, size, size))
        if case['data'] == 'complex':
            factors = factors + 1j * rng.normal(size=factors.shape)
        weights = np.eye(size) + factors @ factors.conj().swapaxes(1, 2) / (2 * size)

    def arrange(array):
        if case['layout'] == 'negative':
            return array[::-1].copy()[::-1]
        return np.array(array, order=case['layout'])

    arrays = tuple(arrange(a) for a in (nodes, values, weights))
    selected = np.linspace(0, n - 1, count, dtype=int)
    ordinary = (np.linspace(0.1, 2., count) if case['query'] == 'real'
                else 0.2 + 1j * np.geomspace(0.01, 100., count))
    offset = 1e-13 * (1 + abs(nodes[selected]))
    points = {'ordinary': ordinary, 'support': nodes[selected],
              'near': nodes[selected] + offset * (0.7 + 1j)}
    if case['data'] == 'real':
        points['near_real'] = nodes[selected] + offset
    return arrays, points


def _reference(nodes, values, weights, side, s):
    """Independent unscaled quotient and exact-support derivative identities."""
    delta = s - nodes
    hits = np.flatnonzero(delta == 0)
    if len(hits):
        j = hits[0]
        keep = np.arange(len(nodes)) != j
        difference = values[keep] - values[j]
        if weights.ndim == 1:
            derivative = np.einsum('i,ijk->jk', weights[keep] / delta[keep], difference) / weights[j]
        else:
            products = weights[keep] @ difference if side == 'left' else difference @ weights[keep]
            rhs = np.einsum('i,ijk->jk', 1 / delta[keep], products)
            derivative = spla.solve(weights[j], rhs) if side == 'left' else spla.solve(weights[j].T, rhs.T).T
        return values[j], derivative
    if weights.ndim == 1:
        coefficients = weights / delta
        denominator = coefficients.sum()
        value = np.einsum('i,ijk->jk', coefficients, values) / denominator
        dcoefficients = -coefficients / delta
        derivative = (np.einsum('i,ijk->jk', dcoefficients, values)
                      - value * dcoefficients.sum()) / denominator
    else:
        products = weights @ values if side == 'left' else values @ weights
        basis, dbasis = 1 / delta, -(1 / delta) / delta
        numerator = np.einsum('i,ijk->jk', basis, products)
        denominator = np.einsum('i,ijk->jk', basis, weights)
        dN = np.einsum('i,ijk->jk', dbasis, products)
        dD = np.einsum('i,ijk->jk', dbasis, weights)
        if side == 'left':
            value = spla.solve(denominator, numerator)
            derivative = spla.solve(denominator, dN - dD @ value)
        else:
            value = spla.solve(denominator.T, numerator.T).T
            derivative = spla.solve(denominator.T, (dN - value @ dD).T).T
    return value, derivative


def _error(actual, expected):
    assert np.all(np.isfinite(actual)), 'Nonfinite evaluation.'
    assert np.all(np.isfinite(expected)), 'Nonfinite reference.'
    error = float(np.max(abs(actual - expected) / np.maximum(1., abs(expected))))
    assert error < 5e-10, f'Evaluation discrepancy: {error}'
    return error


def _measure(functions, points, repeat):
    for function in functions:
        function(points[0])
    samples = [[], []]
    enabled = gc.isenabled()
    gc.disable()
    try:
        for trial in range(repeat):
            for index in ([0, 1] if trial % 2 == 0 else [1, 0]):
                function = functions[index]
                start = time.perf_counter_ns()
                for s in points:
                    function(s)
                samples[index].append((time.perf_counter_ns() - start) / (1000 * len(points)))
    finally:
        if enabled:
            gc.enable()
    before, after = (float(np.median(sample)) for sample in samples)
    return {'before_us': before, 'after_us': after, 'speedup': before / after,
            'before_samples_us': samples[0], 'after_samples_us': samples[1]}


def _array_bytes(model):
    roots = {}
    for array in model.__dict__.values():
        if isinstance(array, np.ndarray):
            while isinstance(array.base, np.ndarray):
                array = array.base
            roots[id(array)] = array.nbytes
    return sum(roots.values())


def _run_case(case, index, baseline, count, repeat):
    arrays, point_groups = _fixture(case, 20261114 + index, count)
    side = 'right' if case['form'] == 'right' else 'left'
    models = [cls(*arrays, side=side) for cls in (baseline, BarycentricRationalTransferFunction)]
    for model in models:
        model.disable_caching()
    digest = hashlib.sha256()
    for array in (*arrays, *point_groups.values()):
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    record = {'case': case, 'fixture_sha256': digest.hexdigest(), 'groups': {},
              'stored_array_bytes': dict(zip(['before', 'after'], map(_array_bytes, models), strict=True))}
    independent = [0., 0.]
    for group, points in point_groups.items():
        record['groups'][group] = {}
        for derivative in (False, True):
            functions = [model.dtf if derivative else model.tf for model in models]
            results = [np.array([function(s) for s in points]) for function in functions]
            assert all(a.shape == (count, *case['shape']) for a in results)
            discrepancy = _error(results[1], results[0])
            if group in ('ordinary', 'support'):
                for j in np.unique(np.linspace(0, count - 1, min(count, 3), dtype=int)):
                    expected = _reference(*arrays, side, points[j])[derivative]
                    for k in (0, 1):
                        independent[k] = max(independent[k], _error(results[k][j], expected))
            result = _measure(functions, points, repeat)
            result['max_scaled_change'] = discrepancy
            record['groups'][group]['derivative' if derivative else 'value'] = result
    record['max_independent_reference_error'] = dict(zip(['before', 'after'], independent, strict=True))
    representative = (case['layout'] == 'C' and case['data'] == 'real' and case['query'] == 'complex'
                      and ((case['form'] == 'scalar' and case['shape'] == (1, 1)
                            and case['supports'] in (2, 128, 512))
                           or (case['form'] in ('left', 'right') and case['shape'] == (2, 3)
                               and case['supports'] == 32)))
    if representative:
        record['public_uncached'] = {}
        record['scalar_input_types'] = {}
        for derivative in (False, True):
            operation = 'derivative' if derivative else 'value'
            functions = [model.eval_dtf if derivative else model.eval_tf for model in models]
            record['public_uncached'][operation] = _measure(functions, point_groups['ordinary'], repeat)
            functions = [model.dtf if derivative else model.tf for model in models]
            inputs = {'python_scalar': point_groups['ordinary'].tolist(),
                      'zero_dimensional_array': [np.array(s) for s in point_groups['ordinary']]}
            for label, points in inputs.items():
                for s in points[:3]:
                    _error(functions[1](s), functions[0](s))
                record['scalar_input_types'].setdefault(label, {})[operation] = _measure(functions, points, repeat)
    return record


def _markdown(results):
    rows = results['cases']
    lines = ['# Single-point barycentric optimization', '',
             (f"{len(rows)} fixtures; {results['metadata']['points_per_group']} points per group; "
              f"median of {results['metadata']['repeat']} alternating passes; one BLAS thread."), '',
             'No fitting, construction, batching, reference work or cache hits in timings.', '',
             '| Group | Operation | Median speedup | Minimum | Maximum |',
             '|---|---|---:|---:|---:|']
    for group in ('ordinary', 'support', 'near', 'near_real'):
        for operation in ('value', 'derivative'):
            ratios = [row['groups'][group][operation]['speedup'] for row in rows if group in row['groups']]
            if not ratios:
                continue
            lines.append(f'| {group} | {operation} | {np.median(ratios):.2f}x '
                         f'| {min(ratios):.2f}x | {max(ratios):.2f}x |')
    lines += ['', '## Ordinary points excluding single-support constants', '',
              '| Weight form | Fixtures | Median value speedup | Median derivative speedup |',
              '|---|---:|---:|---:|']
    for form in ('scalar', 'left', 'right'):
        selected = [row for row in rows if row['case']['form'] == form and row['case']['supports'] > 1]
        if selected:
            value, derivative = (np.median([r['groups']['ordinary'][op]['speedup'] for r in selected])
                                 for op in ('value', 'derivative'))
            lines.append(f'| {form} | {len(selected)} | {value:.2f}x | {derivative:.2f}x |')
    worst = max(group[op]['max_scaled_change'] for row in rows
                for group in row['groups'].values() for op in ('value', 'derivative'))
    independent = max(row['max_independent_reference_error']['after'] for row in rows)
    footprints = [row['stored_array_bytes']['after'] / row['stored_array_bytes']['before'] for row in rows]
    lines += ['', f'Maximum scaled change versus frozen source: {worst:.3g}.',
              f'Maximum independent quotient/support-derivative reference error: {independent:.3g}.',
              f'Stored-array footprint ratio: {min(footprints):.2f} to {max(footprints):.2f}.',
              'Precomputed scalar numerators trade memory for speed; footprints exclude transient work arrays.', '',
              'Frozen source: `' + results['metadata']['baseline_source'] + '`.',
              'All fixture hashes, independent discrepancies, array footprints and raw timings are in the JSON.',
              'Reordered reductions can differ near ill-conditioned denominators; these fixtures are well-conditioned.',
              'Near-support checks use the frozen evaluator and independent unit-test identities.',
              '`query` labels the ordinary grid; `near` uses complex offsets; `near_real` uses real offsets.',
              '`near_real` is included only for real models. Layout labels describe constructor inputs.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('barycentric-single-results.json'))
    parser.add_argument('--points', type=int, default=128)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    assert args.points > 0
    assert args.repeat > 0
    assert args.limit is None or args.limit > 0
    spec = importlib.util.spec_from_file_location('_barycentric_before', args.baseline_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    baseline = module.BarycentricRationalTransferFunction
    current_source = Path(sys.modules[BarycentricRationalTransferFunction.__module__].__file__)
    metadata = {'python': platform.python_version(), 'platform': platform.platform(),
                'numpy': np.__version__, 'scipy': scipy.__version__, 'points_per_group': args.points,
                'repeat': args.repeat, 'fixture_seed': 20261114,
                'baseline_source': str(args.baseline_source.resolve()),
                'baseline_sha256': hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
                'current_sha256': hashlib.sha256(current_source.read_bytes()).hexdigest(),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'timing_scope': 'uncached tf/dtf single-point callables; selected uncached public methods',
                'error_metric': 'max(abs(actual-reference)/max(1,abs(reference)))'}
    results = {'metadata': metadata, 'cases': []}
    with threadpool_limits(limits=1):
        metadata['threadpools'] = threadpool_info()
        for index, case in enumerate(_cases()[:args.limit]):
            results['cases'].append(_run_case(case, index, baseline, args.points, args.repeat))
            gc.collect()
            if (index + 1) % 30 == 0:
                print(f'{index + 1} fixtures checked and timed.', flush=True)
    assert hashlib.sha256(current_source.read_bytes()).hexdigest() == metadata['current_sha256']
    assert hashlib.sha256(args.baseline_source.read_bytes()).hexdigest() == metadata['baseline_sha256']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == metadata['runner_sha256']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    markdown = _markdown(results)
    args.output.with_suffix('.md').write_text(markdown)
    print(markdown)


if __name__ == '__main__':
    main()
