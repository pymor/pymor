# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Compare fixed-data barycentric evaluation, not AAA fitting algorithms.

Run with benchmark-only dependencies, without modifying project dependencies::

    uv run --no-sync --with baryrat==2.1.2 --with mpmath==1.3.0 \
        python -m benchmarks.barycentric_landscape --output landscape.json

Chebfun is read from --chebfun-path (default: ~/.cache/pymor-benchmarks/chebfun).
Clone https://github.com/chebfun/chebfun there, or use --no-octave. SciPy >= 1.15
is required. Its actual AAA evaluator is supplied frozen data through a private
state adapter because it has no public constructor accepting existing weights.
Chebfun AAA's reval/diffbary functions are extracted from the recorded source,
with only their top-level names changed. The public Chebfun bary function is
reported separately. No derivative is invented for SciPy AAA or Chebfun bary.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import scipy
import scipy.io as spio
from cyclopts import App, Parameter
from cyclopts.validators import Number
from threadpoolctl import threadpool_info, threadpool_limits

from pymor.models.barycentric import BarycentricRationalTransferFunction


@dataclass
class Case:
    name: str
    nodes: np.ndarray
    values: np.ndarray
    weights: np.ndarray
    grid: np.ndarray
    description: str

    @property
    def flat_values(self):
        return self.values.reshape(len(self.nodes), -1)


def _scipy_model(case):
    from scipy.interpolate import AAA

    model = AAA.__new__(AAA)
    model._support_points = case.nodes
    model._support_values = case.values
    model.weights = case.weights
    model._shape = case.values.shape[1:]
    model._axis = 0
    return model


def _check_scipy_adapter():
    from scipy.interpolate import AAA

    x = np.linspace(-1, 1, 101)
    original = AAA(x, np.exp(x), clean_up=False)
    case = Case('check', original.support_points, original.support_values,
                original.weights, x, 'Public AAA adapter self-check')
    cloned = _scipy_model(case)
    checks = np.concatenate([x[::7], original.support_points])
    np.testing.assert_array_equal(cloned(checks), original(checks))


def _cases(points, supports):
    from scipy.interpolate import AAA, FloaterHormannInterpolator

    real_grid = np.linspace(-0.99, 0.99, points)
    cases = [Case('toy_one_pole', np.array([0., 1.]), np.array([1., 0.5]), np.array([-1., 2.]),
                  np.linspace(-0.5, 2, points), 'Exact H(s)=1/(s+1), two supports.')]
    samples = np.linspace(-1, 1, 201)
    fitted = AAA(samples, np.exp(samples), rtol=1e-13, max_terms=30, clean_up=False)
    cases.append(Case('toy_exp_aaa', fitted.support_points, fitted.support_values, fitted.weights,
                      real_grid, 'Frozen SciPy AAA fit of exp on [-1,1].'))
    samples = 1j * np.logspace(0, 2, 100)
    values = (3 - samples) / (samples**2 + samples + 5)
    fitted = AAA(samples, values, rtol=1e-13, max_terms=30, clean_up=False)
    cases.append(Case('toy_complex_rational', fitted.support_points, fitted.support_values, fitted.weights,
                      1j * np.logspace(0, 2, points), 'Complex (1,2) entry of the block-AAA nonsymmetric toy.'))
    for n in supports:
        nodes = np.linspace(-1, 1, n)
        values = np.sin(25 * nodes) + 1 / (1 + 25 * nodes**2)
        fitted = FloaterHormannInterpolator(nodes, values, d=3)
        cases.append(Case(f'large_fh_{n}', nodes, values, fitted.weights, real_grid,
                          'Degree-3 Floater-Hormann weights; many-support rational interpolant.'))
    rng = np.random.default_rng(0)
    n = min(128, max(supports))
    nodes = np.linspace(-1, 1, n)
    fitted = FloaterHormannInterpolator(nodes, np.sin(nodes), d=3)
    for size in (2, 4):
        A, B = rng.normal(size=(2, size, size))
        values = (np.sin(12 * nodes)[:, None, None] * A
                  + (1 / (1 + 9 * nodes**2))[:, None, None] * B)
        cases.append(Case(f'shared_mimo_{size}x{size}', nodes, values, fitted.weights, real_grid,
                          'Matrix samples with shared scalar weights; adapters explicitly labeled.'))
    return cases


def _accuracy_points(case):
    ordinary = case.grid[np.linspace(0, len(case.grid) - 1, 32, dtype=int)]
    indices = np.linspace(0, len(case.nodes) - 1, min(12, len(case.nodes)), dtype=int)
    exact = case.nodes[indices]
    scale = max(1., float(np.max(abs(case.nodes))))
    direction = 1 if np.iscomplexobj(case.nodes) else 1j
    near = np.concatenate([exact + direction * scale * distance for distance in (1e-4, 1e-8, 1e-12, 1e-15)])
    assert not np.any(near[:, None] == case.nodes[None, :])
    return {'ordinary': ordinary, 'supports': exact, 'near_supports': near}


def _reference(case, points, dps):
    """High-precision value/derivative of the SAME binary64 representation."""
    import mpmath as mp

    def convert(x):
        x = complex(x)
        return mp.mpc(mp.mpf(x.real), mp.mpf(x.imag))

    with mp.workdps(dps):
        z, w = list(map(convert, case.nodes)), list(map(convert, case.weights))
        values = [[convert(x) for x in col] for col in case.flat_values.T]
        result, derivatives = [], []
        for point in points:
            s = convert(point)
            matches = [i for i, node in enumerate(z) if s == node]
            if matches:
                j = matches[0]
                keep = [i for i in range(len(z)) if i != j]
                a = [w[i] / (z[j] - z[i]) for i in keep]
                result.append([complex(col[j]) for col in values])
                derivatives.append([complex(mp.fdot(a, [col[i] - col[j] for i in keep]) / w[j])
                                    for col in values])
            else:
                a = [wi / (s - zi) for zi, wi in zip(z, w, strict=True)]
                da = [-ai / (s - zi) for zi, ai in zip(z, a, strict=True)]
                D, dD = mp.fsum(a), mp.fsum(da)
                H = [mp.fdot(a, col) / D for col in values]
                result.append([complex(h) for h in H])
                derivatives.append([complex((mp.fdot(da, col) - h * dD) / D)
                                    for col, h in zip(values, H, strict=True)])
    return np.asarray(result), np.asarray(derivatives)


def _metrics(actual, reference):
    actual = np.asarray(actual).reshape(reference.shape)
    bad = int(np.count_nonzero(~np.isfinite(actual)))
    if bad:
        return {'status': 'nonfinite', 'nonfinite': bad}
    error = abs(actual - reference) / np.maximum(1, abs(reference))
    return {'status': 'ok', 'nonfinite': 0, 'max_scaled_error': float(np.max(error)),
            'rms_scaled_error': float(np.sqrt(np.mean(error**2)))}


def _time(evaluate, grid, repeat, point_mode):
    try:
        evaluate(grid[0]) if point_mode else evaluate(grid[:1])
        durations = []
        for _ in range(repeat):
            start = time.perf_counter()
            if point_mode:
                for s in grid:
                    evaluate(s)
            else:
                evaluate(grid)
            durations.append(time.perf_counter() - start)
        median = float(np.median(durations))
        return {'status': 'ok', 'median_seconds': median, 'samples_seconds': durations,
                'points': len(grid), 'microseconds_per_point': median / len(grid) * 1e6}
    except Exception as error:
        return {'status': 'error', 'message': f'{type(error).__name__}: {error}'}


def _load_baseline(source):
    spec = importlib.util.spec_from_file_location('_pymor_landscape_baseline', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BarycentricRationalTransferFunction


def _python_backends(case, baseline=None):
    import baryrat

    def pymor_backend(cls):
        tf = cls(case.nodes, case.values, case.weights)
        return {'value': (tf.tf, lambda x: np.stack([tf.tf(s) for s in x])),
                'derivative': (tf.dtf, lambda x: np.stack([tf.dtf(s) for s in x]))}

    scalar_models = [baryrat.BarycentricRational(case.nodes, col, case.weights) for col in case.flat_values.T]
    if len(scalar_models) == 1:
        br_value, br_derivative = scalar_models[0], scalar_models[0].eval_deriv
    else:
        def br_value(x):
            return np.stack([model(x) for model in scalar_models], axis=-1)

        def br_derivative(x):
            return np.stack([model.eval_deriv(x) for model in scalar_models], axis=-1)

    scipy_model = _scipy_model(case)
    backends = {'pymor_old': pymor_backend(baseline)} if baseline is not None else {}
    backends.update({
        'pymor': pymor_backend(BarycentricRationalTransferFunction),
        'baryrat': {'value': (br_value, br_value), 'derivative': (br_derivative, br_derivative)},
        'scipy_aaa': {'value': (scipy_model, scipy_model), 'derivative': None},
    })
    return backends


def _run_python(case, groups, refs, point_grid, derivative_grid, repeat, baseline=None):
    rows = []
    for backend, operations in _python_backends(case, baseline).items():
        for operation, functions in operations.items():
            row = {'case': case.name, 'backend': backend, 'operation': operation,
                   'accuracy': {}, 'accuracy_point': {}}
            if functions is None:
                row.update(point={'status': 'unsupported'}, batch={'status': 'unsupported'})
                rows.append(row)
                continue
            point, batch = functions
            grid = derivative_grid if operation == 'derivative' else case.grid
            point_subset = derivative_grid if operation == 'derivative' else point_grid
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                row['point'] = _time(point, point_subset, repeat, True)
                row['batch'] = _time(batch, grid, repeat, False)
                for name, samples in groups.items():
                    expected = refs[name][operation]
                    for mode in ('accuracy', 'accuracy_point'):
                        try:
                            actual = batch(samples) if mode == 'accuracy' else np.array([point(s) for s in samples])
                            row[mode][name] = _metrics(actual, expected)
                        except Exception as error:
                            row[mode][name] = {'status': 'error', 'message': f'{type(error).__name__}: {error}'}
                row['warnings'] = sorted({str(w.message) for w in caught})[:5]
            rows.append(row)
            print(f'{case.name} {backend} {operation}: point={row["point"]["status"]} '
                  f'batch={row["batch"]["status"]}', flush=True)
    return rows


def _extract_chebfun(chebfun, destination):
    source = (chebfun / 'aaa.m').read_text()
    hashes = {}
    for name, result in [('reval', 'r'), ('diffbary', 'dRx')]:
        start = source.index(f'function {result} = {name}(')
        marker = f'end % End of {name.upper()}.'
        end = source.index(marker, start) + len(marker)
        body = source[start:end]
        hashes[name] = hashlib.sha256(body.encode()).hexdigest()
        body = body.replace(f'function {result} = {name}(', f'function {result} = pymor_chebfun_{name}(', 1)
        header = ('% Extracted unchanged except the function name from Chebfun aaa.m.\n'
                  '% Copyright The University of Oxford and the Chebfun Developers.\n'
                  '% See the LICENSE.txt in the recorded Chebfun checkout.\n')
        (destination / f'pymor_chebfun_{name}.m').write_text(header + body + '\n')
    commit = subprocess.check_output(['git', '-C', str(chebfun), 'rev-parse', 'HEAD'], text=True).strip()
    return {'commit': commit, 'kernel_sha256': hashes,
            'aaa_source_sha256': hashlib.sha256(source.encode()).hexdigest()}


def _quote_octave(path):
    return "'" + str(path).replace("'", "''") + "'"


def _run_octave(payload, repeat, chebfun, timeout):
    executable = shutil.which('octave-cli') or shutil.which('octave')
    if executable is None:
        raise RuntimeError('Octave not found; install Octave or use --no-octave.')
    if not (chebfun / 'aaa.m').is_file() or not (chebfun / 'bary.m').is_file():
        raise RuntimeError(f'Chebfun checkout not found at {chebfun}; clone it or set --chebfun-path.')
    with tempfile.TemporaryDirectory(prefix='pymor-landscape-') as directory:
        temporary = Path(directory)
        metadata = _extract_chebfun(chebfun, temporary)
        input_path, output_path = temporary / 'input.mat', temporary / 'output.json'
        spio.savemat(input_path, {'cases': np.array(payload, dtype=object), 'repeat': repeat})
        script_dir = Path(__file__).resolve().parent
        command = (f'addpath({_quote_octave(script_dir)}); '
                   f'pymor_barycentric_landscape({_quote_octave(input_path)}, {_quote_octave(output_path)}, '
                   f'{_quote_octave(chebfun.resolve())}, {_quote_octave(temporary)});')
        env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        completed = subprocess.run([executable, '--quiet', '--eval', command], env=env,
                                   text=True, capture_output=True, timeout=timeout, check=False)
        if completed.returncode:
            raise RuntimeError(f'Octave benchmark failed: {completed.stdout[-2000:]}\n{completed.stderr[-2000:]}')
        result = json.loads(output_path.read_text())
        for row in result['rows']:
            for mode in ('accuracy', 'accuracy_point'):
                for metric in row.get(mode, {}).values():
                    if metric['status'] == 'ok':
                        for name in ('max_scaled_error', 'rms_scaled_error'):
                            metric[name] = float(metric[name])
        result['source'] = metadata
        result['stdout'] = completed.stdout
        result['stderr'] = completed.stderr
        return result


def _case_metadata(case):
    digest = hashlib.sha256()
    for array in (case.nodes, case.values, case.weights):
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return {'name': case.name, 'supports': len(case.nodes), 'shape': list(case.values.shape[1:]) or [1, 1],
            'channels': case.flat_values.shape[1], 'description': case.description,
            'fixture_sha256': digest.hexdigest()}


def _run(points, point_points, derivative_points, supports, repeat, dps, octave, chebfun, timeout,
         baseline_source=None):
    _check_scipy_adapter()
    baseline = _load_baseline(baseline_source) if baseline_source is not None else None
    cases = _cases(points, supports)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'platform': platform.platform(),
                              'baryrat': importlib.metadata.version('baryrat'),
                              'mpmath': importlib.metadata.version('mpmath'), 'blas': threadpool_info()},
              'protocol': {'same_fixed_representation': True, 'fitting_timed': False, 'cache_enabled': False,
                           'reference_decimal_digits': dps, 'reference_crosscheck_decimal_digits': dps + 40,
                           'repeat': repeat, 'batch_value_points': points,
                           'point_value_points': point_points, 'derivative_points': derivative_points,
                           'scipy_private_frozen_adapter_checked': True},
              'source': {'pymor_class_sha256': hashlib.sha256(
                  (Path(__file__).resolve().parents[1] / 'src/pymor/models/barycentric.py').read_bytes()).hexdigest(),
                         'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'octave_driver_sha256': hashlib.sha256(
                             Path(__file__).with_name('pymor_barycentric_landscape.m').read_bytes()).hexdigest()},
              'cases': [], 'rows': []}
    if baseline_source is not None:
        result['source']['pymor_old'] = {'path': str(baseline_source.resolve()),
                                         'sha256': hashlib.sha256(baseline_source.read_bytes()).hexdigest()}
        result['protocol']['pymor_old_and_optimized_fresh_same_process'] = True
    payload, fixtures = [], {}
    for case in cases:
        groups = _accuracy_points(case)
        refs = {}
        for name, samples in groups.items():
            value, derivative = _reference(case, samples, dps)
            refs[name] = {'value': value, 'derivative': derivative}
        checks = np.concatenate([groups['ordinary'][:1], groups['near_supports'][-1:]])
        checked_value, checked_derivative = _reference(case, checks, dps + 40)
        for operation, checked in [('value', checked_value), ('derivative', checked_derivative)]:
            original = np.concatenate([refs['ordinary'][operation][:1], refs['near_supports'][operation][-1:]])
            np.testing.assert_array_equal(checked, original)
        point_grid = case.grid[np.linspace(0, points - 1, point_points, dtype=int)]
        derivative_grid = case.grid[np.linspace(0, points - 1, derivative_points, dtype=int)]
        result['cases'].append(_case_metadata(case))
        result['rows'].extend(_run_python(case, groups, refs, point_grid, derivative_grid, repeat, baseline))
        group_payload = [{'name': name, 'points': samples[:, None], 'reference_value': refs[name]['value'],
                          'reference_derivative': refs[name]['derivative']} for name, samples in groups.items()]
        payload.append({'name': case.name, 'nodes': case.nodes[:, None], 'values': case.flat_values,
                        'weights': case.weights[:, None], 'value_grid': case.grid[:, None],
                        'point_grid': point_grid[:, None], 'derivative_grid': derivative_grid[:, None],
                        'accuracy': np.array(group_payload, dtype=object)})
        for name, data in [('nodes', case.nodes), ('values', case.values), ('weights', case.weights)]:
            fixtures[f'{case.name}_{name}'] = data
    if octave:
        print('Running Octave/Chebfun; process startup and data transfer are not timed.', flush=True)
        octave_result = _run_octave(payload, repeat, chebfun, timeout)
        result['rows'].extend(octave_result.pop('rows'))
        result['octave'] = octave_result
    return result, fixtures


def _number(value):
    return f'{value:.3g}' if value is not None else '—'


def _timing(row, mode):
    timing = row.get(mode, {})
    return _number(timing.get('microseconds_per_point')) if timing.get('status') == 'ok' else timing.get('status', '—')


def _error(row, group, point_mode=False):
    metric = row.get('accuracy_point' if point_mode else 'accuracy', {}).get(group, {})
    return _number(metric.get('max_scaled_error')) if metric.get('status') == 'ok' else metric.get('status', '—')


def _write_report(result, output):
    labels = {'pymor_old': 'pyMOR old', 'pymor': 'pyMOR optimized'}
    text = """# Barycentric software-landscape benchmark

## Protocol

- Every evaluator receives identical nodes, sample values and weights. AAA fitting is not timed.
- Python calls bypass result caches. Point and batch times are separate, in microseconds per point.
- pyMOR batch evaluation uses an explicit Python loop; there is no native batch API.
- baryrat and Chebfun AAA use per-entry adapters for matrix samples, not native MIMO support.
- SciPy uses its actual AAA value evaluator with frozen private state. The adapter is checked
  against a public AAA object. This does not claim public matrix-valued AAA fitting support.
- Chebfun AAA uses its source reval/diffbary kernels under Octave. Only function names change.
  Chebfun public bary is a separate value-only backend.
- SciPy AAA and Chebfun bary have no derivative method in this comparison.
  No replacement derivative formula is added.
- High-precision references evaluate the same stored binary64 quotient, then round to binary64.
  This measures evaluation error, not approximation error against a target function.
- Error = max(abs(actual-reference) / max(1,abs(reference))) over points and output entries.
  Nonfinite outputs and exceptions are reported, not discarded.
- One BLAS thread is requested. Timings are medians of repeated passes. Octave startup, MAT
  transfer, package loading and reference calculations are excluded. BLAS libraries can differ.
- Full matrix weights are not compared: scalar competitors cannot express that form directly.
- Accuracy is checked for point and batch calls separately. Grids contain 32 ordinary points,
  up to 12 exact supports, and nearby offsets from 1e-4 to 1e-15 times max(1,max(abs(nodes))).
- Octave accuracy metrics cross JSON as decimal strings to avoid its sub-epsilon rounding.
""".splitlines()
    text += ['', (f'Reference precision: {result["protocol"]["reference_decimal_digits"]} decimal digits; '
                  f'cross-checked at {result["protocol"]["reference_crosscheck_decimal_digits"]} digits.'),
             '', '## Environment', '', '```json', json.dumps(result['environment'], indent=2), '```', '']
    if 'pymor_old' in result['source']:
        baseline = result['source']['pymor_old']
        text += ['## pyMOR versions', '',
                 '- Both versions were measured afresh in this process; old timings were not copied.',
                 f'- Old source: `{baseline["path"]}`; SHA256 `{baseline["sha256"]}`.',
                 f'- Optimized source SHA256: `{result["source"]["pymor_class_sha256"]}`.', '',
                 'The two pyMOR bulk paths are Python loops, not native batch algorithms.', '']
    if 'octave' in result:
        metadata = {k: v for k, v in result['octave'].items() if k not in ('stdout', 'stderr')}
        text += ['### Octave/Chebfun', '', '```json', json.dumps(metadata, indent=2), '```', '']
    for case in result['cases']:
        text += [f'## {case["name"]}', '',
                 f'{case["description"]} Supports: {case["supports"]}; shape: {case["shape"]}.', '',
                 '| Backend | Value point | Value batch | Derivative point | Derivative batch |',
                 '|---|---:|---:|---:|---:|']
        rows = {(v['backend'], v['operation']): v for v in result['rows'] if v['case'] == case['name']}
        backends = list(dict.fromkeys(key[0] for key in rows))
        for backend in backends:
            value, derivative = rows[backend, 'value'], rows[backend, 'derivative']
            text.append(f'| {labels.get(backend, backend)} | {_timing(value, "point")} | {_timing(value, "batch")} '
                        f'| {_timing(derivative, "point")} | {_timing(derivative, "batch")} |')
        text += ['', 'Accuracy entries are **point / batch** max scaled errors.', '',
                 '| Backend | Value ordinary | Value near | Derivative ordinary | Derivative near |',
                 '|---|---:|---:|---:|---:|']
        for backend in backends:
            pairs = [labels.get(backend, backend)]
            for operation in ('value', 'derivative'):
                for group in ('ordinary', 'near_supports'):
                    row = rows[backend, operation]
                    pairs.append(f'{_error(row, group, True)} / {_error(row, group)}')
            text.append('| ' + ' | '.join(pairs) + ' |')
        text.append('')
    text += ['## Errors and limitations', '']
    for row in result['rows']:
        for mode in ('point', 'batch'):
            timing = row.get(mode, {})
            if timing.get('status') == 'error':
                text.append(f'- {row["case"]} / {row["backend"]} / {row["operation"]} / {mode}: {timing["message"]}')
        for mode in ('accuracy', 'accuracy_point'):
            for group, metric in row.get(mode, {}).items():
                if metric.get('status') != 'ok':
                    label = f'{row["case"]} / {row["backend"]} / {row["operation"]} / {mode} / {group}'
                    text.append(f'- {label}: {metric}')
    text += ['', ('Results are specific to these fixtures and machine. Speed does not imply acceptable accuracy. '
                  'Check near-support derivative errors separately from ordinary-point timings.'), '',
             ('The NPZ file stores the exact common inputs. '
              'Re-run with the recorded package versions and Chebfun commit.'), '']
    output.with_suffix('.md').write_text('\n'.join(text))


app = App(help_on_error=True)


@app.default
def main(
    *,
    output: Path = Path('barycentric-landscape-results.json'),
    points: Annotated[int, Parameter(validator=Number(gte=32))] = 4096,
    point_points: Annotated[int, Parameter(validator=Number(gte=16))] = 128,
    derivative_points: Annotated[int, Parameter(validator=Number(gte=16))] = 64,
    repeat: Annotated[int, Parameter(validator=Number(gte=1))] = 3,
    supports: Annotated[
        list[int], Parameter(consume_multiple=1, negative=(), validator=Number(gte=4))
    ] = [32, 128, 512],
    dps: Annotated[int, Parameter(validator=Number(gte=40))] = 80,
    octave: bool = True,
    chebfun_path: Path | None = None,
    octave_timeout: Annotated[int, Parameter(validator=Number(gte=1))] = 600,
    baseline_source: Path | None = None,
):
    """Run a fixed-representation accuracy and speed comparison across packages."""
    if point_points > points or derivative_points > points:
        raise ValueError('Point and derivative counts must not exceed the value-grid size.')
    chebfun = (chebfun_path
               or Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'pymor-benchmarks/chebfun')
    with threadpool_limits(limits=1):
        result, fixtures = _run(points, point_points, derivative_points, supports, repeat, dps,
                                octave, chebfun, octave_timeout, baseline_source)
    if baseline_source is not None:
        assert hashlib.sha256(baseline_source.read_bytes()).hexdigest() == result['source']['pymor_old']['sha256']
    current_source = Path(__file__).resolve().parents[1] / 'src/pymor/models/barycentric.py'
    assert hashlib.sha256(current_source.read_bytes()).hexdigest() == result['source']['pymor_class_sha256']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == result['source']['benchmark_sha256']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(output.with_suffix('.npz'), **fixtures)
    _write_report(result, output)
    print(f'Results: {output}; report: {output.with_suffix(".md")}', flush=True)


if __name__ == '__main__':
    app()
