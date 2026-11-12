# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Sampled barycentric evaluation benchmarks and block-AAA paper checks.

Run ``uv run --no-sync python -m benchmarks.barycentric --output results.json``.
The two rational functions and the 100 fitting samples reproduce Section 5.1 of
Gosea and Guettel, https://arxiv.org/abs/2003.06410 (reference example1.m at
https://github.com/nla-group/block_aaa). This is not the paper's full comparison
against RKFIT, vector fitting and Loewner methods. The small fitting routines
below generate benchmark data; they are not a public block-AAA implementation.
"""

import json
import platform
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import scipy
import scipy.linalg as spla
from cyclopts import App, Parameter
from cyclopts.validators import Number
from threadpoolctl import threadpool_info, threadpool_limits

from pymor.models.barycentric import BarycentricRationalTransferFunction


def _toy(s, nonsymmetric, derivative=False):
    s = np.asarray(s)
    values = np.empty(s.shape + (2, 2), dtype=np.result_type(s, np.complex128))
    upper = s**2 + s + (5 if nonsymmetric else -5)
    lower = s**2 + s - 5
    cubic = s**3 + 3 * s**2 - 1
    if derivative:
        values[..., 0, 0] = -2 / (s + 1)**2
        values[..., 0, 1] = (-upper - (3 - s) * (2 * s + 1)) / upper**2
        values[..., 1, 0] = (-lower - (3 - s) * (2 * s + 1)) / lower**2
        values[..., 1, 1] = (2 * s * cubic - (s**2 + 2) * (3 * s**2 + 6 * s)) / cubic**2
    else:
        values[..., 0, 0] = 2 / (s + 1)
        values[..., 0, 1] = (3 - s) / upper
        values[..., 1, 0] = (3 - s) / lower
        values[..., 1, 1] = (s**2 + 2) / cubic
    return values


def _fit(points, values, kind, max_supports=12, tol=1e-12):
    """Small greedy set-valued/block-AAA fit, solely for benchmark fixtures."""
    selected = []
    approximation = np.broadcast_to(values.mean(axis=0), values.shape)
    scale = np.max(np.linalg.norm(values, axis=(1, 2)))
    history = []
    for _ in range(max_supports):
        errors = np.linalg.norm(values - approximation, axis=(1, 2))
        errors[selected] = -1
        selected.append(int(np.argmax(errors)))
        remaining = np.ones(len(points), dtype=bool)
        remaining[selected] = False
        nodes, samples = points[selected], values[selected]
        loewner = (values[remaining, None] - samples[None, :]) / (
            points[remaining, None] - nodes[None, :]
        )[..., None, None]
        n, p, m = samples.shape
        if kind == 'scalar':
            L = loewner.transpose(0, 2, 3, 1).reshape(-1, n)
            _, _, vh = spla.svd(L, full_matrices=False)
            weights = vh[-1].conj()
        else:
            L = loewner.transpose(1, 2, 0, 3).reshape(n * p, -1)
            u, _, _ = spla.svd(L, full_matrices=False)
            weights = u[:, -p:].conj().T.reshape(p, n, p).transpose(1, 0, 2)
        tf = BarycentricRationalTransferFunction(nodes, samples, weights)
        approximation = np.array([tf.tf(s) for s in points])
        errors = np.linalg.norm(values - approximation, axis=(1, 2))
        history.append({'order': n - 1, 'rmse': float(np.sqrt(np.mean(errors**2))),
                        'max_relative_error': float(np.max(errors) / scale)})
        if np.max(errors) <= tol * scale:
            break
    return tf, history


class _DenseReference:
    """Independent arrowhead descriptor solve for the same zero-constant quotient."""

    def __init__(self, tf):
        nodes = tf.nodes
        values = tf.values.reshape(len(nodes), 1, 1) if tf.values.ndim == 1 else tf.values
        weights = tf.weights / np.max(abs(tf.weights))
        self._right = tf.side == 'right' and weights.ndim == 3
        if self._right:
            values, weights = values.swapaxes(1, 2), weights.swapaxes(1, 2)
        self._shape = values.shape[1:]
        n, p, m = values.shape
        self._scalar = weights.ndim == 1
        q = 1 if self._scalar else p
        self._q = q
        size = (n + 1) * q
        self.dimension = size
        self._matrix = np.zeros((size, size), dtype=complex)
        self._matrix[:q, q:] = np.tile(np.eye(q), (1, n))
        self._matrix[q:, :q] = weights.reshape(n * q, q)
        self._diagonal = np.arange(q, size)
        self._matrix[self._diagonal, self._diagonal] = -np.repeat(nodes, q)
        coefficients = weights[:, None, None] * values if self._scalar else weights @ values
        columns = p * m if self._scalar else m
        self._rhs = np.zeros((size, columns), dtype=complex)
        self._rhs[q:] = coefficients.reshape(n * q, columns)

    def evaluate(self, s, derivative=False):
        matrix = self._matrix.copy()
        matrix[self._diagonal, self._diagonal] += s
        factors = spla.lu_factor(matrix, check_finite=False)
        states = spla.lu_solve(factors, self._rhs, check_finite=False)
        if derivative:
            rhs = -states
            rhs[:self._q] = 0
            states = spla.lu_solve(factors, rhs, check_finite=False)
        result = states[:self._q].reshape(self._shape)
        return result.T if self._right else result


def _discrepancy(tf, reference, points):
    result = {}
    for derivative in (False, True):
        evaluate = tf.dtf if derivative else tf.tf
        errors = []
        for s in points:
            expected = reference(s, derivative)
            errors.append(spla.norm(evaluate(s) - expected) / max(1, spla.norm(expected)))
        result['derivative' if derivative else 'value'] = float(max(errors))
    return result


def _time(evaluate, points, derivative, repeat):
    evaluate(points[0], derivative)
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        for s in points:
            evaluate(s, derivative)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def _timings(tf, points, repeat):
    dense = _DenseReference(tf)
    check_points = points[np.linspace(0, len(points) - 1, 16, dtype=int)]
    discrepancy = _discrepancy(tf, dense.evaluate, check_points)
    if max(discrepancy.values()) > 1e-8:
        raise AssertionError(f'Dense descriptor reference disagrees: {discrepancy}')

    def direct(s, derivative=False):
        return tf.dtf(s) if derivative else tf.tf(s)

    result = {'dense_dimension': dense.dimension, 'dense_discrepancy': discrepancy}
    for derivative in (False, True):
        fast = _time(direct, points, derivative, repeat)
        slow = _time(dense.evaluate, points, derivative, repeat)
        result['derivative' if derivative else 'value'] = {
            'direct_seconds': fast, 'dense_seconds': slow, 'speedup': slow / fast,
        }
    return result


class BarycentricEvaluation:
    """ASV timings use uncached callables, not repeated cache hits."""

    version = 1
    param_names = ('kind', 'nonsymmetric')
    params = (['scalar', 'matrix'], [False, True])

    def setup(self, kind, nonsymmetric):
        samples = 1j * np.logspace(0, 2, 100)
        self.tf, history = _fit(samples, _toy(samples, nonsymmetric), kind)
        assert history[-1]['max_relative_error'] < 1e-10
        self.points = 1j * np.logspace(0, 2, 1000)

    def time_values(self, kind, nonsymmetric):
        for s in self.points:
            self.tf.tf(s)

    def time_derivatives(self, kind, nonsymmetric):
        for s in self.points:
            self.tf.dtf(s)


def _run(point_count, repeat, supports):
    points = 1j * np.logspace(0, 2, point_count)
    fitting_points = 1j * np.logspace(0, 2, 100)
    results = {'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                               'scipy': scipy.__version__, 'platform': platform.platform(),
                               'blas': threadpool_info()},
               'evaluation_points': point_count, 'repeat': repeat,
               'paper_examples': [], 'scaling': []}
    for nonsymmetric in (False, True):
        values = _toy(fitting_points, nonsymmetric)
        for kind in ('scalar', 'matrix'):
            tf, history = _fit(fitting_points, values, kind)
            reference = lambda s, derivative=False, nonsymmetric=nonsymmetric: _toy(s, nonsymmetric, derivative)
            errors = _discrepancy(tf, reference, points)
            near = np.concatenate([tf.nodes, tf.nodes + 1e-14 * (1 + abs(tf.nodes))])
            near_errors = _discrepancy(tf, reference, near)
            if max(errors.values()) > 1e-8 or max(near_errors.values()) > 1e-8:
                raise AssertionError(f'Paper example fit failed: {kind}, {errors}, {near_errors}')
            timed = _timings(tf, points, repeat)
            row = {'nonsymmetric': nonsymmetric, 'kind': kind, 'supports': len(tf.nodes),
                   'convergence': history, 'analytic_discrepancy': errors,
                   'support_discrepancy': near_errors, **timed}
            results['paper_examples'].append(row)
            print(f'paper symmetric={not nonsymmetric}, {kind}: supports={len(tf.nodes)}, '
                  f'RMSE={history[-1]["rmse"]:.2e}, value/derivative discrepancy={errors}', flush=True)

    rng = np.random.default_rng(0)
    scaling_points = 0.5 + points
    for n in supports:
        nodes = 1j * np.logspace(0, 2, n)
        values = _toy(nodes, True)
        for kind in ('scalar', 'matrix'):
            if kind == 'scalar':
                weights = rng.uniform(0.5, 1.5, n)
            else:
                perturbation = rng.normal(size=(n, 2, 2)) + 1j * rng.normal(size=(n, 2, 2))
                weights = np.eye(2) + 0.05 * (perturbation + perturbation.conj().swapaxes(1, 2))
            tf = BarycentricRationalTransferFunction(nodes, values, weights)
            timed = _timings(tf, scaling_points, repeat)
            results['scaling'].append({'supports': n, 'kind': kind, **timed})
            print(f'scaling n={n}, {kind}: value={timed["value"]["speedup"]:.2f}x, '
                  f'derivative={timed["derivative"]["speedup"]:.2f}x', flush=True)
    return results


app = App(help_on_error=True)

@app.default
def main(
    *,
    points: Annotated[int, Parameter(validator=Number(gte=16))] = 1000,
    repeat: Annotated[int, Parameter(validator=Number(gte=1))] = 3,
    supports: Annotated[
        list[int], Parameter(consume_multiple=1, negative=(), validator=Number(gte=1))
    ] = [8, 32, 128],
    output: Path | None = None,
):
    """Run sampled barycentric evaluation benchmarks and block-AAA paper checks.

    Parameters
    ----------
    points
        Number of evaluation points (at least 16).
    repeat
        Number of timing repetitions (at least 1).
    supports
        Positive support counts for the scaling benchmarks.
    output
        Optional path for the JSON results.
    """
    with threadpool_limits(limits=1):
        results = _run(points, repeat, supports)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    app()
