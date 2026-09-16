# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.linalg as spla

from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction
from pymor.reductors.quadbt import QuadBTReductor

pytestmark = pytest.mark.builtin


@pytest.mark.parametrize('sampling_time', [0, .1])
@pytest.mark.parametrize('shape', ['siso', 'matrix_siso', 'mimo'])
@pytest.mark.parametrize('shared_nodes', [False, True])
@pytest.mark.parametrize('real', [False, True])
def test_quadbt_resolvent_factors(rng, sampling_time, shape, shared_nodes, real):
    n = 3
    p, m = (2, 3) if shape == 'mimo' else (1, 1)
    E = np.array([[2., .1, 0], [0, 1., .2], [.1, 0, 3.]])
    F = np.diag([.2, .5, .8] if sampling_time else [-.5, -2., -5.]) + np.diag([.1, .2], 1)
    A, B, C, D = E @ F, rng.normal(size=(n, m)), rng.normal(size=(p, n)), rng.normal(size=(p, m))
    left = 1j * np.array([0, .7, -.7, 2., -2.])
    right = 1j * np.array([.3, -.3, 1.2, -1.2])
    wl, wr = np.array([.2, .4, .4, .7, .7]), np.array([.3, .3, .6, .6])
    if sampling_time:
        left, right = np.exp(left), np.exp(right)
    if shared_nodes:
        right, wr = left.copy(), 2 * wl

    def resolvent(z):
        return spla.solve(z * E - A, np.eye(n))

    def transfer_function(z):
        return C @ resolvent(z) @ B + D

    lv = np.array([transfer_function(z) for z in left])
    rv = np.array([transfer_function(z) for z in right])
    derivatives = np.array([-C @ resolvent(z) @ E @ resolvent(z) @ B for z in left]) if shared_nodes else None
    if shape == 'siso':
        lv, rv = lv[:, 0, 0], rv[:, 0, 0]
        if derivatives is not None:
            derivatives = derivatives[:, 0, 0]
    reductor = QuadBTReductor(left, right, lv, rv, wl, wr, derivatives=derivatives,
                             feedthrough=D, sampling_time=sampling_time, real=real)
    K, M, Bd, Cd = reductor.quadrature_matrices()
    assert K.shape == M.shape == (len(left) * p, len(right) * m)
    assert Bd.shape == (len(left) * p, m)
    assert Cd.shape == (p, len(right) * m)

    O = np.vstack([np.sqrt(w) * C @ resolvent(z) for z, w in zip(left, wl, strict=True)])
    R = np.hstack([np.sqrt(w) * resolvent(z) @ B for z, w in zip(right, wr, strict=True)])
    reference = (O @ E @ R, O @ A @ R, O @ B, C @ R)
    if real:
        assert all(np.isrealobj(matrix) for matrix in (K, M, Bd, Cd))
        for actual, expected in zip((K, M, Bd, Cd), reference, strict=True):
            assert np.allclose(spla.svdvals(actual), spla.svdvals(expected))
    else:
        for actual, expected in zip((K, M, Bd, Cd), reference, strict=True):
            assert np.allclose(actual, expected)

    # Compare a genuinely reduced ROM to projection with explicit resolvent factors.
    U, sv, Vh = spla.svd(reference[0], full_matrices=False)
    X = U[:, :2].conj().T / np.sqrt(sv[:2])[:, np.newaxis]
    Y = Vh[:2].conj().T / np.sqrt(sv[:2])
    Ar, Br, Cr = X @ reference[1] @ Y, X @ reference[2], reference[3] @ Y
    reduced, recovered = reductor.reduce(2), reductor.reduce(n)
    assert reduced.order == 2
    assert reduced.sampling_time == sampling_time
    for z in [1.1 + .4j, .2 + 2.3j, 0]:
        expected = Cr @ spla.solve(z * np.eye(2) - Ar, Br) + D
        assert np.allclose(reduced.transfer_function.eval_tf(z), expected)
        assert np.allclose(recovered.transfer_function.eval_tf(z), transfer_function(z))
    assert np.allclose(recovered.D.matrix, D)
    if real:
        assert all(np.isrealobj(op.matrix) for op in (reduced.A, reduced.B, reduced.C))
    with pytest.raises(ValueError, match='exceeds the numerical rank 3'):
        reductor.reduce(n + 1)


def _siso_reductor(**kwargs):
    left, right = np.array([1j, -1j]), np.array([2j, -2j])
    args = dict(left_nodes=left, right_nodes=right, left_values=1 / (left + 1),
                right_values=1 / (right + 1), left_weights=np.ones(2), right_weights=np.ones(2))
    return QuadBTReductor(**(args | kwargs))


@pytest.mark.parametrize(('kwargs', 'message'), [
    ({'left_nodes': []}, 'left_nodes'),
    ({'right_nodes': [np.inf, -2j]}, 'right_nodes'),
    ({'left_nodes': [1j, 1j]}, 'distinct'),
    ({'left_nodes': [1j, 2j]}, 'conjugate pairs'),
    ({'left_weights': [1]}, 'left_weights'),
    ({'right_weights': [-1, -1]}, 'right_weights'),
    ({'left_weights': [np.nan, np.nan]}, 'left_weights'),
    ({'left_weights': [np.inf, np.inf]}, 'left_weights'),
    ({'left_weights': [1j, 1j]}, 'left_weights'),
    ({'right_weights': [1, 2]}, 'equal at conjugate nodes'),
    ({'left_values': [1]}, 'left_values'),
    ({'left_values': [np.nan, np.nan]}, 'left_values'),
    ({'right_values': np.ones((2, 2, 1))}, 'matching input and output'),
    ({'derivatives': np.ones((2, 2, 1))}, 'derivatives must have the same shape'),
    ({'sampling_time': -.1}, 'sampling_time'),
    ({'sampling_time': np.inf}, 'sampling_time'),
    ({'feedthrough': np.ones((2, 1))}, 'feedthrough'),
    ({'feedthrough': np.nan}, 'feedthrough'),
    ({'feedthrough': 1j}, 'feedthrough must be real'),
])
def test_quadbt_invalid_data(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _siso_reductor(**kwargs)


def test_quadbt_missing_derivatives():
    nodes = np.array([1j, -1j])
    reductor = _siso_reductor(right_nodes=nodes, right_values=1 / (nodes + 1))
    with pytest.raises(ValueError, match='require derivative_terms'):
        reductor.reduce(1)


def test_quadbt_nonreal_samples():
    with pytest.raises(ValueError, match='not real'):
        _siso_reductor(left_values=[1, 2j]).reduce(1)


@pytest.mark.parametrize('r', [0, -1, 1.5, True, None, 2])
def test_quadbt_invalid_order(r):
    with pytest.raises(ValueError, match=r'positive integer|exceeds the numerical rank'):
        _siso_reductor().reduce(r)


@pytest.mark.parametrize('rank_tol', [-1, 1, np.nan, np.inf, 1j, [1e-12]])
def test_quadbt_invalid_rank_tolerance(rank_tol):
    with pytest.raises(ValueError, match='rank_tol'):
        _siso_reductor().reduce(1, rank_tol=rank_tol)


@pytest.mark.parametrize('kwargs', [{'left_values': [0, 0], 'right_values': [0, 0]},
                                   {'left_weights': [0, 0]}])
def test_quadbt_zero_rank(kwargs):
    with pytest.raises(ValueError, match='numerical rank 0'):
        _siso_reductor(**kwargs).reduce(1)


def test_quadbt_cached_svd(monkeypatch):
    calls = []
    svd = spla.svd

    def counting_svd(*args, **kwargs):
        calls.append(1)
        return svd(*args, **kwargs)

    monkeypatch.setattr(spla, 'svd', counting_svd)
    reductor = _siso_reductor()
    reductor.reduce(1)
    reductor.reduce(1, rank_tol=1e-10)
    assert len(calls) == 1


def test_quadbt_scalar_feedthrough_and_complex_rom():
    left, right = np.array([1j, -1j]), np.array([2j, -2j])
    D = 2 + 3j
    reductor = _siso_reductor(left_values=1 / (left + 1) + D, right_values=1 / (right + 1) + D,
                             left_weights=[1, 2], right_weights=[3, 4], feedthrough=D, real=False)
    rom = reductor.reduce(np.int64(1))
    assert np.allclose(rom.transfer_function.eval_tf(3j), 1 / (3j + 1) + D)


@pytest.mark.parametrize('sampling_time', [0, .1])
@pytest.mark.parametrize('mimo', [False, True])
@pytest.mark.parametrize('model_input', [False, True])
@pytest.mark.parametrize('shared_nodes', [False, True])
def test_quadbt_from_model(rng, sampling_time, mimo, model_input, shared_nodes):
    p, m = (2, 3) if mimo else (1, 1)
    E = np.diag([1., 2., 3.])
    A = E @ np.diag([.2, .5, .7] if sampling_time else [-.5, -2., -5.])
    B, C, D = rng.normal(size=(3, m)), rng.normal(size=(p, 3)), rng.normal(size=(p, m))
    fom = LTIModel.from_matrices(A, B, C, D=D, E=E, sampling_time=sampling_time)
    left, right = 1j * np.array([2., .2, .7]), 1j * np.array([1.5, .4, 2.5])
    if sampling_time:
        left, right = np.exp(left), np.exp(right)
    if shared_nodes:
        right = left.conj()[::-1]
    reductor = QuadBTReductor.from_model(fom if model_input else fom.transfer_function, left, right,
                                        feedthrough=None if model_input else D)
    assert reductor.sampling_time == sampling_time
    assert np.array_equal(reductor.feedthrough, D)
    assert len(reductor.left_nodes) == len(reductor.right_nodes) == 6
    tf = fom.transfer_function
    derivatives = np.array([tf.eval_dtf(z) for z in left]) if shared_nodes else None
    reference = QuadBTReductor(left, right, np.array([tf.eval_tf(z) for z in left]),
                               np.array([tf.eval_tf(z) for z in right]), derivatives=derivatives,
                               sampling_time=sampling_time, feedthrough=D, conjugate=True)
    for actual, expected in zip(reductor.quadrature_matrices(), reference.quadrature_matrices(), strict=True):
        assert np.allclose(actual, expected)
    rom = reductor.reduce(3)
    assert rom.sampling_time == sampling_time
    for z in [0, 1.2 + .3j, 2j]:
        assert np.allclose(rom.transfer_function.eval_tf(z), tf.eval_tf(z))


def test_quadbt_continuous_trapezoid_ordering():
    left = 1j * np.array([2, -1, -2, 1, 0])
    right = 1j * np.array([3, -.5, -3, .5])
    reductor = QuadBTReductor(left, right, 1 / (left + 1), 1 / (right + 1))
    assert np.array_equal(reductor.left_nodes, left)
    assert np.array_equal(reductor.right_nodes, right)
    assert np.allclose(reductor.left_weights, np.array([.5, 1, .5, 1, 1]) / (2 * np.pi))
    assert np.allclose(reductor.right_weights, np.array([1.25, 1.75, 1.25, 1.75]) / (2 * np.pi))
    rom = reductor.reduce(1)
    assert np.allclose(rom.transfer_function.eval_tf(3j), 1 / (3j + 1))


def test_quadbt_continuous_trapezoid_completion():
    left, right = 1j * np.array([10, 1]), 1j * np.array([20, 2])
    reductor = QuadBTReductor(left, right, 1 / (left + 1), 1 / (right + 1), conjugate=True)
    assert np.array_equal(reductor.left_nodes, [10j, 1j, -10j, -1j])
    assert np.allclose(reductor.left_weights, np.array([4.5, 5.5, 4.5, 5.5]) / (2 * np.pi))
    assert np.allclose(reductor.right_weights, 2 * reductor.left_weights)


@pytest.mark.parametrize('sampling_time', [.01, .8])
def test_quadbt_periodic_trapezoid(sampling_time):
    tf = TransferFunction(1, 1, lambda z: np.array([[1 / (z - .5)]]), sampling_time=sampling_time)
    left = np.exp(2j * np.pi * np.arange(8) / 8)
    right = np.exp(2j * np.pi * (np.arange(16) + .5) / 16)
    reductor = QuadBTReductor.from_model(tf, left, right)
    assert len(reductor.left_nodes) == 8
    assert len(reductor.right_nodes) == 16
    assert reductor.left_nodes[0] == 1
    assert reductor.left_nodes[4] == -1
    assert np.allclose(reductor.left_weights, 1 / 8)
    assert np.allclose(reductor.right_weights, 1 / 16)
    # An already complete grid also works without requesting conjugate completion.
    raw = QuadBTReductor(left, right, 1 / (left - .5), 1 / (right - .5), sampling_time=sampling_time)
    assert np.array_equal(raw.left_nodes, reductor.left_nodes)
    assert np.array_equal(raw.right_nodes, reductor.right_nodes)
    for actual, expected in zip(raw.quadrature_matrices(), reductor.quadrature_matrices(), strict=True):
        assert np.allclose(actual, expected)
    rom = reductor.reduce(1)
    assert rom.sampling_time == sampling_time
    assert np.allclose(rom.transfer_function.eval_tf(1j), tf.eval_tf(1j))


def test_quadbt_nonuniform_periodic_trapezoid():
    nodes = np.exp(1j * np.array([0, np.pi / 3, np.pi, -np.pi / 3]))
    values = 1 / (nodes - .5)
    reductor = QuadBTReductor(nodes, nodes, values, values, derivatives=-values**2,
                             sampling_time=.1, conjugate=True)
    assert len(reductor.left_nodes) == 4
    assert np.allclose(reductor.left_weights, [1 / 6, 1 / 4, 1 / 3, 1 / 4])
    assert np.allclose(reductor.right_weights, reductor.left_weights)
    assert np.allclose(reductor.reduce(1).transfer_function.eval_tf(1j), 1 / (1j - .5))


def test_quadbt_completion_preserves_supplied_data():
    left, right = np.array([1j, 2j]), np.array([-1j, -3j])
    reductor = QuadBTReductor(left, right, 1 / (left + 1), 1 / (right + 1),
                             [.2, .5], [.8, .4], derivatives=-1 / (left + 1)**2, conjugate=True)
    assert np.array_equal(reductor.left_weights, [.2, .5, .2, .5])
    assert np.array_equal(reductor.right_weights, [.8, .4, .8, .4])
    assert np.allclose(reductor.left_values[:, 0, 0], 1 / (reductor.left_nodes + 1))
    assert np.allclose(reductor.right_values[:, 0, 0], 1 / (reductor.right_nodes + 1))
    assert np.allclose(reductor.derivatives[:, 0, 0], -1 / (reductor.left_nodes + 1)**2)
    assert np.allclose(reductor.reduce(1).transfer_function.eval_tf(0), 1)


def test_quadbt_samples_only_needed_derivatives():
    calls = []

    def derivative(z):
        calls.append(z)
        return np.array([[-1 / (z + 1)**2]])

    tf = TransferFunction(1, 1, lambda z: np.array([[1 / (z + 1)]]), dtf=derivative)
    reductor = QuadBTReductor.from_model(tf, [1j, 2j], [-1j, 3j])
    assert calls == [1j]
    assert np.allclose(reductor.reduce(1).transfer_function.eval_tf(0), 1)


def test_quadbt_from_model_explicit_weights():
    tf = TransferFunction(1, 1, lambda z: np.array([[1 / (z + 1)]]))
    reductor = QuadBTReductor.from_model(tf, [1j, 2j], [3j, 4j],
                                        left_weights=[.2, .5], right_weights=[.8, .4])
    assert np.array_equal(reductor.left_weights, [.2, .5, .2, .5])
    assert np.array_equal(reductor.right_weights, [.8, .4, .8, .4])
    assert np.allclose(reductor.reduce(1).transfer_function.eval_tf(0), 1)


def test_quadbt_from_model_missing_derivatives():
    tf = TransferFunction(1, 1, lambda z: np.array([[1 / (z + 1)]]))
    with pytest.raises(ValueError, match='require transfer function derivatives'):
        QuadBTReductor.from_model(tf, [1j, 2j], [-1j, 3j])
    reductor = QuadBTReductor.from_model(tf, [1j, 2j], [-1j, 3j],
                                        derivatives=-1 / (np.array([1j, 2j]) + 1)**2)
    assert np.allclose(reductor.reduce(1).transfer_function.eval_tf(0), 1)


@pytest.mark.parametrize(('nodes', 'sampling_time', 'conjugate', 'message'), [
    ([1j], 0, False, 'at least two imaginary-axis nodes'),
    ([1 + 1j, 1 - 1j], 0, False, 'imaginary-axis nodes'),
    ([2j, -2j], .1, False, 'unit-circle nodes'),
    ([1, np.exp(2j * np.pi)], .1, True, 'distinct'),
    ([1, np.exp(2j * np.pi)], .1, False, 'do not repeat the endpoint'),
])
def test_quadbt_invalid_trapezoid_nodes(nodes, sampling_time, conjugate, message):
    with pytest.raises(ValueError, match=message):
        QuadBTReductor(nodes, nodes, np.ones(len(nodes)), np.ones(len(nodes)),
                       sampling_time=sampling_time, conjugate=conjugate, real=False)


def test_quadbt_from_model_complex_system():
    tf = TransferFunction(1, 1, lambda z: np.array([[1 / (z + 1 + 1j) + 2j]]))
    reductor = QuadBTReductor.from_model(tf, [1j, 2j], [3j, 4j],
                                        conjugate=False, real=False, feedthrough=2j)
    assert len(reductor.left_nodes) == 2
    assert np.allclose(reductor.reduce(1).transfer_function.eval_tf(0), tf.eval_tf(0))


def test_quadbt_from_model_invalid_input():
    with pytest.raises(TypeError, match='fom must be'):
        QuadBTReductor.from_model(np.ones(2), [1j], [2j])
    tf = TransferFunction(1, 1, lambda z, mu: np.array([[1 / (z + mu['a'][0])]]), parameters={'a': 1})
    with pytest.raises(ValueError, match='nonparametric'):
        QuadBTReductor.from_model(tf, [1j], [2j])
