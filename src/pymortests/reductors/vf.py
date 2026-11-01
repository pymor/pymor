# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.linalg as spla

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.examples import penzl_example
from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction
from pymor.reductors.vf import VectorFittingReductor, _cauchy_svd, h2_quadrature_rule

pytestmark = pytest.mark.builtin


def random_lti(rng, num_pairs, num_real, p, m, unstable=False, D=None):
    """Random real LTI model with simple poles and rank-one residues."""
    blocks = []
    for omega in np.geomspace(0.1, 10, num_pairs):
        blocks.append(np.array([[-0.1 * omega, omega], [-omega, -0.1 * omega]]))
    blocks.extend(-np.geomspace(0.2, 5, num_real)[:, np.newaxis, np.newaxis])
    A = spla.block_diag(*blocks)
    if unstable:
        A[0, 0] = A[1, 1] = 0.1
    n = len(A)
    return LTIModel.from_matrices(A, rng.standard_normal((n, m)), rng.standard_normal((p, n)), D)


def assert_poles_close(poles1, poles2, rtol=1e-8):
    assert len(poles1) == len(poles2)
    dist = np.abs(poles1[:, np.newaxis] - poles2) / np.abs(poles1)[:, np.newaxis]
    assert np.max(np.min(dist, axis=1)) < rtol
    assert np.max(np.min(dist, axis=0)) < rtol


def relative_error(rom, s, Hs):
    Hr = np.array([rom.transfer_function.eval_tf(x) for x in s])
    return spla.norm(Hr - Hs) / spla.norm(Hs)


def test_sampled_transfer_function():
    fom = TransferFunction(1, 1, lambda s: np.array([[1 / (s + 1)]]))

    s = 1j * np.arange(1, 3)
    Hs = VectorFittingReductor.generate_samples(s, fom)
    reductor = VectorFittingReductor(s, Hs, weights=np.array([2., 3.]))

    assert np.array_equal(reductor.s, [1j, 2j, -1j, -2j])
    assert np.allclose(reductor.Hs[:, 0, 0], [1 / (1 + 1j), 1 / (1 + 2j), 1 / (1 - 1j), 1 / (1 - 2j)])
    assert np.allclose(reductor.weights, [2., 3., 2., 3.])

    reductor = VectorFittingReductor(s, Hs[:, 0, 0], force_real=False)
    assert np.array_equal(reductor.s, s)
    assert reductor.Hs.shape == (2, 1, 1)


@pytest.mark.parametrize(('p', 'm'), [(1, 1), (2, 3)])
@pytest.mark.parametrize('relax', [True, False])
@pytest.mark.parametrize('fit_D', [False, True])
@pytest.mark.parametrize('force_real', [True, False])
def test_exact_recovery(rng, p, m, relax, fit_D, force_real):
    fom = random_lti(rng, 4, 2, p, m, D=rng.standard_normal((p, m)) if fit_D else None)
    s = 1j * np.logspace(-2, 2, 100)
    if not force_real:
        s = np.concatenate((s, s.conj()))
    Hs = VectorFittingReductor.generate_samples(s, fom)

    reductor = VectorFittingReductor(s, Hs, force_real=force_real)
    rom = reductor.reduce(fom.order, fit_D=fit_D, relax=relax)

    assert rom.order == fom.order
    assert relative_error(rom, s, Hs) < 1e-10
    assert_poles_close(reductor.poles, fom.poles())
    assert all(np.sum(spla.svdvals(residue) > 1e-8 * spla.norm(residue)) == 1 for residue in reductor.residues)
    if force_real:
        assert all(np.isrealobj(to_matrix(op, format='dense')) for op in (rom.A, rom.B, rom.C, rom.D))


def test_residue_rank(rng):
    fom = random_lti(rng, 6, 0, 3, 3)
    s = 1j * np.logspace(-2, 2, 100)
    Hs = VectorFittingReductor.generate_samples(s, fom)
    # perturb the data such that the full-rank residues do not have rank one
    Hs = Hs + 1e-3 * rng.standard_normal(Hs.shape)
    reductor = VectorFittingReductor(s, Hs)

    rom_full = reductor.reduce(8, residue_rank=None)
    error_full = relative_error(rom_full, reductor.s, reductor.Hs)
    rom_2 = reductor.reduce(8, residue_rank=2)
    error_2 = relative_error(rom_2, reductor.s, reductor.Hs)
    rom_1 = reductor.reduce(8)
    error_1 = relative_error(rom_1, reductor.s, reductor.Hs)

    assert rom_full.order == 24
    assert rom_2.order == 16
    assert rom_1.order == 8
    assert error_full <= error_2 <= error_1 < 1e-1


@pytest.mark.parametrize('initial_poles', ['log', 'lin', 'real', 'array'])
@pytest.mark.parametrize('r', [5, 6])
def test_initial_poles(initial_poles, r):
    fom = penzl_example()
    s = 1j * np.logspace(-1, 4, 100)
    Hs = VectorFittingReductor.generate_samples(s, fom)
    if initial_poles == 'array':
        pairs = [-10 + 50j, -10 - 50j, -10 + 150j, -10 - 150j, -10 + 500j, -10 - 500j]
        initial_poles = np.array(pairs if r == 6 else [-10, *pairs[:4]])
        r = None
    reductor = VectorFittingReductor(s, Hs)
    rom = reductor.reduce(r, initial_poles=initial_poles)
    assert rom.order == len(reductor.poles)
    assert np.all(rom.poles().real < 0)
    assert relative_error(rom, s, Hs) < 0.5


def test_maxit_zero():
    fom = penzl_example()
    s = 1j * np.logspace(-1, 4, 100)
    Hs = VectorFittingReductor.generate_samples(s, fom)
    reductor = VectorFittingReductor(s, Hs)
    poles = np.array([-1 + 100j, -1 - 100j])
    rom = reductor.reduce(initial_poles=poles, maxit=0)
    assert_poles_close(rom.poles(), poles)
    assert reductor.errors == []


@pytest.mark.parametrize('force_stability', [True, False])
def test_force_stability(rng, force_stability):
    fom = random_lti(rng, 3, 1, 1, 1, unstable=True)
    s = 1j * np.logspace(-2, 2, 100)
    Hs = VectorFittingReductor.generate_samples(s, fom)
    reductor = VectorFittingReductor(s, Hs)
    rom = reductor.reduce(fom.order, force_stability=force_stability)
    if force_stability:
        assert np.all(rom.poles().real < 0)
    else:
        assert relative_error(rom, s, Hs) < 1e-10
        assert_poles_close(reductor.poles, fom.poles())


def test_complex_data(rng):
    poles = np.array([-1 + 2j, -0.5 - 1j, -2 + 0.5j])
    residues = rng.standard_normal((3, 1, 1)) + 1j * rng.standard_normal((3, 1, 1))
    s = 1j * np.linspace(-5, 5, 101)
    Hs = np.tensordot(1 / (s[:, np.newaxis] - poles), residues, 1)
    reductor = VectorFittingReductor(s, Hs, force_real=False)
    rom = reductor.reduce(3)
    assert_poles_close(reductor.poles, poles)
    assert relative_error(rom, s, Hs) < 1e-10


def test_noise_level(rng):
    fom = penzl_example()
    s = 1j * np.logspace(-1, 4, 200)
    Hs = VectorFittingReductor.generate_samples(s, fom)
    noise = rng.standard_normal(Hs.shape) + 1j * rng.standard_normal(Hs.shape)
    Hs = Hs + 1e-2 * noise * spla.norm(Hs) / spla.norm(noise)
    reductor = VectorFittingReductor(s, Hs)
    rom = reductor.reduce(20, noise_level=2e-2)
    error = relative_error(rom, reductor.s, reductor.Hs)
    assert 1e-2 < error < 3e-2


@pytest.mark.parametrize('constant', [False, True])
def test_cauchy_svd(rng, constant):
    x = 1j * np.logspace(-3, 3, 60)
    y = -np.logspace(-3, 3, 20) / 10 + 1j * np.logspace(-3, 3, 20)
    w = rng.random(60) + 0.5
    U, sv, V = _cauchy_svd(x, y, w, constant)
    G = w[:, np.newaxis] / (x[:, np.newaxis] - y)
    if constant:
        G = np.hstack((G, w[:, np.newaxis]))
    assert np.allclose(U.conj().T @ U, np.eye(len(sv)))
    assert np.allclose(V.conj().T @ V, np.eye(len(sv)))
    assert np.all(np.diff(sv) <= 0)
    assert spla.norm(U * sv @ V.conj().T - G) <= 1e-13 * spla.norm(G)
    assert np.allclose(sv, spla.svdvals(G), rtol=1e-8, atol=1e-13 * sv[0])


def test_h2_quadrature_rule(rng):
    fom = random_lti(rng, 10, 5, 1, 1)
    nodes, weights = h2_quadrature_rule(400)
    assert np.array_equal(nodes, -nodes[::-1])
    assert np.array_equal(weights, weights[::-1])
    Hs = VectorFittingReductor.generate_samples(nodes, fom)
    assert np.isclose(np.sqrt(np.sum(weights * spla.norm(Hs, axis=(1, 2))**2)), fom.h2_norm(), rtol=1e-3)

    nodes, weights = h2_quadrature_rule(101)
    Hs = VectorFittingReductor.generate_samples(nodes, fom)
    reductor = VectorFittingReductor(nodes, Hs, weights=weights)
    assert np.array_equal(reductor.s, nodes)
    rom = reductor.reduce(10)
    assert (fom - rom).h2_norm() / fom.h2_norm() < 0.5
