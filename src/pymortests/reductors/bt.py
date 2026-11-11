# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from math import factorial

import numpy as np
import pytest
import scipy.linalg as spla
from scipy.integrate import quad_vec

from pymor.models.iosys import LTIModel
from pymor.reductors.bt import (
    BRBTReductor,
    BSTReductor,
    BTReductor,
    FLBTReductor,
    FWBTReductor,
    HinfBTReductor,
    LQGBTReductor,
    TLBTReductor,
    TWBTReductor,
)

pytestmark = pytest.mark.builtin


def random_fom(rng, n=20, m=2, p=2, with_D=False, sampling_time=0):
    A = rng.standard_normal((n, n))
    if sampling_time == 0:
        A -= (np.max(spla.eigvals(A).real) + 1) * np.eye(n)
    else:
        A /= 1.1 * np.max(np.abs(spla.eigvals(A)))
    M = rng.standard_normal((n, n))
    E = np.eye(n) + 0.1 * M @ M.T / n
    B = rng.standard_normal((n, m))
    C = rng.standard_normal((p, n))
    D = rng.standard_normal((p, m)) + 3 * np.eye(p, m) if with_D else None
    return LTIModel.from_matrices(E @ A, E @ B, C, D, E, sampling_time=sampling_time)


def static_weight(dim, sampling_time=0):
    """Return a weight with transfer function equal to the identity."""
    A = -np.eye(1) if sampling_time == 0 else np.zeros((1, 1))
    return LTIModel.from_matrices(A, np.zeros((1, dim)), np.zeros((dim, 1)), np.eye(dim),
                                  sampling_time=sampling_time)


def sampled_hinf_norm(tfw):
    return np.max(np.linalg.norm(tfw, ord=2, axis=(1, 2)))


w = np.logspace(-3, 3, 500)


def hinf_bound(fom):
    """Upper bound for the H_inf-norm (for systems without feedthrough)."""
    return 2 * fom.hsv().sum()


reductor_list = [
    (BTReductor, {}, False),
    (LQGBTReductor, {}, False),
    (BRBTReductor, None, False),
    (BSTReductor, {}, True),
    (HinfBTReductor, {'gamma': 2}, False),
    (FWBTReductor, {'input_weight': LTIModel.from_matrices(-np.eye(1), np.ones((1, 2)), np.ones((2, 1)))}, False),
    (FWBTReductor, {'output_weight': LTIModel.from_matrices(-np.eye(1), np.ones((1, 2)), np.ones((2, 1)))}, False),
    (FLBTReductor, {'omega': (0.5, 4)}, False),
    (TLBTReductor, {'time_interval': (0.1, 2)}, False),
    (TWBTReductor, {'k': 1}, False),
]


@pytest.mark.parametrize(('reductor', 'kwargs', 'with_D'), reductor_list)
@pytest.mark.parametrize('projection', ['sr', 'bfsr', 'biorth'])
def test_bt_reductors(reductor, kwargs, with_D, projection, rng):
    fom = random_fom(rng, with_D=with_D)
    if reductor is BRBTReductor:
        kwargs = {'gamma': hinf_bound(fom)}
    red = reductor(fom, **kwargs)
    r = 5
    rom = red.reduce(r, projection=projection)
    assert isinstance(rom, LTIModel)
    assert rom.order == r
    assert red.V in fom.solution_space
    assert red.W in fom.solution_space
    assert len(red.V) == len(red.W) == r
    u = rom.solution_space.ones()
    assert red.reconstruct(u) in fom.solution_space


@pytest.mark.parametrize(('reductor', 'kwargs'), [
    (BTReductor, {}),
    (BRBTReductor, None),
])
def test_hinf_error_bounds(reductor, kwargs, rng):
    fom = random_fom(rng)
    if reductor is BRBTReductor:
        kwargs = {'gamma': hinf_bound(fom)}
    red = reductor(fom, **kwargs)
    error_bounds = red.error_bounds()
    assert np.all(np.diff(error_bounds) <= 0)
    fom_tfw = fom.transfer_function.freq_resp(w)
    for r in [2, 5, 10]:
        rom = red.reduce(r)
        err = sampled_hinf_norm(fom_tfw - rom.transfer_function.freq_resp(w))
        assert err <= error_bounds[r - 1]


@pytest.mark.parametrize(('m', 'p'), [(2, 2), (3, 2), (2, 3)])
def test_bst_error_bounds(m, p, rng):
    fom = random_fom(rng, m=m, p=p, with_D=True)
    red = BSTReductor(fom)
    sv = red._hankel_svd()[1]
    assert np.all(sv < 1)
    error_bounds = red.error_bounds()
    assert np.all(np.diff(error_bounds) <= 0)
    fom_tfw = fom.transfer_function.freq_resp(w)
    for r in [2, 5, 10]:
        rom = red.reduce(r)
        assert np.all(rom.poles().real < 0)
        err_tfw = fom_tfw - rom.transfer_function.freq_resp(w)
        if m >= p:
            rel_err_tfw = np.linalg.pinv(fom_tfw) @ err_tfw
        else:
            rel_err_tfw = err_tfw @ np.linalg.pinv(fom_tfw)
        assert sampled_hinf_norm(rel_err_tfw) <= error_bounds[r - 1]


@pytest.mark.parametrize(('reductor', 'kwargs', 'with_D'), [
    (BTReductor, {}, False),
    (LQGBTReductor, {}, False),
    (BSTReductor, {}, True),
    (HinfBTReductor, {'gamma': 2}, False),
])
def test_reduce_tol(reductor, kwargs, with_D, rng):
    fom = random_fom(rng, with_D=with_D)
    red = reductor(fom, **kwargs)
    error_bounds = red.error_bounds()
    tol = error_bounds[4]
    rom = red.reduce(tol=tol)
    assert rom.order <= 5
    assert error_bounds[rom.order - 1] <= tol


def test_hinfbt_lqgbt(rng):
    fom = random_fom(rng)
    sv_lqg = LQGBTReductor(fom)._hankel_svd()[1]
    sv_hinf = HinfBTReductor(fom, gamma=1e8)._hankel_svd()[1]
    assert np.allclose(sv_hinf, sv_lqg)
    sv_hinf = HinfBTReductor(fom, gamma=2)._hankel_svd()[1]
    assert np.all(sv_hinf[:5] > sv_lqg[:5])


@pytest.mark.parametrize(('reductor', 'kwargs'), [
    (FWBTReductor, {'input_weight': static_weight(2), 'output_weight': static_weight(2)}),
    (FLBTReductor, {'omega': (0, np.inf)}),
    (TLBTReductor, {'time_interval': (0, np.inf)}),
    (TWBTReductor, {'k': 0}),
])
def test_weighted_bt_limit(reductor, kwargs, rng):
    fom = random_fom(rng)
    sv_bt = BTReductor(fom)._hankel_svd()[1][:10]
    sv = reductor(fom, **kwargs)._hankel_svd()[1][:10]
    assert np.allclose(sv, sv_bt)


def dense_gramians(fom, f_c, f_o, a, b):
    A, B, C, _, E = fom.to_matrices(format='dense')
    A = spla.solve(E, A)
    B = spla.solve(E, B)
    P = quad_vec(lambda t: f_c(A, B, t), a, b, epsabs=1e-12)[0]
    Q = quad_vec(lambda t: f_o(A, C, t), a, b, epsabs=1e-12)[0]
    return P, Q, E


def check_gramians(red, P, Q, E):
    cf, of = red._gramians()
    cf = cf.to_numpy()
    of = E.T @ of.to_numpy()
    assert spla.norm(cf @ cf.T - P) <= 1e-6 * spla.norm(P)
    assert spla.norm(of @ of.T - Q) <= 1e-6 * spla.norm(Q)


def test_flbt_gramians(rng):
    fom = random_fom(rng, n=10)
    omega = (0.5, 4)

    def f_c(A, B, nu):
        R = spla.inv(1j * nu * np.eye(A.shape[0]) - A)
        return (R @ B @ B.T @ R.conj().T).real / np.pi

    def f_o(A, C, nu):
        R = spla.inv(1j * nu * np.eye(A.shape[0]) - A)
        return (R.conj().T @ C.T @ C @ R).real / np.pi

    P, Q, E = dense_gramians(fom, f_c, f_o, *omega)
    check_gramians(FLBTReductor(fom, omega), P, Q, E)


def test_tlbt_gramians(rng):
    fom = random_fom(rng, n=10)
    time_interval = (0.1, 2)

    def f_c(A, B, t):
        F = spla.expm(t * A)
        return F @ B @ B.T @ F.T

    def f_o(A, C, t):
        F = spla.expm(t * A)
        return F.T @ C.T @ C @ F

    P, Q, E = dense_gramians(fom, f_c, f_o, *time_interval)
    check_gramians(TLBTReductor(fom, time_interval), P, Q, E)


@pytest.mark.parametrize('k', [1, 2])
def test_twbt_gramians(k, rng):
    fom = random_fom(rng, n=10)

    def f_c(A, B, t):
        F = spla.expm(t * A)
        return t**k / factorial(k) * F @ B @ B.T @ F.T

    def f_o(A, C, t):
        F = spla.expm(t * A)
        return t**k / factorial(k) * F.T @ C.T @ C @ F

    P, Q, E = dense_gramians(fom, f_c, f_o, 0, 60)
    check_gramians(TWBTReductor(fom, k), P, Q, E)


def test_fwbt_gramians(rng):
    fom = random_fom(rng, n=10)
    wi = LTIModel.from_matrices(-np.eye(1), np.ones((1, 2)), np.ones((2, 1)), np.eye(2))
    wo = LTIModel.from_matrices(-2 * np.eye(1), np.ones((1, 2)), np.ones((2, 1)), np.eye(2))
    cf, of = FWBTReductor(fom, input_weight=wi, output_weight=wo)._gramians()
    cf_wi = (fom * wi).gramian('c_lr').to_numpy()[:10]
    of_wo = (wo * fom).gramian('o_lr').to_numpy()[1:]
    assert np.allclose(cf.to_numpy() @ cf.to_numpy().T, cf_wi @ cf_wi.T)
    assert np.allclose(of.to_numpy() @ of.to_numpy().T, of_wo @ of_wo.T)


@pytest.mark.parametrize(('reductor', 'kwargs'), [
    (BTReductor, {}),
    (LQGBTReductor, {}),
    (BRBTReductor, None),
    (FWBTReductor, {'input_weight': static_weight(2, sampling_time=0.1)}),
])
def test_discrete_time(reductor, kwargs, rng):
    fom = random_fom(rng, sampling_time=0.1)
    if reductor is BRBTReductor:
        kwargs = {'gamma': hinf_bound(fom)}
    red = reductor(fom, **kwargs)
    r = 5
    rom = red.reduce(r)
    assert rom.order == r
    assert rom.sampling_time == fom.sampling_time
    assert np.all(np.abs(rom.poles()) < 1)
    if reductor is BTReductor:
        wd = np.linspace(0, np.pi / fom.sampling_time, 500)
        err = sampled_hinf_norm(fom.transfer_function.freq_resp(wd) - rom.transfer_function.freq_resp(wd))
        assert err <= red.error_bounds()[r - 1]
    elif reductor in (LQGBTReductor, BRBTReductor):
        with pytest.raises(NotImplementedError):
            red.error_bounds()


@pytest.mark.parametrize(('reductor', 'kwargs'), [
    (BSTReductor, {}),
    (HinfBTReductor, {'gamma': 2}),
    (FLBTReductor, {'omega': (0, 1)}),
    (TLBTReductor, {'time_interval': (0, 1)}),
    (TWBTReductor, {}),
])
def test_discrete_time_not_implemented(reductor, kwargs, rng):
    fom = random_fom(rng, with_D=True, sampling_time=0.1)
    with pytest.raises(NotImplementedError):
        reductor(fom, **kwargs)
