# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from functools import partial
from numbers import Number
import itertools
import numpy as np
import pytest

from hypothesis import given

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.timestepping import (
        ImplicitEulerTimeStepper, ExplicitEulerTimeStepper, ExplicitRungeKuttaTimeStepper)
from pymor.analyticalproblems.burgers import burgers_problem
from pymor.discretizers.builtin.cg import discretize_instationary_cg, discretize_stationary_cg
from pymor.discretizers.builtin.fv import discretize_instationary_fv
from pymor.tools.floatcmp import almost_less

from pymortests.base import runmodule
from pymortests.fixtures.analyticalproblem import linear_transport_problems, parabolic_problems


def assert_all_iterator_variants_work(fom, mu=None, expected_len=None, additional_check=lambda U: True):
    solve = partial(fom.time_stepper.solve,
                    initial_time=0., end_time=fom.T, initial_data=fom.initial_data, operator=fom.operator, rhs=fom.rhs,
                    mass=fom.mass, mu=mu)
    # variant returning an iterator yielding U(t)
    U_iter = fom.solution_space.empty()
    for U in solve(return_iter=True, return_times=False):
        U_iter.append(U)
    if expected_len:
        assert len(U_iter) == expected_len
    assert additional_check(U_iter)
    # variant returning an iterator yielding U(t), t
    U_iter_times = fom.solution_space.empty()
    times_iter = []
    for U, t in solve(return_iter=True, return_times=True):
        assert isinstance(t, Number)
        assert almost_less(0., t)
        assert almost_less(t, fom.T)
        U_iter_times.append(U)
        times_iter.append(t)
    if expected_len:
        assert len(U_iter_times) == expected_len
    assert additional_check(U_iter_times)
    # variant returning the full trajectory U(0), ..., U(T)
    U_full = solve(return_iter=False, return_times=False)
    if expected_len:
        assert len(U_full) == expected_len
    assert additional_check(U_full)
    # variant returning the full trajectory and time points U, times
    U_full_times, times_full = solve(return_iter=False, return_times=True)
    if expected_len:
        assert len(U_full_times) == expected_len
    assert additional_check(U_full_times)
    assert np.allclose(times_iter, times_full)


@pytest.mark.parametrize('num_values_factor', [None, 0.3, 1, 17])
def test_ExplicitEuler_linear_transport_exact(num_values_factor):
    nt = 10
    fom, _ = discretize_instationary_fv(linear_transport_problems[0], diameter=1/nt, nt=nt)
    if num_values_factor:
        num_values = num_values_factor*nt + 1
        fom = fom.with_time_stepper(num_values=num_values)
    else:
        num_values = nt + 1

    assert_all_iterator_variants_work(
            fom, mu=None, expected_len=num_values, additional_check=lambda U: almost_equal(U[0], U[-1]))


@pytest.mark.parametrize('num_values_factor', [None, 0.3, 1, 17])
def test_ImplicitEuler_parabolic_equilibrium(num_values_factor):
    nt = 10
    fom, _ = discretize_instationary_cg(parabolic_problems[0], nt=nt)
    equi_fom, _ = discretize_stationary_cg(parabolic_problems[0].stationary_part)
    if num_values_factor:
        num_values = num_values_factor*nt + 1
        fom = fom.with_time_stepper(num_values=num_values)
    else:
        num_values = nt + 1

    assert_all_iterator_variants_work(fom, mu=None, expected_len=num_values,
            additional_check=lambda U: fom.h1_0_norm(U[-1] - equi_fom.solve()) < 1e-10)


@pytest.mark.parametrize('method_and_num_values_factor',
        itertools.product(ExplicitRungeKuttaTimeStepper.available_RK_methods.keys(),
                          [None, 0.3, 1, 17]))
def test_ExplicitRungeKutta_burgers_similar_to_ExplicitEuler(method_and_num_values_factor):
    # expected results are for burgers_problem, diameter=1/nt, nt=nt, mu=2
    expected_linf_error = {'explicit_euler': 0,
                           'RK1': 0,
                           'heun2': 0.11413868866639532,
                           'midpoint': 0.11351891541172932,
                           'ralston': 0.11372752576416811,
                           'RK2': 0.11351891541172932,
                           'simpson': 0.11211746195498029,
                           'heun3': 0.1121689990087954,
                           'RK3': 0.11211746195498029,
                           '3/8': 0.11145513797445739,
                           'RK4': 0.11145685666230609}
    method, num_values_factor = method_and_num_values_factor
    assert method in expected_linf_error, 'Missing expected results!'
    p = burgers_problem()
    nt = 10
    mu = p.parameters.parse(2)
    U_ref = discretize_instationary_fv(p, diameter=1/nt, nt=nt)[0].solve(mu=mu)[-1]
    fom, _ = discretize_instationary_fv(
            p, diameter=1/nt, time_stepper=ExplicitRungeKuttaTimeStepper(method=method, nt=nt))
    if num_values_factor:
        num_values = num_values_factor*nt + 1
        fom = fom.with_time_stepper(num_values=num_values)
    else:
        num_values = nt + 1

    assert_all_iterator_variants_work(fom, mu=mu, expected_len=num_values,
            additional_check=lambda U: almost_less((U_ref - U[-1]).sup_norm()[0], expected_linf_error[method]))


if __name__ == "__main__":
    runmodule(filename=__file__)
