# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.analyticalproblems.functions import (
    ConstantFunction,
    ExpressionFunction,
    GenericFunction,
    TimeDiscreteFunction,
)
from pymor.core.cache import build_cache_key
from pymor.core.pickle import dumps, loads
from pymor.parameters.base import Mu
from pymortests.core.pickling import assert_picklable, assert_picklable_without_dumps_function

pytestmark = pytest.mark.builtin


def function_argument(f, count, rng):
    if isinstance(count, tuple):
        return rng.random(count + (f.dim_domain,))
    else:
        return rng.random((count, f.dim_domain))


def mu_of_type(parameters, rng):
    while True:
        if parameters is None:
            yield None
        else:
            yield Mu({k: rng.random(v) for k, v in parameters.items()})


def test_evaluate(function, rng):
    f = function
    mus = mu_of_type(f.parameters, rng)
    for count in [0, 1, 5, (0, 1), (2, 2, 2)]:
        arg = function_argument(f, count, rng)
        result = f.evaluate(arg, next(mus))
        assert result.shape == arg.shape[:-1] + f.shape_range


def test_lincomb_function():
    for steps in (1, 10):
        x = np.linspace(0, 1, num=steps)
        zero = ConstantFunction(0.0, dim_domain=steps)
        for zero in (ConstantFunction(0.0, dim_domain=steps),
                     GenericFunction(lambda X: np.zeros(X.shape[:-1]), dim_domain=steps)):
            for one in (ConstantFunction(1.0, dim_domain=steps),
                        GenericFunction(lambda X: np.ones(X.shape[:-1]), dim_domain=steps), 1.0):
                add = (zero + one) + 1 - 1
                add_ = 1 - 1 + (zero + one)
                sub = (zero - one) + np.zeros(())
                neg = - zero
                assert np.allclose(sub(x), [-1])
                assert np.allclose(add(x), [1.0])
                assert np.allclose(add_(x), [1.0])
                assert np.allclose(neg(x), [0.0])
                (repr(add), str(add), repr(one), str(one))  # just to cover the respective special funcs too
                mul = neg * 1. * 1.
                mul_ = 1. * 1. * neg
                assert np.allclose(mul(x), [0.0])
                assert np.allclose(mul_(x), [0.0])
        with pytest.raises(AssertionError):
            zero + ConstantFunction(dim_domain=steps + 1)
    with pytest.raises(AssertionError):
        ConstantFunction(dim_domain=0)


def test_pickle(function):
    assert_picklable(function)


def test_pickle_without_dumps_function(picklable_function):
    assert_picklable_without_dumps_function(picklable_function)


def test_pickle_by_evaluation(function, rng):
    f = function
    f2 = loads(dumps(f))
    mus = mu_of_type(f.parameters, rng)
    for arg in function_argument(f, 10, rng):
        mu = next(mus)
        assert np.all(f.evaluate(arg, mu) == f2.evaluate(arg, mu))


def test_invalid_expressions():
    with pytest.raises(TypeError):
        ExpressionFunction('(-1 < x[0]) and (x[0] < 1)', 1)


def test_random_bitmap_function():
    from pymor.analyticalproblems.functions import BitmapFunction
    f = BitmapFunction.random(shape=(10, 10))
    for i in range(9):
        for j in range(9):
            # all values are between 0 and 1
            assert 0 <= f([i / 10., j / 10.]) <= 1
            # neighboring cells have different values
            assert (f([i/10., j/10.]) != f([(i+1)/10., (j+1)/10.]))
            # values in same cell are equal
            assert (f([i/10., j/10.]) == f([(i+0.3)/10., (j+0.3)/10.]))


def test_time_discrete_function_linear():
    times = np.array([0., 1., 3.])
    values = np.array([[0., 1.], [2., 3.], [6., 11.]])
    f = TimeDiscreteFunction(times, values)
    assert f.dim_domain == 1
    assert f.shape_range == (2,)
    # values at the time points are reproduced
    assert np.allclose(f(times[:, np.newaxis]), values)
    # linear interpolation in between
    assert np.allclose(f(np.array([0.5])), [1., 2.])
    assert np.allclose(f(np.array([[2.], [2.5]])), [[4., 7.], [5., 9.]])


def test_time_discrete_function_constant():
    times = np.array([0., 1., 3.])
    values = np.array([10., 20., 30.])
    f = TimeDiscreteFunction(times, values, interpolation='constant')
    assert f.shape_range == ()
    t = np.array([0., 0.5, 1., 2.9, 3.])[:, np.newaxis]
    assert np.all(f(t) == [10., 10., 20., 20., 30.])


@pytest.mark.parametrize('interpolation', ['linear', 'constant'])
def test_time_discrete_function_round_off(interpolation):
    # time points computed by time steppers may be slightly off due to round-off errors
    nt = 7
    times = np.linspace(0, 0.7, nt + 1)
    values = np.arange(nt + 1.)
    f = TimeDiscreteFunction(times, values, interpolation=interpolation)
    t, dt = 0., 0.1
    for n in range(nt + 1):
        assert np.isclose(f(np.array([t])), n)
        t += dt


@pytest.mark.parametrize('interpolation', ['linear', 'constant'])
def test_time_discrete_function_outside_time_range(interpolation):
    f = TimeDiscreteFunction([1., 2.], [1., 2.], interpolation=interpolation)
    for t in [0.9, 2.1]:
        with pytest.raises(ValueError):
            f(np.array([t]))
    with pytest.raises(ValueError):
        f(np.array([[1.5], [2.5]]))


def test_time_discrete_function_invalid_arguments():
    with pytest.raises(AssertionError):
        TimeDiscreteFunction([0.], [1.])
    with pytest.raises(AssertionError):
        TimeDiscreteFunction([0., 1., 1.], [1., 2., 3.])
    with pytest.raises(AssertionError):
        TimeDiscreteFunction([0., 1.], [1., 2., 3.])
    with pytest.raises(AssertionError):
        TimeDiscreteFunction([0., 1.], [1., 2.], interpolation='cubic')


def test_time_discrete_function_cache_key():
    f = TimeDiscreteFunction([0., 1.], [[0.], [1.]])
    assert build_cache_key(f) == build_cache_key(TimeDiscreteFunction([0., 1.], [[0.], [1.]]))
    assert build_cache_key(f) != build_cache_key(TimeDiscreteFunction([0., 1.], [[0.], [2.]]))
    assert build_cache_key(f) != build_cache_key(f.with_(interpolation='constant'))
