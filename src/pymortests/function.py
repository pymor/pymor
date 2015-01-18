# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from pymor.core.pickle import dumps, loads
from pymor.functions.basic import ConstantFunction, GenericFunction
from pymortests.fixtures.function import function, picklable_function, function_argument
from pymortests.fixtures.parameter import parameters_of_type
from pymortests.pickle import assert_picklable, assert_picklable_without_dumps_function


# monkey np.testing.assert_allclose to behave the same as np.allclose
# for some reason, the default atol of np.testing.assert_allclose is 0
# while it is 1e-8 for np.allclose

real_assert_allclose = np.testing.assert_allclose


def monkey_allclose(a, b, rtol=1.e-5, atol=1.e-8):
    real_assert_allclose(a, b, rtol=rtol, atol=atol)
np.testing.assert_allclose = monkey_allclose


def test_evaluate(function):
    f = function
    mus = parameters_of_type(f.parameter_type, 4711)
    for count in [0, 1, 5, (0, 1), (2, 2, 2)]:
        arg = function_argument(f, count, 454)
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
                add = (zero + one) + 0
                sub = (zero - one) + np.zeros(())
                neg = - zero
                assert np.allclose(sub(x), [-1])
                assert np.allclose(add(x), [1.0])
                assert np.allclose(neg(x), [0.0])
                (repr(add), str(add), repr(one), str(one))  # just to cover the respective special funcs too
                mul = neg * 1.
                assert np.allclose(mul(x), [0.0])
        with pytest.raises(AssertionError):
            zero + ConstantFunction(dim_domain=steps + 1)
        with pytest.raises(AssertionError):
            zero * ConstantFunction(dim_domain=steps)
    with pytest.raises(AssertionError):
        ConstantFunction(dim_domain=0)


def test_pickle(function):
    assert_picklable(function)


def test_pickle_without_dumps_function(picklable_function):
    assert_picklable_without_dumps_function(picklable_function)


def test_pickle_by_evaluation(function):
    f = function
    f2 = loads(dumps(f))
    mus = parameters_of_type(f.parameter_type, 47)
    for arg in function_argument(f, 10, 42):
        mu = next(mus)
        assert np.all(f.evaluate(arg, mu) == f2.evaluate(arg, mu))
