# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.analyticalproblems.functions import ConstantFunction, GenericFunction, ExpressionFunction
from pymor.core.pickle import dumps, loads
from pymortests.fixtures.function import function_argument
from pymortests.fixtures.parameter import mu_of_type
from pymortests.pickling import assert_picklable, assert_picklable_without_dumps_function


def test_evaluate(function):
    f = function
    mus = mu_of_type(f.parameters, 4711)
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


def test_pickle_by_evaluation(function):
    f = function
    f2 = loads(dumps(f))
    mus = mu_of_type(f.parameters, 47)
    for arg in function_argument(f, 10, 42):
        mu = next(mus)
        assert np.all(f.evaluate(arg, mu) == f2.evaluate(arg, mu))


def test_invalid_expressions():
    with pytest.raises(TypeError):
        ExpressionFunction('-1 < x[0] < 1', 1)
    with pytest.raises(TypeError):
        ExpressionFunction('(-1 < x[0]) and (x[0] < 1)', 1)
