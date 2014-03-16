# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from cPickle import dumps, loads

import numpy as np
import pytest

from pymortests.fixtures.function import function, function_argument
from pymortests.fixtures.parameter import parameter_of_type


# monkey np.testing.assert_allclose to behave the same as np.allclose
# for some reason, the default atol of np.testing.assert_allclose is 0
# while it is 1e-8 for np.allclose

real_assert_allclose = np.testing.assert_allclose


def monkey_allclose(a, b, rtol=1.e-5, atol=1.e-8):
    real_assert_allclose(a, b, rtol=rtol, atol=atol)
np.testing.assert_allclose = monkey_allclose


def test_evaluate(function):
    f = function
    for count in [0, 1, 5, (0, 1), (2, 2, 2)]:
        arg = function_argument(f, count, 454)
        result = f.evaluate(arg, parameter_of_type(f.parameter_type, 4711))
        assert result.shape == arg.shape[:-1] + f.shape_range


def test_pickle(function):
    f = function
    f2 = loads(dumps(f, -1))
    arg = function_argument(f, 1, 454)
    mu = parameter_of_type(f.parameter_type, 4711)
    np.testing.assert_allclose(f.evaluate(arg, mu), f2.evaluate(arg, mu))
