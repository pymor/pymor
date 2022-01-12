# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

import pytest

from pymor.analyticalproblems.functions import ConstantFunction, GenericFunction, ExpressionFunction


constant_functions = [
    ConstantFunction(),
    ConstantFunction(np.array([1., 2., 3.]), dim_domain=7),
    ConstantFunction(np.eye(27), dim_domain=2),
    ConstantFunction(np.array(3), dim_domain=1),
]


def importable_function(x):
    return (x[..., 0] * x[..., 1])[..., np.newaxis]


class A:

    @staticmethod
    def unimportable_function(x):
        return np.max(x, axis=-1)


def get_function_with_closure(y):

    def function_with_closure(x):
        return np.concatenate((x + y, x - y), axis=-1)

    return function_with_closure


generic_functions = [
    GenericFunction(lambda x: x, dim_domain=2, shape_range=(2,)),
    GenericFunction(lambda x, mu: mu['c'][0]*x, dim_domain=1, shape_range=(1,), parameters={'c': 1}),
    GenericFunction(A.unimportable_function, dim_domain=7, shape_range=()),
    GenericFunction(get_function_with_closure(42), dim_domain=1, shape_range=(2,)),
]


picklable_generic_functions = [
    GenericFunction(importable_function, dim_domain=3, shape_range=(1,)),
]

expression_functions = [
    ExpressionFunction('x', dim_domain=2),
    ExpressionFunction("c[0]*x", dim_domain=1, parameters={'c': 1}),
    ExpressionFunction("c[2]*sin(x)", dim_domain=1, parameters={'c': 3}),
]


@pytest.fixture(params=constant_functions + generic_functions + picklable_generic_functions + expression_functions)
def function(request):
    return request.param


@pytest.fixture(params=constant_functions + picklable_generic_functions + expression_functions)
def picklable_function(request):
    return request.param


def function_argument(f, count, seed):
    np.random.seed(seed)
    if isinstance(count, tuple):
        return np.random.random(count + (f.dim_domain,))
    else:
        return np.random.random((count, f.dim_domain))
