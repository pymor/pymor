# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from pymor.la.numpyvectorarray import NumpyVectorArray
from pymor.operators.numpy import NumpyMatrixOperator


def random_integers(count, seed):
    np.random.seed(seed)
    return list(np.random.randint(0, 3200, count))


def numpy_matrix_operator_with_arrays_factory(dim_source, dim_range, count_source, count_range, seed):
    np.random.seed(seed)
    op = NumpyMatrixOperator(np.random.random((dim_range, dim_source)))
    s = NumpyVectorArray(np.random.random((count_source, dim_source)), copy=False)
    r = NumpyVectorArray(np.random.random((count_range, dim_range)), copy=False)
    return op, None, s, r


numpy_matrix_operator_with_arrays_factory_arguments = \
    zip([0, 0, 2, 10],           # dim_source
        [0, 1, 4, 10],           # dim_range
        [3, 3, 3, 3],            # count_source
        [3, 3, 3, 3],            # count_range
        random_integers(4, 44))  # seed


numpy_matrix_operator_with_arrays_generators = \
    [lambda args=args: numpy_matrix_operator_with_arrays_factory(*args)
     for args in numpy_matrix_operator_with_arrays_factory_arguments]


numpy_matrix_operator_generators = \
    [lambda args=args: numpy_matrix_operator_with_arrays_factory(*args)[0:2]
     for args in numpy_matrix_operator_with_arrays_factory_arguments]


def thermalblock_factory(xblocks, yblocks, diameter, seed):
    from pymor.analyticalproblems.thermalblock import ThermalBlockProblem
    from pymor.discretizers.elliptic import discretize_elliptic_cg
    from pymor.functions.basic import GenericFunction
    from pymor.operators.cg import InterpolationOperator
    p = ThermalBlockProblem((xblocks, yblocks))
    d, d_data = discretize_elliptic_cg(p, diameter)
    f = GenericFunction(lambda X, mu: X[..., 0]**mu['exp'] + X[..., 1],
                        dim_domain=2, parameter_type={'exp': tuple()})
    iop = InterpolationOperator(d_data['grid'], f)
    U = d.operator.source.empty()
    V = d.operator.range.empty()
    np.random.seed(seed)
    for exp in np.random.random(5):
        U.append(iop.as_vector(exp))
    for exp in np.random.random(6):
        V.append(iop.as_vector(exp))
    return d.operator, next(d.parameter_space.sample_randomly(1, seed=seed)), U, V, d.h1_product, d.l2_product


thermalblock_factory_arguments = \
    [(2, 2, 1./10., 333),
     (1, 1, 1./20., 444)]


thermalblock_operator_generators = \
    [lambda args=args: thermalblock_factory(*args)[0:2] for args in thermalblock_factory_arguments]


thermalblock_operator_with_arrays_generators = \
    [lambda args=args: thermalblock_factory(*args)[0:4] for args in thermalblock_factory_arguments]


thermalblock_operator_with_arrays_and_products_generators = \
    [lambda args=args: thermalblock_factory(*args) for args in thermalblock_factory_arguments]


@pytest.fixture(params=thermalblock_operator_with_arrays_and_products_generators)
def operator_with_arrays_and_products(request):
    return request.param()


@pytest.fixture(params=numpy_matrix_operator_with_arrays_generators +
                       thermalblock_operator_with_arrays_generators)
def operator_with_arrays(request):
    return request.param()


@pytest.fixture(params=numpy_matrix_operator_generators +
                       thermalblock_operator_generators)
def operator(request):
    return request.param()
