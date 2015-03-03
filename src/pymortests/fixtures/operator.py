# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorArray


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


def thermalblock_assemble_factory(xblocks, yblocks, diameter, seed):
    op, mu, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, seed)
    return op.assemble(mu), None, U, V, sp, rp


def thermalblock_concatenation_factory(xblocks, yblocks, diameter, seed):
    from pymor.operators.constructions import Concatenation
    op, mu, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, seed)
    op = Concatenation(sp, op)
    return op, mu, U, V, sp, rp


def thermalblock_identity_factory(xblocks, yblocks, diameter, seed):
    from pymor.operators.constructions import IdentityOperator
    _, _, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, seed)
    return IdentityOperator(U.space), None, U, V, sp, rp


def thermalblock_vectorarray_factory(transposed, xblocks, yblocks, diameter, seed):
    from pymor.operators.constructions import VectorArrayOperator
    _, _, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, seed)
    op = VectorArrayOperator(U, transposed)
    if transposed:
        U = V
        V = NumpyVectorArray(np.random.random((7, op.range.dim)), copy=False)
        sp = rp
        rp = NumpyMatrixOperator(np.eye(op.range.dim) * 2)
    else:
        U = NumpyVectorArray(np.random.random((7, op.source.dim)), copy=False)
        sp = NumpyMatrixOperator(np.eye(op.source.dim) * 2)
    return op, None, U, V, sp, rp


def thermalblock_vector_factory(xblocks, yblocks, diameter, seed):
    from pymor.operators.constructions import VectorOperator
    _, _, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, seed)
    op = VectorOperator(U.copy(ind=0))
    U = NumpyVectorArray(np.random.random((7, 1)), copy=False)
    sp = NumpyMatrixOperator(np.eye(1) * 2)
    return op, None, U, V, sp, rp


def thermalblock_vectorfunc_factory(product, xblocks, yblocks, diameter, seed):
    from pymor.operators.constructions import VectorFunctional
    _, _, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, seed)
    op = VectorFunctional(U.copy(ind=0), product=sp if product else None)
    U = V
    V = NumpyVectorArray(np.random.random((7, 1)), copy=False)
    sp = rp
    rp = NumpyMatrixOperator(np.eye(1) * 2)
    return op, None, U, V, sp, rp


def thermalblock_fixedparam_factory(xblocks, yblocks, diameter, seed):
    from pymor.operators.constructions import FixedParameterOperator
    op, mu, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, seed)
    return FixedParameterOperator(op, mu=mu), None, U, V, sp, rp


thermalblock_factory_arguments = \
    [(2, 2, 1./2., 333),
     (1, 1, 1./4., 444)]


thermalblock_operator_generators = \
    [lambda args=args: thermalblock_factory(*args)[0:2] for args in thermalblock_factory_arguments]


thermalblock_operator_with_arrays_generators = \
    [lambda args=args: thermalblock_factory(*args)[0:4] for args in thermalblock_factory_arguments]


thermalblock_operator_with_arrays_and_products_generators = \
    [lambda args=args: thermalblock_factory(*args) for args in thermalblock_factory_arguments]


thermalblock_assemble_operator_generators = \
    [lambda args=args: thermalblock_assemble_factory(*args)[0:2] for args in thermalblock_factory_arguments]


thermalblock_assemble_operator_with_arrays_generators = \
    [lambda args=args: thermalblock_assemble_factory(*args)[0:4] for args in thermalblock_factory_arguments]


thermalblock_assemble_operator_with_arrays_and_products_generators = \
    [lambda args=args: thermalblock_assemble_factory(*args) for args in thermalblock_factory_arguments]


thermalblock_concatenation_operator_generators = \
    [lambda args=args: thermalblock_concatenation_factory(*args)[0:2] for args in thermalblock_factory_arguments]


thermalblock_concatenation_operator_with_arrays_generators = \
    [lambda args=args: thermalblock_concatenation_factory(*args)[0:4] for args in thermalblock_factory_arguments]


thermalblock_concatenation_operator_with_arrays_and_products_generators = \
    [lambda args=args: thermalblock_concatenation_factory(*args) for args in thermalblock_factory_arguments]


thermalblock_identity_operator_generators = \
    [lambda args=args: thermalblock_identity_factory(*args)[0:2] for args in thermalblock_factory_arguments]


thermalblock_identity_operator_with_arrays_generators = \
    [lambda args=args: thermalblock_identity_factory(*args)[0:4] for args in thermalblock_factory_arguments]


thermalblock_identity_operator_with_arrays_and_products_generators = \
    [lambda args=args: thermalblock_identity_factory(*args) for args in thermalblock_factory_arguments]


thermalblock_vectorarray_operator_generators = \
    [lambda args=args: thermalblock_vectorarray_factory(False, *args)[0:2] for args in thermalblock_factory_arguments] + \
    [lambda args=args: thermalblock_vectorarray_factory(True, *args)[0:2] for args in thermalblock_factory_arguments]


thermalblock_vectorarray_operator_with_arrays_generators = \
    [lambda args=args: thermalblock_vectorarray_factory(False, *args)[0:4] for args in thermalblock_factory_arguments] + \
    [lambda args=args: thermalblock_vectorarray_factory(True, *args)[0:4] for args in thermalblock_factory_arguments]


thermalblock_vectorarray_operator_with_arrays_and_products_generators = \
    [lambda args=args: thermalblock_vectorarray_factory(False, *args) for args in thermalblock_factory_arguments] + \
    [lambda args=args: thermalblock_vectorarray_factory(True, *args) for args in thermalblock_factory_arguments]


thermalblock_vector_operator_generators = \
    [lambda args=args: thermalblock_vector_factory(*args)[0:2] for args in thermalblock_factory_arguments]


thermalblock_vector_operator_with_arrays_generators = \
    [lambda args=args: thermalblock_vector_factory(*args)[0:4] for args in thermalblock_factory_arguments]


thermalblock_vector_operator_with_arrays_and_products_generators = \
    [lambda args=args: thermalblock_vector_factory(*args) for args in thermalblock_factory_arguments]


thermalblock_vectorfunc_operator_generators = \
    [lambda args=args: thermalblock_vectorfunc_factory(False, *args)[0:2] for args in thermalblock_factory_arguments] + \
    [lambda args=args: thermalblock_vectorfunc_factory(True, *args)[0:2] for args in thermalblock_factory_arguments]


thermalblock_vectorfunc_operator_with_arrays_generators = \
    [lambda args=args: thermalblock_vectorfunc_factory(False, *args)[0:4] for args in thermalblock_factory_arguments] + \
    [lambda args=args: thermalblock_vectorfunc_factory(True, *args)[0:4] for args in thermalblock_factory_arguments]


thermalblock_vectorfunc_operator_with_arrays_and_products_generators = \
    [lambda args=args: thermalblock_vectorfunc_factory(False, *args) for args in thermalblock_factory_arguments] + \
    [lambda args=args: thermalblock_vectorfunc_factory(True, *args) for args in thermalblock_factory_arguments]


thermalblock_fixedparam_operator_generators = \
    [lambda args=args: thermalblock_fixedparam_factory(*args)[0:2] for args in thermalblock_factory_arguments]


thermalblock_fixedparam_operator_with_arrays_generators = \
    [lambda args=args: thermalblock_fixedparam_factory(*args)[0:4] for args in thermalblock_factory_arguments]


thermalblock_fixedparam_operator_with_arrays_and_products_generators = \
    [lambda args=args: thermalblock_fixedparam_factory(*args) for args in thermalblock_factory_arguments]


@pytest.fixture(params=thermalblock_operator_with_arrays_and_products_generators +
                       thermalblock_assemble_operator_with_arrays_and_products_generators +
                       thermalblock_concatenation_operator_with_arrays_and_products_generators +
                       thermalblock_identity_operator_with_arrays_and_products_generators +
                       thermalblock_vectorarray_operator_with_arrays_and_products_generators +
                       thermalblock_vector_operator_with_arrays_and_products_generators +
                       thermalblock_vectorfunc_operator_with_arrays_and_products_generators +
                       thermalblock_fixedparam_operator_with_arrays_and_products_generators)
def operator_with_arrays_and_products(request):
    return request.param()


@pytest.fixture(params=numpy_matrix_operator_with_arrays_generators +
                       thermalblock_operator_with_arrays_generators +
                       thermalblock_assemble_operator_with_arrays_generators +
                       thermalblock_concatenation_operator_with_arrays_generators +
                       thermalblock_identity_operator_with_arrays_generators +
                       thermalblock_vectorarray_operator_with_arrays_generators +
                       thermalblock_vector_operator_with_arrays_generators +
                       thermalblock_vectorfunc_operator_with_arrays_generators +
                       thermalblock_fixedparam_operator_with_arrays_generators)
def operator_with_arrays(request):
    return request.param()


@pytest.fixture(params=numpy_matrix_operator_generators +
                       thermalblock_operator_generators +
                       thermalblock_assemble_operator_generators +
                       thermalblock_concatenation_operator_generators +
                       thermalblock_identity_operator_generators +
                       thermalblock_vectorarray_operator_generators +
                       thermalblock_vector_operator_generators +
                       thermalblock_vectorfunc_operator_generators +
                       thermalblock_fixedparam_operator_generators)
def operator(request):
    return request.param()
