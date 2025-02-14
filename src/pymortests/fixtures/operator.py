# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from importlib import import_module
from itertools import product

import numpy as np
import pytest
import scipy.sparse as sps
from numpy.polynomial.polynomial import Polynomial
from packaging.version import parse

from pymor.core.config import config
from pymor.operators.constructions import IdentityOperator
from pymor.operators.interface import Operator
from pymor.operators.list import NumpyListVectorArrayMatrixOperator
from pymor.operators.numpy import (
    NumpyCirculantOperator,
    NumpyHankelOperator,
    NumpyMatrixOperator,
    NumpyToeplitzOperator,
)
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.base import BUILTIN_DISABLED


class MonomOperator(Operator):
    source = range = NumpyVectorSpace(1)

    def __init__(self, order, monom=None):
        self.monom = monom if monom else Polynomial(np.identity(order + 1)[order])
        assert isinstance(self.monom, Polynomial)
        self.order = order
        self.derivative = self.monom.deriv()
        self.linear = order == 1

    def apply(self, U, mu=None):
        return self.source.make_array(self.monom(U.to_numpy_TP().T))

    def apply_adjoint(self, U, mu=None):
        return self.apply(U, mu=None)

    def jacobian(self, U, mu=None):
        assert len(U) == 1
        return NumpyMatrixOperator(self.derivative(U.to_numpy_TP().T).reshape((1, 1)))

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        return self.range.make_array(1. / V.to_numpy_TP().T)


def numpy_matrix_operator_with_arrays_factory(dim_source, dim_range, count_source, count_range, rng, sparse=False):
    assert not sparse or sparse in ('matrix', 'array')
    mat = rng.random((dim_range, dim_source))
    if sparse == 'matrix':
        mat = sps.csc_matrix(mat)
    elif sparse == 'array':
        mat = sps.csc_array(mat)
    op = NumpyMatrixOperator(rng.random((dim_range, dim_source)))
    s = op.source.make_array(rng.random((count_source, dim_source)))
    r = op.range.make_array(rng.random((count_range, dim_range)))
    return op, None, s, r


def numpy_list_vector_array_matrix_operator_with_arrays_factory(
    dim_source, dim_range, count_source, count_range, rng, sparse=False,
):
    op, _, s, r = numpy_matrix_operator_with_arrays_factory(
        dim_source, dim_range, count_source, count_range, rng, sparse
    )
    op = op.with_(new_type=NumpyListVectorArrayMatrixOperator)
    s = op.source.from_numpy(s.to_numpy_TP().T)
    r = op.range.from_numpy(r.to_numpy_TP().T)
    return op, None, s, r


def numpy_structured_matrix_operator_with_arrays_factory(structure, iscomplex, even, r_none, shape, blockshape,
                                                         count_source, count_range, rng):
    n = 5 if even else 4
    nr = 2 if shape == 'skinny' else n
    nc = 2 if shape == 'fat' else n
    blocks = (2 if blockshape == 'skinny' else 3, 2 if blockshape == 'fat' else 3)
    if blockshape == 'scalar':
        blocks = ()
    if iscomplex:
        c = rng.random((nc, *blocks)) + 1j * rng.random((nc, *blocks))
        r = rng.random((nr, *blocks)) + 1j * rng.random((nr, *blocks))
    else:
        c = rng.random((nc, *blocks))
        r = rng.random((nr, *blocks))
    if structure == NumpyCirculantOperator:
        op = structure(c)
    elif structure == NumpyToeplitzOperator:
        r[0] = c[0]
        op = structure(c, r=None if r_none else r)
    elif structure == NumpyHankelOperator:
        r[0] = c[-1]
        op = structure(c, r=None if r_none else r)
    U, V = op.source.random(count_source), op.range.random(count_range)
    return op, None, U, V


def numpy_matrix_operator_with_arrays_and_products_factory(dim_source, dim_range, count_source, count_range, rng):
    from scipy.linalg import eigh
    op, _, U, V = numpy_matrix_operator_with_arrays_factory(dim_source, dim_range, count_source, count_range, rng)
    if dim_source > 0:
        while True:
            sp = rng.random((dim_source, dim_source))
            sp = sp.T.dot(sp)
            evals = eigh(sp, eigvals_only=True)
            if np.min(evals) > 1e-6:
                break
        sp = NumpyMatrixOperator(sp)
    else:
        sp = NumpyMatrixOperator(np.zeros((0, 0)))
    if dim_range > 0:
        while True:
            rp = rng.random((dim_range, dim_range))
            rp = rp.T.dot(rp)
            evals = eigh(rp, eigvals_only=True)
            if np.min(evals) > 1e-6:
                break
        rp = NumpyMatrixOperator(rp)
    else:
        rp = NumpyMatrixOperator(np.zeros((0, 0)))
    return op, None, U, V, sp, rp

if parse(import_module('scipy').__version__) >= parse('1.8.0'):
    _sparse_opts = (False, 'matrix', 'array')
else:
    _sparse_opts = (False, 'matrix')

numpy_matrix_operator_with_arrays_factory_arguments = list(product(
    zip(
        [0, 0, 2, 10],       # dim_source
        [0, 1, 4, 10],       # dim_range
        [3, 3, 3, 3],        # count_source
        [3, 3, 3, 3],        # count_range
    ),
    [{'sparse': opt} for opt in _sparse_opts],
))

numpy_structured_matrix_operator_with_arrays_factory_arguments = list(product(
    [NumpyCirculantOperator, NumpyHankelOperator, NumpyToeplitzOperator],
    [False, True],
    [False, True],
    [False, True],
    ['fat', 'skinny', 'square'],
    ['fat', 'skinny', 'square'],
    [1, 3],
    [1, 3],
))

numpy_matrix_operator_with_arrays_generators = \
    [lambda rng, args=args, kwargs=kwargs: numpy_matrix_operator_with_arrays_factory(*args, rng=rng, **kwargs)
     for args, kwargs in numpy_matrix_operator_with_arrays_factory_arguments]


numpy_matrix_operator_generators = \
    [lambda rng, args=args, kwargs=kwargs: numpy_matrix_operator_with_arrays_factory(*args, rng=rng, **kwargs)[0:2]
     for args, kwargs in numpy_matrix_operator_with_arrays_factory_arguments]


numpy_list_vector_array_matrix_operator_with_arrays_generators = \
    [lambda rng, args=args, kwargs=kwargs: numpy_list_vector_array_matrix_operator_with_arrays_factory(
        *args, rng=rng, **kwargs)
     for args, kwargs in numpy_matrix_operator_with_arrays_factory_arguments]


numpy_list_vector_array_matrix_operator_generators = \
    [lambda rng, args=args, kwargs=kwargs: numpy_list_vector_array_matrix_operator_with_arrays_factory(
        *args, rng=rng, **kwargs)[0:2]
     for args, kwargs in numpy_matrix_operator_with_arrays_factory_arguments]


numpy_structured_matrix_operator_with_arrays_generators = \
    [lambda rng, args=args: numpy_structured_matrix_operator_with_arrays_factory(*args, rng=rng)
     for args in numpy_structured_matrix_operator_with_arrays_factory_arguments]


numpy_structured_matrix_operator_generators = \
    [lambda rng, args=args: numpy_structured_matrix_operator_with_arrays_factory(*args, rng=rng)[0:2]
     for args in numpy_structured_matrix_operator_with_arrays_factory_arguments]


def thermalblock_factory(xblocks, yblocks, diameter, rng):
    from pymor.analyticalproblems.functions import GenericFunction
    from pymor.analyticalproblems.thermalblock import thermal_block_problem
    from pymor.discretizers.builtin import discretize_stationary_cg
    from pymor.discretizers.builtin.cg import InterpolationOperator
    p = thermal_block_problem((xblocks, yblocks))
    m, m_data = discretize_stationary_cg(p, diameter)
    f = GenericFunction(lambda X, mu: X[..., 0]**mu['exp'][0] + X[..., 1],
                        dim_domain=2, parameters={'exp': 1})
    iop = InterpolationOperator(m_data['grid'], f)
    U = m.operator.source.empty()
    V = m.operator.range.empty()
    for exp in rng.random(5):
        U.append(iop.as_vector(f.parameters.parse(exp)))
    for exp in rng.random(6):
        V.append(iop.as_vector(f.parameters.parse(exp)))
    mu = p.parameter_space.sample_randomly(1)[0]
    return m.operator, mu, U, V, m.h1_product, m.l2_product


def thermalblock_assemble_factory(xblocks, yblocks, diameter, rng):
    op, mu, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, rng)
    return op.assemble(mu), None, U, V, sp, rp


def thermalblock_concatenation_factory(xblocks, yblocks, diameter, rng):
    op, mu, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, rng)
    op = sp @ op
    return op, mu, U, V, sp, rp


def thermalblock_identity_factory(xblocks, yblocks, diameter, rng):
    from pymor.operators.constructions import IdentityOperator
    _, _, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, rng)
    return IdentityOperator(U.space), None, U, V, sp, rp


def thermalblock_zero_factory(xblocks, yblocks, diameter, rng):
    from pymor.operators.constructions import ZeroOperator
    _, _, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, rng)
    return ZeroOperator(V.space, U.space), None, U, V, sp, rp


def thermalblock_constant_factory(xblocks, yblocks, diameter, rng):
    from pymor.operators.constructions import ConstantOperator
    _, _, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, rng)
    return ConstantOperator(V[0], U.space), None, U, V, sp, rp


def thermalblock_vectorarray_factory(adjoint, xblocks, yblocks, diameter, rng):
    from pymor.operators.constructions import VectorArrayOperator
    _, _, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, rng)
    op = VectorArrayOperator(U, adjoint)
    if adjoint:
        U = V
        V = op.range.make_array(rng.random((7, op.range.dim)))
        sp = rp
        rp = NumpyMatrixOperator(np.eye(op.range.dim) * 2)
    else:
        U = op.source.make_array(rng.random((7, op.source.dim)))
        sp = NumpyMatrixOperator(np.eye(op.source.dim) * 2)
    return op, None, U, V, sp, rp


def thermalblock_vector_factory(xblocks, yblocks, diameter, rng):
    from pymor.operators.constructions import VectorOperator
    _, _, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, rng)
    op = VectorOperator(U[0])
    U = op.source.make_array(rng.random((7, 1)))
    sp = NumpyMatrixOperator(np.eye(1) * 2)
    return op, None, U, V, sp, rp


def thermalblock_vectorfunc_factory(product, xblocks, yblocks, diameter, rng):
    from pymor.operators.constructions import VectorFunctional
    _, _, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, rng)
    op = VectorFunctional(U[0], product=sp if product else None)
    U = V
    V = op.range.make_array(rng.random((7, 1)))
    sp = rp
    rp = NumpyMatrixOperator(np.eye(1) * 2)
    return op, None, U, V, sp, rp


def thermalblock_fixedparam_factory(xblocks, yblocks, diameter, rng):
    from pymor.operators.constructions import FixedParameterOperator
    op, mu, U, V, sp, rp = thermalblock_factory(xblocks, yblocks, diameter, rng)
    return FixedParameterOperator(op, mu=mu), None, U, V, sp, rp


thermalblock_factory_arguments = \
    [(2, 2, 1./2.),
     (1, 1, 1./4.)]

thermalblock_operator_generators = \
    [lambda rng, args=args: thermalblock_factory(*args, rng=rng)[0:2]
     for args in thermalblock_factory_arguments]

thermalblock_operator_with_arrays_generators = \
    [lambda rng, args=args: thermalblock_factory(*args, rng=rng)[0:4]
     for args in thermalblock_factory_arguments]

thermalblock_operator_with_arrays_and_products_generators = \
    [lambda rng, args=args: thermalblock_factory(*args, rng=rng)
     for args in thermalblock_factory_arguments]

thermalblock_assemble_operator_generators = \
    [lambda rng, args=args: thermalblock_assemble_factory(*args, rng=rng)[0:2]
     for args in thermalblock_factory_arguments]

thermalblock_assemble_operator_with_arrays_generators = \
    [lambda rng, args=args: thermalblock_assemble_factory(*args, rng=rng)[0:4]
     for args in thermalblock_factory_arguments]

thermalblock_assemble_operator_with_arrays_and_products_generators = \
    [lambda rng, args=args: thermalblock_assemble_factory(*args, rng=rng)
     for args in thermalblock_factory_arguments]

thermalblock_concatenation_operator_generators = \
    [lambda rng, args=args: thermalblock_concatenation_factory(*args, rng=rng)[0:2]
     for args in thermalblock_factory_arguments]

thermalblock_concatenation_operator_with_arrays_generators = \
    [lambda rng, args=args: thermalblock_concatenation_factory(*args, rng=rng)[0:4]
     for args in thermalblock_factory_arguments]

thermalblock_concatenation_operator_with_arrays_and_products_generators = \
    [lambda rng, args=args: thermalblock_concatenation_factory(*args, rng=rng)
     for args in thermalblock_factory_arguments]

thermalblock_identity_operator_generators = \
    [lambda rng, args=args: thermalblock_identity_factory(*args, rng=rng)[0:2]
     for args in thermalblock_factory_arguments]

thermalblock_zero_operator_generators = \
    [lambda rng, args=args: thermalblock_zero_factory(*args, rng=rng)[0:2]
     for args in thermalblock_factory_arguments]


thermalblock_constant_operator_generators = \
    [lambda rng, args=args: thermalblock_constant_factory(*args, rng=rng)[0:2]
     for args in thermalblock_factory_arguments]


thermalblock_identity_operator_with_arrays_generators = \
    [lambda rng, args=args: thermalblock_identity_factory(*args, rng=rng)[0:4]
     for args in thermalblock_factory_arguments]

thermalblock_zero_operator_with_arrays_generators = \
    [lambda rng, args=args: thermalblock_zero_factory(*args, rng=rng)[0:4]
     for args in thermalblock_factory_arguments]

thermalblock_constant_operator_with_arrays_generators = \
    [lambda rng, args=args: thermalblock_constant_factory(*args, rng=rng)[0:4]
     for args in thermalblock_factory_arguments]

thermalblock_identity_operator_with_arrays_and_products_generators = \
    [lambda rng, args=args: thermalblock_identity_factory(*args, rng=rng)
     for args in thermalblock_factory_arguments]

thermalblock_zero_operator_with_arrays_and_products_generators = \
    [lambda rng, args=args: thermalblock_zero_factory(*args, rng=rng)
     for args in thermalblock_factory_arguments]

thermalblock_constant_operator_with_arrays_and_products_generators = \
    [lambda rng, args=args: thermalblock_constant_factory(*args, rng=rng)
     for args in thermalblock_factory_arguments]

thermalblock_vectorarray_operator_generators = \
    [lambda rng, args=args: thermalblock_vectorarray_factory(False, *args, rng=rng)[0:2]
     for args in thermalblock_factory_arguments] \
    + [lambda rng, args=args: thermalblock_vectorarray_factory(True, *args, rng=rng)[0:2]
       for args in thermalblock_factory_arguments]

thermalblock_vectorarray_operator_with_arrays_generators = \
    [lambda rng, args=args: thermalblock_vectorarray_factory(False, *args, rng=rng)[0:4]
     for args in thermalblock_factory_arguments] \
    + [lambda rng, args=args: thermalblock_vectorarray_factory(True, *args, rng=rng)[0:4]
       for args in thermalblock_factory_arguments]

thermalblock_vectorarray_operator_with_arrays_and_products_generators = \
    [lambda rng, args=args: thermalblock_vectorarray_factory(False, *args, rng=rng)
     for args in thermalblock_factory_arguments] \
    + [lambda rng, args=args: thermalblock_vectorarray_factory(True, *args, rng=rng)
       for args in thermalblock_factory_arguments]

thermalblock_vector_operator_generators = \
    [lambda rng, args=args: thermalblock_vector_factory(*args, rng=rng)[0:2]
     for args in thermalblock_factory_arguments]

thermalblock_vector_operator_with_arrays_generators = \
    [lambda rng, args=args: thermalblock_vector_factory(*args, rng=rng)[0:4]
     for args in thermalblock_factory_arguments]

thermalblock_vector_operator_with_arrays_and_products_generators = \
    [lambda rng, args=args: thermalblock_vector_factory(*args, rng=rng)
     for args in thermalblock_factory_arguments]

thermalblock_vectorfunc_operator_generators = \
    [lambda rng, args=args: thermalblock_vectorfunc_factory(False, *args, rng=rng)[0:2]
     for args in thermalblock_factory_arguments] \
    + [lambda rng, args=args: thermalblock_vectorfunc_factory(True, *args, rng=rng)[0:2]
       for args in thermalblock_factory_arguments]

thermalblock_vectorfunc_operator_with_arrays_generators = \
    [lambda rng, args=args: thermalblock_vectorfunc_factory(False, *args, rng=rng)[0:4]
     for args in thermalblock_factory_arguments] \
    + [lambda rng, args=args: thermalblock_vectorfunc_factory(True, *args, rng=rng)[0:4]
       for args in thermalblock_factory_arguments]

thermalblock_vectorfunc_operator_with_arrays_and_products_generators = \
    [lambda rng, args=args: thermalblock_vectorfunc_factory(False, *args, rng=rng)
     for args in thermalblock_factory_arguments] \
    + [lambda rng, args=args: thermalblock_vectorfunc_factory(True, *args, rng=rng)
       for args in thermalblock_factory_arguments]

thermalblock_fixedparam_operator_generators = \
    [lambda rng, args=args: thermalblock_fixedparam_factory(*args, rng=rng)[0:2]
     for args in thermalblock_factory_arguments]

thermalblock_fixedparam_operator_with_arrays_generators = \
    [lambda rng, args=args: thermalblock_fixedparam_factory(*args, rng=rng)[0:4]
     for args in thermalblock_factory_arguments]

thermalblock_fixedparam_operator_with_arrays_and_products_generators = \
    [lambda rng, args=args: thermalblock_fixedparam_factory(*args, rng=rng)
     for args in thermalblock_factory_arguments]


num_misc_operators = 13


def misc_operator_with_arrays_and_products_factory(n, rng):
    if n == 0:
        from pymor.operators.constructions import ComponentProjectionOperator
        _, _, U, V, sp, rp = numpy_matrix_operator_with_arrays_and_products_factory(100, 10, 4, 3, rng)
        op = ComponentProjectionOperator(rng.integers(0, 100, 10), U.space)
        return op, _, U, V, sp, rp
    elif n == 1:
        from pymor.operators.constructions import ComponentProjectionOperator
        _, _, U, V, sp, rp = numpy_matrix_operator_with_arrays_and_products_factory(100, 0, 4, 3, rng)
        op = ComponentProjectionOperator([], U.space)
        return op, _, U, V, sp, rp
    elif n == 2:
        from pymor.operators.constructions import ComponentProjectionOperator
        _, _, U, V, sp, rp = numpy_matrix_operator_with_arrays_and_products_factory(100, 3, 4, 3, rng)
        op = ComponentProjectionOperator([3, 3, 3], U.space)
        return op, _, U, V, sp, rp
    elif n == 3:
        from pymor.operators.constructions import AdjointOperator
        op, _, U, V, sp, rp = numpy_matrix_operator_with_arrays_and_products_factory(100, 20, 4, 3, rng)
        op = AdjointOperator(op, with_apply_inverse=True)
        return op, _, V, U, rp, sp
    elif n == 4:
        from pymor.operators.constructions import AdjointOperator
        op, _, U, V, sp, rp = numpy_matrix_operator_with_arrays_and_products_factory(100, 20, 4, 3, rng)
        op = AdjointOperator(op, with_apply_inverse=False)
        return op, _, V, U, rp, sp
    elif 5 <= n <= 7:
        from pymor.operators.constructions import SelectionOperator
        from pymor.parameters.functionals import ProjectionParameterFunctional
        op0, _, U, V, sp, rp = numpy_matrix_operator_with_arrays_and_products_factory(30, 30, 4, 3, rng)
        op1 = NumpyMatrixOperator(rng.random((30, 30)))
        op2 = NumpyMatrixOperator(rng.random((30, 30)))
        op = SelectionOperator([op0, op1, op2], ProjectionParameterFunctional('x'), [0.3, 0.6])
        return op, op.parameters.parse((n-5)/2), V, U, rp, sp
    elif n == 8:
        from pymor.operators.block import BlockDiagonalOperator
        op0, _, U0, V0, sp0, rp0 = numpy_matrix_operator_with_arrays_and_products_factory(10, 10, 4, 3, rng)
        op1, _, U1, V1, sp1, rp1 = numpy_matrix_operator_with_arrays_and_products_factory(20, 20, 4, 3, rng)
        op2, _, U2, V2, sp2, rp2 = numpy_matrix_operator_with_arrays_and_products_factory(30, 30, 4, 3, rng)
        op = BlockDiagonalOperator([op0, op1, op2])
        sp = BlockDiagonalOperator([sp0, sp1, sp2])
        rp = BlockDiagonalOperator([rp0, rp1, rp2])
        U = op.source.make_array([U0, U1, U2])
        V = op.range.make_array([V0, V1, V2])
        return op, _, U, V, sp, rp
    elif n == 9:
        from pymor.operators.block import BlockDiagonalOperator, BlockOperator
        from pymor.parameters.functionals import ProjectionParameterFunctional
        op0a, _, U0, V0, sp0, rp0 = numpy_matrix_operator_with_arrays_and_products_factory(10, 10, 4, 3, rng)
        op0b, _, _, _, _, _       = numpy_matrix_operator_with_arrays_and_products_factory(10, 10, 4, 3, rng)
        op0c, _, _, _, _, _       = numpy_matrix_operator_with_arrays_and_products_factory(10, 10, 4, 3, rng)
        op1, _, U1, V1, sp1, rp1  = numpy_matrix_operator_with_arrays_and_products_factory(20, 20, 4, 3, rng)
        op2a, _, _, _, _, _       = numpy_matrix_operator_with_arrays_and_products_factory(20, 10, 4, 3, rng)
        op2b, _, _, _, _, _       = numpy_matrix_operator_with_arrays_and_products_factory(20, 10, 4, 3, rng)
        op0 = (op0a * ProjectionParameterFunctional('p', 3, 0)
               + op0b * ProjectionParameterFunctional('p', 3, 1)
               + op0c * ProjectionParameterFunctional('p', 3, 1))
        op2 = (op2a * ProjectionParameterFunctional('p', 3, 0)
               + op2b * ProjectionParameterFunctional('q', 1))
        op = BlockOperator([[op0, op2],
                            [None, op1]])
        mu = op.parameters.parse({'p': [1, 2, 3], 'q': 4})
        sp = BlockDiagonalOperator([sp0, sp1])
        rp = BlockDiagonalOperator([rp0, rp1])
        U = op.source.make_array([U0, U1])
        V = op.range.make_array([V0, V1])
        return op, mu, U, V, sp, rp
    elif n == 10:
        from pymor.operators.block import BlockColumnOperator, BlockDiagonalOperator
        from pymor.parameters.functionals import ProjectionParameterFunctional
        op0, _, U0, V0, sp0, rp0 = numpy_matrix_operator_with_arrays_and_products_factory(10, 10, 4, 3, rng)
        op1, _, U1, V1, sp1, rp1 = numpy_matrix_operator_with_arrays_and_products_factory(20, 20, 4, 3, rng)
        op2a, _, _, _, _, _       = numpy_matrix_operator_with_arrays_and_products_factory(20, 10, 4, 3, rng)
        op2b, _, _, _, _, _       = numpy_matrix_operator_with_arrays_and_products_factory(20, 10, 4, 3, rng)
        op2 = (op2a * ProjectionParameterFunctional('p', 3, 0)
               + op2b * ProjectionParameterFunctional('q', 1))
        op = BlockColumnOperator([op2, op1])
        mu = op.parameters.parse({'p': [1, 2, 3], 'q': 4})
        sp = sp1
        rp = BlockDiagonalOperator([rp0, rp1])
        U = U1
        V = op.range.make_array([V0, V1])
        return op, mu, U, V, sp, rp
    elif n == 11:
        from pymor.operators.block import BlockDiagonalOperator, BlockRowOperator
        from pymor.parameters.functionals import ProjectionParameterFunctional
        op0, _, U0, V0, sp0, rp0 = numpy_matrix_operator_with_arrays_and_products_factory(10, 10, 4, 3, rng)
        op1, _, U1, V1, sp1, rp1 = numpy_matrix_operator_with_arrays_and_products_factory(20, 20, 4, 3, rng)
        op2a, _, _, _, _, _       = numpy_matrix_operator_with_arrays_and_products_factory(20, 10, 4, 3, rng)
        op2b, _, _, _, _, _       = numpy_matrix_operator_with_arrays_and_products_factory(20, 10, 4, 3, rng)
        op2 = (op2a * ProjectionParameterFunctional('p', 3, 0)
               + op2b * ProjectionParameterFunctional('q', 1))
        op = BlockRowOperator([op0, op2])
        mu = op.parameters.parse({'p': [1, 2, 3], 'q': 4})
        sp = BlockDiagonalOperator([sp0, sp1])
        rp = rp0
        U = op.source.make_array([U0, U1])
        V = V0
        return op, mu, U, V, sp, rp
    elif n == 12:
        from pymor.operators.constructions import NumpyConversionOperator
        from pymor.vectorarrays.block import BlockVectorSpace
        space = BlockVectorSpace([NumpyVectorSpace(1), NumpyVectorSpace(2)])
        op = NumpyConversionOperator(space)
        return op, None, op.source.random(), op.range.random(), IdentityOperator(op.source), IdentityOperator(op.range)
    else:
        assert False


num_unpicklable_misc_operators = 1


def unpicklable_misc_operator_with_arrays_and_products_factory(n, rng):
    if n == 0:
        from pymor.operators.numpy import NumpyGenericOperator
        op, _, U, V, sp, rp = numpy_matrix_operator_with_arrays_and_products_factory(100, 20, 4, 3, rng)
        mat = op.matrix
        op2 = NumpyGenericOperator(mapping=lambda U: mat.dot(U.T).T, adjoint_mapping=lambda U: mat.T.dot(U.T).T,
                                   dim_source=100, dim_range=20, linear=True)
        return op2, _, U, V, sp, rp
    else:
        assert False


misc_operator_generators = \
    [lambda rng, n=n: misc_operator_with_arrays_and_products_factory(n, rng)[0:2] for n in range(num_misc_operators)]

misc_operator_with_arrays_generators = \
    [lambda rng, n=n: misc_operator_with_arrays_and_products_factory(n, rng)[0:4] for n in range(num_misc_operators)]

misc_operator_with_arrays_and_products_generators = \
    [lambda rng, n=n: misc_operator_with_arrays_and_products_factory(n, rng) for n in range(num_misc_operators)]

unpicklable_misc_operator_generators = \
    [lambda rng, n=n: unpicklable_misc_operator_with_arrays_and_products_factory(n, rng)[0:2]
     for n in range(num_unpicklable_misc_operators)]

unpicklable_misc_operator_with_arrays_generators = \
    [lambda rng, n=n: unpicklable_misc_operator_with_arrays_and_products_factory(n, rng)[0:4]
     for n in range(num_unpicklable_misc_operators)]

unpicklable_misc_operator_with_arrays_and_products_generators = \
    [lambda rng, n=n: unpicklable_misc_operator_with_arrays_and_products_factory(n, rng)
     for n in range(num_unpicklable_misc_operators)]


if config.HAVE_FENICS:
    def fenics_matrix_operator_factory():
        import dolfin as df

        from pymor.bindings.fenics import FenicsMatrixOperator

        mesh = df.UnitSquareMesh(10, 10)
        V = df.FunctionSpace(mesh, 'CG', 2)

        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        c = df.Constant([1, 1])

        op = FenicsMatrixOperator(
            df.assemble(u * v * df.dx + df.inner(c, df.grad(u)) * v * df.dx),
            V, V)

        prod = FenicsMatrixOperator(df.assemble(u*v*df.dx), V, V)
        return op, None, op.source.random(), op.range.random(), prod, prod

    def fenics_nonlinear_operator_factory():
        import dolfin as df

        from pymor.bindings.fenics import FenicsMatrixOperator, FenicsOperator, FenicsVectorSpace

        class DirichletBoundary(df.SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[0] - 1.0) < df.DOLFIN_EPS and on_boundary

        mesh = df.UnitSquareMesh(10, 10)
        V = df.FunctionSpace(mesh, 'CG', 2)

        g = df.Constant(1.)
        c = df.Constant(1.)
        db = DirichletBoundary()
        bc = df.DirichletBC(V, g, db)

        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        w = df.Function(V)
        f = df.Expression('x[0]*sin(x[1])', degree=2)
        F = df.inner((1 + c*w**2)*df.grad(w), df.grad(v))*df.dx - f*v*df.dx

        space = FenicsVectorSpace(V)
        op = FenicsOperator(F, space, space, w, (bc,),
                            parameter_setter=lambda mu: c.assign(mu['c'].item()),
                            parameters={'c': 1},
                            solver_options={'inverse': {'type': 'newton', 'rtol': 1e-6}})

        prod = FenicsMatrixOperator(df.assemble(u*v*df.dx), V, V)
        return op, op.parameters.parse(42), op.source.random(), op.range.random(), prod, prod

    fenics_with_arrays_and_products_generators = [
        lambda rng: fenics_matrix_operator_factory(),
        lambda rng: fenics_nonlinear_operator_factory(),
    ]
    fenics_with_arrays_generators = [
        lambda rng: fenics_matrix_operator_factory()[:4],
        lambda rng: fenics_nonlinear_operator_factory()[:4],
    ]
else:
    fenics_with_arrays_and_products_generators = []
    fenics_with_arrays_generators = []


builtin_operator_with_arrays_and_products_generators = (
    [] if BUILTIN_DISABLED else
    thermalblock_operator_with_arrays_and_products_generators
    + thermalblock_assemble_operator_with_arrays_and_products_generators
    + thermalblock_concatenation_operator_with_arrays_and_products_generators
    + thermalblock_identity_operator_with_arrays_and_products_generators
    + thermalblock_zero_operator_with_arrays_and_products_generators
    + thermalblock_constant_operator_with_arrays_and_products_generators
    + thermalblock_vectorarray_operator_with_arrays_and_products_generators
    + thermalblock_vector_operator_with_arrays_and_products_generators
    + thermalblock_vectorfunc_operator_with_arrays_and_products_generators
    + thermalblock_fixedparam_operator_with_arrays_and_products_generators
    + misc_operator_with_arrays_and_products_generators
    + unpicklable_misc_operator_with_arrays_and_products_generators
)


@pytest.fixture(params=(
    builtin_operator_with_arrays_and_products_generators
    + fenics_with_arrays_and_products_generators
))
def operator_with_arrays_and_products(rng, request):
    return request.param(rng)


builtin_operator_with_arrays_generators = (
    [] if BUILTIN_DISABLED else
    numpy_matrix_operator_with_arrays_generators
    + numpy_list_vector_array_matrix_operator_with_arrays_generators
    + numpy_structured_matrix_operator_with_arrays_generators
    + thermalblock_operator_with_arrays_generators
    + thermalblock_assemble_operator_with_arrays_generators
    + thermalblock_concatenation_operator_with_arrays_generators
    + thermalblock_identity_operator_with_arrays_generators
    + thermalblock_zero_operator_with_arrays_generators
    + thermalblock_constant_operator_with_arrays_generators
    + thermalblock_vectorarray_operator_with_arrays_generators
    + thermalblock_vector_operator_with_arrays_generators
    + thermalblock_vectorfunc_operator_with_arrays_generators
    + thermalblock_fixedparam_operator_with_arrays_generators
    + misc_operator_with_arrays_generators
    + unpicklable_misc_operator_with_arrays_generators
)

@pytest.fixture(params=(
    builtin_operator_with_arrays_generators
    + fenics_with_arrays_generators
))
def operator_with_arrays(rng, request):
    return request.param(rng)


builtin_operator_generators = (
    [] if BUILTIN_DISABLED else
    numpy_matrix_operator_generators
    + numpy_list_vector_array_matrix_operator_generators
    + numpy_structured_matrix_operator_generators
    + thermalblock_operator_generators
    + thermalblock_assemble_operator_generators
    + thermalblock_concatenation_operator_generators
    + thermalblock_identity_operator_generators
    + thermalblock_zero_operator_generators
    + thermalblock_constant_operator_generators
    + thermalblock_vectorarray_operator_generators
    + thermalblock_vector_operator_generators
    + thermalblock_vectorfunc_operator_generators
    + thermalblock_fixedparam_operator_generators
    + misc_operator_generators
    + unpicklable_misc_operator_generators
)

@pytest.fixture(params=(
    builtin_operator_generators
))
def operator(rng, request):
    return request.param(rng)


builtin_picklable_operator_generators = (
    [] if BUILTIN_DISABLED else
    numpy_matrix_operator_generators
    + numpy_list_vector_array_matrix_operator_generators
    + numpy_structured_matrix_operator_generators
    + thermalblock_operator_generators
    + thermalblock_assemble_operator_generators
    + thermalblock_concatenation_operator_generators
    + thermalblock_identity_operator_generators
    + thermalblock_zero_operator_generators
    + thermalblock_constant_operator_generators
    + thermalblock_vectorarray_operator_generators
    + thermalblock_vector_operator_generators
    + thermalblock_vectorfunc_operator_generators
    + thermalblock_fixedparam_operator_generators
    + misc_operator_generators
)

@pytest.fixture(params=(
    builtin_picklable_operator_generators
))
def picklable_operator(rng, request):
    return request.param(rng)


@pytest.fixture
def loadable_matrices(shared_datadir):
    return (shared_datadir / 'matrices').glob('*')
