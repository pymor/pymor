# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from scipy.sparse import diags
import pytest

import pymor.algorithms.genericsolvers
from pymor.operators.basic import OperatorBase
from pymor.operators.numpy import NumpyMatrixOperator, sparse_options, dense_options
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray


class GenericOperator(OperatorBase):

    source = range = NumpyVectorSpace(10)
    op = NumpyMatrixOperator(np.eye(10) * np.arange(1, 11))
    linear = True

    def __init__(self, solver_options=None):
        self.solver_options = solver_options

    def apply(self, U, ind=None, mu=None):
        return self.op.apply(U, ind=ind, mu=mu)

    def apply_adjoint(self, U, ind=None, mu=None):
        return self.op.apply_adjoint(U, ind=ind, mu=mu)


@pytest.fixture(params=pymor.algorithms.genericsolvers.options().keys())
def generic_solver(request):
    return {'inverse': request.param}


@pytest.fixture(params=dense_options().keys())
def numpy_dense_solver(request):
    return {'inverse': request.param}


@pytest.fixture(params=sparse_options().keys())
def numpy_sparse_solver(request):
    return {'inverse': request.param}


def test_generic_solvers(generic_solver):
    op = GenericOperator(generic_solver)
    rhs = NumpyVectorArray(np.ones(10))
    solution = op.apply_inverse(rhs)
    assert ((op.apply(solution) - rhs).l2_norm() / rhs.l2_norm())[0] < 1e-8


def test_numpy_dense_solvers(numpy_dense_solver):
    op = NumpyMatrixOperator(np.eye(10) * np.arange(1, 11), solver_options=numpy_dense_solver)
    rhs = NumpyVectorArray(np.ones(10))
    solution = op.apply_inverse(rhs)
    assert ((op.apply(solution) - rhs).l2_norm() / rhs.l2_norm())[0] < 1e-8


def test_numpy_sparse_solvers(numpy_sparse_solver):
    op = NumpyMatrixOperator(diags([np.arange(1., 11.)], [0]), solver_options=numpy_sparse_solver)
    rhs = NumpyVectorArray(np.ones(10))
    solution = op.apply_inverse(rhs)
    assert ((op.apply(solution) - rhs).l2_norm() / rhs.l2_norm())[0] < 1e-8
