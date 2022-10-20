# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from scipy.sparse import diags
import pytest

import pymor.algorithms.genericsolvers
from pymor.bindings.scipy import solver_options as scipy_solver_options
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class GenericOperator(Operator):

    source = range = NumpyVectorSpace(10)
    op = NumpyMatrixOperator(np.eye(10) * np.arange(1, 11))
    linear = True

    def __init__(self, solver_options=None):
        self.solver_options = solver_options

    def apply(self, U, mu=None):
        return self.op.apply(U, mu=mu)

    def apply_adjoint(self, V, mu=None):
        return self.op.apply_adjoint(V, mu=mu)


@pytest.fixture(params=pymor.algorithms.genericsolvers.solver_options().keys())
def generic_solver(request):
    return {'inverse': request.param}


all_sparse_solvers = set(pymor.algorithms.genericsolvers.solver_options().keys())
all_sparse_solvers.update(scipy_solver_options().keys())


@pytest.fixture(params=all_sparse_solvers)
def numpy_sparse_solver(request):
    return {'inverse': request.param}


def test_generic_solvers(generic_solver):
    op = GenericOperator(generic_solver)
    rhs = op.range.make_array(np.ones(10))
    solution = op.apply_inverse(rhs)
    assert ((op.apply(solution) - rhs).norm() / rhs.norm())[0] < 1e-8


def test_numpy_dense_solvers():
    op = NumpyMatrixOperator(np.eye(10) * np.arange(1, 11))
    rhs = op.range.make_array(np.ones(10))
    solution = op.apply_inverse(rhs)
    assert ((op.apply(solution) - rhs).norm() / rhs.norm())[0] < 1e-8


def test_numpy_sparse_solvers(numpy_sparse_solver):
    op = NumpyMatrixOperator(diags([np.arange(1., 11.)], [0]), solver_options=numpy_sparse_solver)
    rhs = op.range.make_array(np.ones(10))
    solution = op.apply_inverse(rhs)
    assert ((op.apply(solution) - rhs).norm() / rhs.norm())[0] < 1e-8
