# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from scipy.sparse import diags

from pymor.bindings.scipy import (
    ScipyBicgStabSolver,
    ScipyBicgStabSpILUSolver,
    ScipyLGMRESSolver,
    ScipyLSMRSolver,
    ScipyLSQRSolver,
    ScipySpSolveSolver,
)
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.solvers.generic import LGMRESSolver, LSMRSolver, LSQRSolver
from pymor.vectorarrays.numpy import NumpyVectorSpace

pytestmark = pytest.mark.builtin


class GenericOperator(Operator):

    source = range = NumpyVectorSpace(10)
    op = NumpyMatrixOperator(np.eye(10) * np.arange(1, 11))
    linear = True

    def __init__(self, solver=None):
        self.solver = solver

    def apply(self, U, mu=None):
        return self.op.apply(U, mu=mu)

    def apply_adjoint(self, V, mu=None):
        return self.op.apply_adjoint(V, mu=mu)


all_generic_solvers = [LGMRESSolver, LSMRSolver, LSQRSolver]
all_scipy_sparse_solvers = [ScipyBicgStabSolver, ScipyBicgStabSpILUSolver, ScipyLGMRESSolver, ScipyLSMRSolver,
                            ScipyLSQRSolver, ScipySpSolveSolver]

@pytest.fixture(params=all_generic_solvers)
def generic_solver(request):
    return request.param()


@pytest.fixture(params=all_generic_solvers + all_scipy_sparse_solvers)
def numpy_sparse_solver(request):
    return request.param()


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
    op = NumpyMatrixOperator(diags([np.arange(1., 11.)], [0], format='csc'), solver=numpy_sparse_solver)
    rhs = op.range.make_array(np.ones(10))
    solution = op.apply_inverse(rhs)
    assert ((op.apply(solution) - rhs).norm() / rhs.norm())[0] < 1e-8
