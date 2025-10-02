# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from scipy.sparse import diags

from pymor.algorithms.basic import almost_equal
from pymor.bindings.scipy import (
    ScipyBicgStabSolver,
    ScipyBicgStabSpILUSolver,
    ScipyLGMRESSolver,
    ScipyLSMRSolver,
    ScipyLSQRSolver,
    ScipySpSolveSolver,
)
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.solvers.generic import LGMRESSolver, LSMRSolver, LSQRSolver

pytestmark = pytest.mark.builtin


mat = np.eye(10) * np.arange(1, 11)
mat[-1,0] = 11
mat_op = NumpyMatrixOperator(mat)


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
    op = mat_op.with_(solver=generic_solver)
    op2 = mat_op
    rhs = op.range.make_array(np.ones(10))
    solution = generic_solver.solve(op, rhs)
    solution2 = op.apply_inverse(rhs)
    solution3 = op2.apply_inverse(rhs, solver=generic_solver)
    assert np.all(almost_equal(solution, solution2))
    assert np.all(almost_equal(solution, solution3))
    assert ((op.apply(solution) - rhs).norm() / rhs.norm())[0] < 1e-6


def test_generic_adjoint_solvers(generic_solver):
    op = mat_op.with_(solver=generic_solver)
    op2 = mat_op
    rhs = op.source.make_array(np.ones(10))
    solution = generic_solver.solve_adjoint(op, rhs)
    solution2 = op.apply_inverse_adjoint(rhs)
    solution3 = op2.apply_inverse_adjoint(rhs, solver=generic_solver)
    assert np.all(almost_equal(solution, solution2))
    assert np.all(almost_equal(solution, solution3))
    assert ((op.apply_adjoint(solution) - rhs).norm() / rhs.norm())[0] < 1e-6


def test_numpy_dense_solvers():
    rhs = mat_op.range.make_array(np.ones(10))
    solution = mat_op.apply_inverse(rhs)
    assert ((mat_op.apply(solution) - rhs).norm() / rhs.norm())[0] < 1e-8


def test_numpy_dense_adjoint_solvers():
    rhs = mat_op.source.make_array(np.ones(10))
    solution = mat_op.apply_inverse_adjoint(rhs)
    assert ((mat_op.apply_adjoint(solution) - rhs).norm() / rhs.norm())[0] < 1e-8


def test_numpy_sparse_solvers(numpy_sparse_solver):
    op = NumpyMatrixOperator(diags([np.arange(1., 11.)], [0], format='csc'), solver=numpy_sparse_solver)
    rhs = op.range.make_array(np.ones(10))
    solution = op.apply_inverse(rhs)
    assert ((op.apply(solution) - rhs).norm() / rhs.norm())[0] < 1e-8


def test_numpy_sparse_adjoint_solvers(numpy_sparse_solver):
    op = NumpyMatrixOperator(diags([np.arange(1., 11.)], [0], format='csc'), solver=numpy_sparse_solver)
    rhs = op.source.make_array(np.ones(10))
    solution = op.apply_inverse_adjoint(rhs)
    assert ((op.apply_adjoint(solution) - rhs).norm() / rhs.norm())[0] < 1e-8
