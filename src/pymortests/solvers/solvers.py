# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from scipy.sparse import coo_matrix, diags

from pymor.algorithms.basic import almost_equal
from pymor.bindings.scipy import (
    ScipyBicgStabSolver,
    ScipyBicgStabSpILUSolver,
    ScipyLGMRESSolver,
    ScipyLSMRSolver,
    ScipyLSQRSolver,
    ScipyLSTSQSolver,
    ScipyLUSolveSolver,
    ScipyQRLSTSQSolver,
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
all_scipy_dense_solvers = [ScipyLUSolveSolver, ScipyLSTSQSolver, ScipyQRLSTSQSolver]

@pytest.fixture(params=all_generic_solvers)
def generic_solver(request):
    return request.param()


@pytest.fixture(params=all_generic_solvers + all_scipy_sparse_solvers)
def numpy_sparse_solver(request):
    return request.param()


@pytest.fixture(params=all_scipy_dense_solvers)
def numpy_dense_solver(request):
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


def test_numpy_dense_solvers(numpy_dense_solver):
    op = mat_op.with_(solver=numpy_dense_solver)
    rhs = op.range.make_array(np.ones(10))
    solution = op.apply_inverse(rhs)
    assert ((op.apply(solution) - rhs).norm() / rhs.norm())[0] < 1e-8


def test_numpy_dense_adjoint_solvers(numpy_dense_solver):
    op = mat_op.with_(solver=numpy_dense_solver)
    rhs = op.source.make_array(np.ones(10))
    solution = op.apply_inverse_adjoint(rhs)
    assert ((op.apply_adjoint(solution) - rhs).norm() / rhs.norm())[0] < 1e-8


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


def test_sp_solve_solver_reuses_factorization_for_non_csc_matrix(monkeypatch):
    # regression test: ScipySpSolveSolver's `_factorizations` cache used to be keyed on the
    # local `matrix` variable *after* it was rebound by `matrix.tocsc()`. For any matrix that
    # wasn't already CSC (or CSR with UMFPACK), that rebound object had no other referrers and
    # was garbage collected as soon as `_solve_impl` returned, evicting the cache entry
    # immediately and silently defeating `keep_factorization=True` for every subsequent call.
    import pymor.bindings.scipy as scipy_bindings

    n = 20
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    K = coo_matrix(A @ A.T + n * np.eye(n))  # SPD; COO is never CSC and never CSR+UMFPACK-exempt

    splu_calls = []
    real_splu = scipy_bindings.splu

    def counting_splu(*args, **kwargs):
        splu_calls.append(None)
        return real_splu(*args, **kwargs)

    monkeypatch.setattr(scipy_bindings, 'splu', counting_splu)

    solver = ScipySpSolveSolver(keep_factorization=True, use_umfpack=False)
    op = NumpyMatrixOperator(K, solver=solver)
    rhs = op.range.make_array(np.ones(n))

    for _ in range(5):
        solution = op.apply_inverse(rhs)
        assert ((op.apply(solution) - rhs).norm() / rhs.norm())[0] < 1e-8

    assert len(splu_calls) == 1
