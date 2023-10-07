# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import chain, product

import numpy as np
import pytest
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.algorithms.lyapunov import _chol
from pymor.algorithms.riccati import solve_pos_ricc_lrcf, solve_ricc_dense, solve_ricc_lrcf
from pymor.operators.numpy import NumpyMatrixOperator
from pymortests.algorithms.lyapunov import conv_diff_1d_fd, conv_diff_1d_fem, fro_norm, skip_if_missing_solver

pytestmark = pytest.mark.builtin


n_list_small = [10, 20]
n_list_big = [250]
m_list = [1, 2]
p_list = [1, 2]
ricc_lrcf_solver_list_small = [
    'scipy',
    'slycot',
    'pymess_dense_nm_gmpcare',
]
ricc_lrcf_solver_list_big = [
    'pymess_lrnm',
    'lrradi'
]
ricc_dense_solver_list = [
    'scipy',
    'slycot'
]


def relative_residual(A, E, B, C, R, S, Z, trans):
    if not trans:
        if E is None:
            linear = A @ Z @ Z.T
            quadratic = Z
        else:
            linear = A @ Z @ (Z.T @ E.T)
            quadratic = E @ Z
        quadratic = quadratic @ (Z.T @ C.T)
        if S is not None:
            quadratic = quadratic + S.T
        RHS = B @ B.T
    else:
        if E is None:
            linear = A.T @ Z @ Z.T
            quadratic = Z
        else:
            linear = A.T @ Z @ (Z.T @ E)
            quadratic = E.T @ Z
        quadratic = quadratic @ (Z.T @ B)
        if S is not None:
            quadratic = quadratic + S
        RHS = C.T @ C
    linear += linear.T
    if R is None:
        quadratic = quadratic @ quadratic.T
    else:
        quadratic = quadratic @ spla.solve(R, quadratic.T)
    res = fro_norm(linear - quadratic + RHS)
    rhs = fro_norm(RHS)
    return res / rhs


@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('with_R', [False, True])
@pytest.mark.parametrize('with_S', [False, True])
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('solver', ricc_dense_solver_list)
def test_ricc_dense(n, m, p, with_E, with_R, with_S, trans, solver):
    skip_if_missing_solver(solver)

    mat_old = []
    mat_new = []
    if not with_E:
        A = conv_diff_1d_fd(n, 1, 1)
        A = A.toarray()
        E = None
    else:
        A, E = conv_diff_1d_fem(n, 1, 1)
        A = A.toarray()
        E = E.toarray()
        mat_old.append(E.copy())
        mat_new.append(E)
    A = np.asfortranarray(A)
    mat_old.append(A.copy())
    mat_new.append(A)
    B = np.random.randn(n, m)
    mat_old.append(B.copy())
    mat_new.append(B)
    C = np.random.randn(p, n)
    mat_old.append(C.copy())
    mat_new.append(C)
    D = np.random.randn(p, m)
    if not trans:
        R0 = np.random.randn(p, p)
        R = D.dot(D.T) + R0.dot(R0.T) if with_R else None
        S = 1e-1 * D @ B.T if with_S else None
    else:
        R0 = np.random.randn(m, m)
        R = D.T.dot(D) + R0.dot(R0.T) if with_R else None
        S = 1e-1 * C.T @ D if with_S else None
    if with_R:
        mat_old.append(R.copy())
        mat_new.append(R)
    if with_S:
        mat_old.append(S.copy())
        mat_new.append(S)

    X = solve_ricc_dense(A, E, B, C, R, S, trans=trans, options=solver)

    assert relative_residual(A, E, B, C, R, S, _chol(X), trans) < 1e-8

    for mat1, mat2 in zip(mat_old, mat_new):
        assert type(mat1) == type(mat2)
        assert np.all(mat1 == mat2)


@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('with_R', [False, True])
@pytest.mark.parametrize('with_S', [False, True])
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('n,solver', chain(product(n_list_small, ricc_lrcf_solver_list_small),
                                           product(n_list_big, ricc_lrcf_solver_list_big)))
def test_ricc_lrcf(n, m, p, with_E, with_R, with_S, trans, solver):
    skip_if_missing_solver(solver)
    if with_S and (solver.startswith('pymess') or solver == 'lrradi'):
        pytest.xfail('solver not implemented')

    mat_old = []
    mat_new = []
    if not with_E:
        A = conv_diff_1d_fd(n, 1, 1)
        E = None
    else:
        A, E = conv_diff_1d_fem(n, 1, 1)
        mat_old.append(E.copy())
        mat_new.append(E)
    mat_old.append(A.copy())
    mat_new.append(A)
    B = np.random.randn(n, m)
    mat_old.append(B.copy())
    mat_new.append(B)
    C = np.random.randn(p, n)
    mat_old.append(C.copy())
    mat_new.append(C)
    D = np.random.randn(p, m)
    if not trans:
        R0 = np.random.randn(p, p)
        R = D.dot(D.T) + R0.dot(R0.T) if with_R else None
        S = 1e-1 * D @ B.T if with_S else None
    else:
        R0 = np.random.randn(m, m)
        R = D.T.dot(D) + R0.dot(R0.T) if with_R else None
        S = 1e-1 * C.T @ D if with_S else None
    if with_R:
        mat_old.append(R.copy())
        mat_new.append(R)
    if with_S:
        mat_old.append(S.copy())
        mat_new.append(S)

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E) if with_E else None
    Bva = Aop.source.from_numpy(B.T)
    Cva = Aop.source.from_numpy(C)
    Sva = Aop.source.from_numpy((S if not trans else S.T)) if with_S else None

    Zva = solve_ricc_lrcf(Aop, Eop, Bva, Cva, R, Sva, trans=trans, options=solver)

    assert len(Zva) <= n

    Z = Zva.to_numpy().T
    assert relative_residual(A, E, B, C, R, S, Z, trans) < 1e-8

    for mat1, mat2 in zip(mat_old, mat_new):
        assert type(mat1) == type(mat2)
        if sps.issparse(mat1):
            mat1 = mat1.toarray()
            mat2 = mat2.toarray()
        assert np.all(mat1 == mat2)


@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('with_R', [False, True])
@pytest.mark.parametrize('with_S', [False, True])
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('solver', ricc_lrcf_solver_list_small)
def test_pos_ricc_lrcf(n, m, p, with_E, with_R, with_S, trans, solver):
    skip_if_missing_solver(solver)
    if with_S and solver.startswith('pymess'):
        pytest.xfail('solver not implemented')

    mat_old = []
    mat_new = []
    if not with_E:
        A = conv_diff_1d_fd(n, 1, 1)
        E = None
    else:
        A, E = conv_diff_1d_fem(n, 1, 1)
        mat_old.append(E.copy())
        mat_new.append(E)
    mat_old.append(A.copy())
    mat_new.append(A)
    B = np.random.randn(n, m)
    mat_old.append(B.copy())
    mat_new.append(B)
    C = np.random.randn(p, n)
    mat_old.append(C.copy())
    mat_new.append(C)
    D = np.random.randn(p, m)
    if not trans:
        R0 = np.random.randn(p, p)
        R = D.dot(D.T) + 10 * R0.dot(R0.T) if with_R else None
        S = np.random.randn(p,n) if with_S else None
    else:
        R0 = np.random.randn(m, m)
        R = D.T.dot(D) + 10 * R0.dot(R0.T) if with_R else None
        S = np.random.randn(n,m) if with_S else None
    if with_R:
        mat_old.append(R.copy())
        mat_new.append(R)
    if with_S:
        mat_old.append(S.copy())
        mat_new.append(S)

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E) if with_E else None
    Bva = Aop.source.from_numpy(B.T)
    Cva = Aop.source.from_numpy(C)
    Sva = Aop.source.from_numpy((S if not trans else S.T)) if with_S else None

    Zva = solve_pos_ricc_lrcf(Aop, Eop, Bva, Cva, R, Sva, trans=trans, options=solver)

    assert len(Zva) <= n

    Z = Zva.to_numpy().T
    if not with_R:
        R = np.eye(p if not trans else m)
    assert relative_residual(A, E, B, C, -R, S, Z, trans) < 1e-8

    for mat1, mat2 in zip(mat_old, mat_new):
        assert type(mat1) == type(mat2)
        if sps.issparse(mat1):
            mat1 = mat1.toarray()
            mat2 = mat2.toarray()
        assert np.all(mat1 == mat2)
