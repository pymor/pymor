# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import chain, product

import numpy as np
import pytest
import scipy.linalg as spla

from pymor.algorithms.lyapunov import _chol
from pymor.algorithms.riccati import solve_pos_ricc_lrcf, solve_ricc_dense, solve_ricc_lrcf
from pymor.operators.numpy import NumpyMatrixOperator
from pymortests.lyapunov import conv_diff_1d_fd, conv_diff_1d_fem, fro_norm
from pymortests.base import skip_if_missing

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


def relative_residual(A, E, B, C, R, Z, trans):
    if not trans:
        if E is None:
            linear = A @ Z @ Z.T
            quadratic = Z
        else:
            linear = A @ Z @ (Z.T @ E.T)
            quadratic = E @ Z
        quadratic = quadratic @ (Z.T @ C.T)
        RHS = B @ B.T
    else:
        if E is None:
            linear = A.T @ Z @ Z.T
            quadratic = Z
        else:
            linear = A.T @ Z @ (Z.T @ E)
            quadratic = E.T @ Z
        quadratic = quadratic @ (Z.T @ B)
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
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('solver', ricc_dense_solver_list)
@skip_if_missing('SLYCOT')
@skip_if_missing('PYMESS')
def test_ricc_dense(n, m, p, with_E, with_R, trans, solver):
    if not with_E:
        A = conv_diff_1d_fd(n, 1, 1)
        A = A.todense()
        E = None
    else:
        A, E = conv_diff_1d_fem(n, 1, 1)
        A = A.todense()
        E = E.todense()
    np.random.seed(0)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    D = np.random.randn(p, m)
    if not trans:
        R0 = np.random.randn(p, p)
        R = D.dot(D.T) + R0.dot(R0.T) if with_R else None
    else:
        R0 = np.random.randn(m, m)
        R = D.T.dot(D) + R0.dot(R0.T) if with_R else None

    X = solve_ricc_dense(A, E, B, C, R, trans=trans, options=solver)

    assert relative_residual(A, E, B, C, R, _chol(X), trans) < 1e-8


@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('with_R', [False, True])
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('n,solver', chain(product(n_list_small, ricc_lrcf_solver_list_small),
                                           product(n_list_big, ricc_lrcf_solver_list_big)))
@skip_if_missing('SLYCOT')
@skip_if_missing('PYMESS')
def test_ricc_lrcf(n, m, p, with_E, with_R, trans, solver):
    if not with_E:
        A = conv_diff_1d_fd(n, 1, 1)
        E = None
    else:
        A, E = conv_diff_1d_fem(n, 1, 1)
    np.random.seed(0)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    D = np.random.randn(p, m)
    if not trans:
        R0 = np.random.randn(p, p)
        R = D.dot(D.T) + R0.dot(R0.T) if with_R else None
    else:
        R0 = np.random.randn(m, m)
        R = D.T.dot(D) + R0.dot(R0.T) if with_R else None

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E) if with_E else None
    Bva = Aop.source.from_numpy(B.T)
    Cva = Aop.source.from_numpy(C)

    try:
        Zva = solve_ricc_lrcf(Aop, Eop, Bva, Cva, R, trans=trans, options=solver)
    except NotImplementedError:
        return

    assert len(Zva) <= n

    Z = Zva.to_numpy().T
    assert relative_residual(A, E, B, C, R, Z, trans) < 1e-8


@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('with_R', [False, True])
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('solver', ricc_lrcf_solver_list_small)
@skip_if_missing('SLYCOT')
@skip_if_missing('PYMESS')
def test_pos_ricc_lrcf(n, m, p, with_E, with_R, trans, solver):
    if not with_E:
        A = conv_diff_1d_fd(n, 1, 1)
        E = None
    else:
        A, E = conv_diff_1d_fem(n, 1, 1)
    np.random.seed(0)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    D = np.random.randn(p, m)
    if not trans:
        R0 = np.random.randn(p, p)
        R = D.dot(D.T) + 10 * R0.dot(R0.T) if with_R else None
    else:
        R0 = np.random.randn(m, m)
        R = D.T.dot(D) + 10 * R0.dot(R0.T) if with_R else None

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E) if with_E else None
    Bva = Aop.source.from_numpy(B.T)
    Cva = Aop.source.from_numpy(C)

    Zva = solve_pos_ricc_lrcf(Aop, Eop, Bva, Cva, R, trans=trans, options=solver)
    assert len(Zva) <= n

    Z = Zva.to_numpy().T
    if not with_R:
        R = np.eye(p if not trans else m)
    assert relative_residual(A, E, B, C, -R, Z, trans) < 1e-8
