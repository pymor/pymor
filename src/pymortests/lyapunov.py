# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import chain, product

import numpy as np
import pytest
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.algorithms.lyapunov import (
    solve_cont_lyap_dense,
    solve_disc_lyap_dense,
    solve_cont_lyap_lrcf,
    solve_disc_lyap_lrcf,
)
from pymor.operators.numpy import NumpyMatrixOperator
from pymortests.base import skip_if_missing

n_list_small = [10, 20]
n_list_big = [300]
m_list = [1, 2]
cont_lyap_lrcf_solver_list = [
    'pymess_lradi',
    'lradi',
]
cont_lyap_dense_solver_list = [
    'scipy',
    'slycot_bartels-stewart',
    'pymess_glyap',
]
disc_lyap_dense_solver_list = [
    'scipy',
    'slycot_bartels-stewart',
]


def fro_norm(A):
    if not sps.issparse(A):
        return spla.norm(A)
    else:
        return sps.linalg.norm(A)


def conv_diff_1d_fd(n, a, b, cont_time=True):
    diagonals = [-a * 2 * (n + 1) ** 2 * np.ones((n,)),
                 (a * (n + 1) ** 2 + b * (n + 1) / 2) * np.ones((n - 1,)),
                 (a * (n + 1) ** 2 - b * (n + 1) / 2) * np.ones((n - 1,))]
    A = sps.diags(diagonals, [0, -1, 1], format='csc')
    if not cont_time:
        dt = 0.1 / (4*a*(n+1)**2 + b*(n+1))
        A = sps.eye(n) + dt * A
    return A


def conv_diff_1d_fem(n, a, b, cont_time=True):
    diagonals = [-a * 2 * (n + 1) ** 2 * np.ones((n,)),
                 (a * (n + 1) ** 2 + b * (n + 1) / 2) * np.ones((n - 1,)),
                 (a * (n + 1) ** 2 - b * (n + 1) / 2) * np.ones((n - 1,))]
    A = sps.diags(diagonals, [0, -1, 1], format='csc')
    diagonals = [2 / 3 * np.ones((n,)),
                 1 / 6 * np.ones((n - 1,)),
                 1 / 6 * np.ones((n - 1,))]
    E = sps.diags(diagonals, [0, -1, 1], format='csc')
    if not cont_time:
        dt = 0.1 / (4*a*(n+1)**2 + b*(n+1))
        A = E + dt * A
    return A, E


def relative_residual(A, E, B, X, cont_time, trans=False):
    if cont_time:
        if not trans:
            if E is None:
                AX = A @ X
                BBT = B @ B.T
                res = fro_norm(AX + AX.T + BBT)
                rhs = fro_norm(BBT)
            else:
                AXET = A @ X @ E.T
                BBT = B @ B.T
                res = fro_norm(AXET + AXET.T + BBT)
                rhs = fro_norm(BBT)
        else:
            if E is None:
                ATX = A.T @ X
                CTC = B.T @ B
                res = fro_norm(ATX + ATX.T + CTC)
                rhs = fro_norm(CTC)
            else:
                ATXE = A.T @ X @ E
                CTC = B.T @ B
                res = fro_norm(ATXE + ATXE.T + CTC)
                rhs = fro_norm(CTC)
    else:
        if not trans:
            AXAT = A @ X @ A.T
            BBT = B @ B.T
            if E is None:
                res = fro_norm(AXAT - X + BBT)
                rhs = fro_norm(BBT)
            else:
                EXET = E @ X @ E.T
                res = fro_norm(AXAT - EXET + BBT)
                rhs = fro_norm(BBT)
        else:
            ATXA = A.T @ X @ A
            CTC = B.T @ B
            if E is None:
                res = fro_norm(ATXA - X + CTC)
                rhs = fro_norm(CTC)
            else:
                ETXE = E.T @ X @ E
                res = fro_norm(ATXA - ETXE + CTC)
                rhs = fro_norm(CTC)
    return res / rhs


@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('n,lyap_solver', chain(product(n_list_small, cont_lyap_dense_solver_list),
                                                product(n_list_big, cont_lyap_lrcf_solver_list)))
@skip_if_missing('SLYCOT')
@skip_if_missing('PYMESS')
def test_cont_lrcf(n, m, with_E, trans, lyap_solver):
    if not with_E:
        A = conv_diff_1d_fd(n, 1, 0.1, cont_time=True)
        E = None
    else:
        A, E = conv_diff_1d_fem(n, 1, 0.1, cont_time=True)
    np.random.seed(0)
    B = np.random.randn(n, m)
    if trans:
        B = B.T

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E) if with_E else None
    Bva = Aop.source.from_numpy(B.T if not trans else B)

    Zva = solve_cont_lyap_lrcf(Aop, Eop, Bva, trans=trans, options=lyap_solver)
    assert len(Zva) <= n

    Z = Zva.to_numpy().T
    assert relative_residual(A, E, B, Z @ Z.T, trans=trans, cont_time=True) < 1e-10


@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('lyap_solver', disc_lyap_dense_solver_list)
@skip_if_missing('SLYCOT')
@skip_if_missing('PYMESS')
def test_disc_lrcf(n, m, with_E, trans, lyap_solver):
    if not with_E:
        A = conv_diff_1d_fd(n, 1, 0.1, cont_time=False)
        E = None
    else:
        A, E = conv_diff_1d_fem(n, 1, 0.1, cont_time=False)

    np.random.seed(0)
    B = np.random.randn(n, m)
    if trans:
        B = B.T

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E) if with_E else None
    Bva = Aop.source.from_numpy(B.T if not trans else B)

    Zva = solve_disc_lyap_lrcf(Aop, Eop, Bva, trans=trans, options=lyap_solver)
    assert len(Zva) <= n

    Z = Zva.to_numpy().T
    assert relative_residual(A, E, B, Z @ Z.T, trans=trans, cont_time=False) < 1e-10


@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('lyap_solver', cont_lyap_dense_solver_list)
@skip_if_missing('SLYCOT')
@skip_if_missing('PYMESS')
def test_cont_dense(n, m, with_E, trans, lyap_solver):
    np.random.seed(0)
    A = np.random.randn(n, n)
    E = np.eye(n) + np.random.randn(n, n) / n if with_E else None
    B = np.random.randn(n, m)
    if trans:
        B = B.T

    X = solve_cont_lyap_dense(A, E, B, trans=trans, options=lyap_solver)
    assert type(X) is np.ndarray

    assert relative_residual(A, E, B, X, trans=trans, cont_time=True) < 1e-10


@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('lyap_solver', disc_lyap_dense_solver_list)
@skip_if_missing('SLYCOT')
@skip_if_missing('PYMESS')
def test_disc_dense(n, m, with_E, trans, lyap_solver):
    np.random.seed(0)
    A = np.random.randn(n, n)
    E = np.eye(n) + np.random.randn(n, n) / n if with_E else None
    B = np.random.randn(n, m)
    if trans:
        B = B.T

    X = solve_disc_lyap_dense(A, E, B, trans=trans, options=lyap_solver)
    assert type(X) is np.ndarray

    assert relative_residual(A, E, B, X, trans=trans, cont_time=False) < 1e-10
