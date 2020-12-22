# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import os
import sys

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.algorithms.lyapunov import solve_lyap_lrcf, solve_lyap_dense
from pymor.core.config import config
from pymor.operators.numpy import NumpyMatrixOperator

import pytest


n_list = [100, 200]
m_list = [1, 2]
lyap_lrcf_solver_list = [
    'scipy',
    'slycot_bartels-stewart',
    'pymess_glyap',
    'pymess_lradi',
    'lradi',
]
lyap_dense_solver_list = [
    'scipy',
    'slycot_bartels-stewart',
    'pymess_glyap',
]


def fro_norm(A):
    if not sps.issparse(A):
        return spla.norm(A)
    else:
        return sps.linalg.norm(A)


def conv_diff_1d_fd(n, a, b):
    diagonals = [-a * 2 * (n + 1) ** 2 * np.ones((n,)),
                 (a * (n + 1) ** 2 + b * (n + 1) / 2) * np.ones((n - 1,)),
                 (a * (n + 1) ** 2 - b * (n + 1) / 2) * np.ones((n - 1,))]
    A = sps.diags(diagonals, [0, -1, 1], format='csc')
    return A


def conv_diff_1d_fem(n, a, b):
    diagonals = [-a * 2 * (n + 1) ** 2 * np.ones((n,)),
                 (a * (n + 1) ** 2 + b * (n + 1) / 2) * np.ones((n - 1,)),
                 (a * (n + 1) ** 2 - b * (n + 1) / 2) * np.ones((n - 1,))]
    A = sps.diags(diagonals, [0, -1, 1], format='csc')
    diagonals = [2 / 3 * np.ones((n,)),
                 1 / 6 * np.ones((n - 1,)),
                 1 / 6 * np.ones((n - 1,))]
    E = sps.diags(diagonals, [0, -1, 1], format='csc')
    return A, E


def relative_residual(A, E, B, X, trans=False):
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
    return res / rhs


def _check_availability(lyap_solver):
    if (lyap_solver.startswith('slycot')
            and not os.environ.get('DOCKER_PYMOR', False)
            and not config.HAVE_SLYCOT):
        pytest.skip('slycot not available')
    if (lyap_solver.startswith('pymess')
            and not os.environ.get('DOCKER_PYMOR', False)
            and not config.HAVE_PYMESS):
        pytest.skip('pymess not available')


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('lyap_solver', lyap_lrcf_solver_list)
def test_lrcf(n, m, with_E, trans, lyap_solver):
    _check_availability(lyap_solver)

    if not with_E:
        A = conv_diff_1d_fd(n, 1, 1)
        E = None
    else:
        A, E = conv_diff_1d_fem(n, 1, 1)
    np.random.seed(0)
    B = np.random.randn(n, m)
    if trans:
        B = B.T

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E) if with_E else None
    Bva = Aop.source.from_numpy(B.T if not trans else B)

    Zva = solve_lyap_lrcf(Aop, Eop, Bva, trans=trans, options=lyap_solver)
    assert len(Zva) <= n

    Z = Zva.to_numpy().T
    assert relative_residual(A, E, B, Z @ Z.T, trans=trans) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('trans', [False, True])
@pytest.mark.parametrize('lyap_solver', lyap_dense_solver_list)
def test_dense(n, m, with_E, trans, lyap_solver):
    _check_availability(lyap_solver)

    np.random.seed(0)
    A = np.random.randn(n, n)
    E = np.eye(n) + np.random.randn(n, n) / n if with_E else None
    B = np.random.randn(n, m)
    if trans:
        B = B.T

    X = solve_lyap_dense(A, E, B, trans=trans, options=lyap_solver)
    assert type(X) is np.ndarray

    assert relative_residual(A, E, B, X, trans=trans) < 1e-10
