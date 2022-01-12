# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.algorithms.sylvester import solve_sylv_schur
from pymor.operators.numpy import NumpyMatrixOperator

import pytest


n_list = [100, 1000]
r_list = [1, 10, 20]
m_list = [1, 2]
p_list = [1, 2]


def fro_norm(A):
    if not sps.issparse(A):
        return spla.norm(A)
    else:
        return sps.linalg.norm(A)


def diff_conv_1d_fd(n, a, b):
    diagonals = [-a * 2 * (n + 1) ** 2 * np.ones((n,)),
                 (a * (n + 1) ** 2 + b * (n + 1) / 2) * np.ones((n - 1,)),
                 (a * (n + 1) ** 2 - b * (n + 1) / 2) * np.ones((n - 1,))]
    A = sps.diags(diagonals, [0, -1, 1])
    return A


def diff_conv_1d_fem(n, a, b):
    diagonals = [-a * 2 * (n + 1) ** 2 * np.ones((n,)),
                 (a * (n + 1) ** 2 + b * (n + 1) / 2) * np.ones((n - 1,)),
                 (a * (n + 1) ** 2 - b * (n + 1) / 2) * np.ones((n - 1,))]
    A = sps.diags(diagonals, [0, -1, 1])
    diagonals = [2 / 3 * np.ones((n,)),
                 1 / 6 * np.ones((n - 1,)),
                 1 / 6 * np.ones((n - 1,))]
    E = sps.diags(diagonals, [0, -1, 1])
    return A, E


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('r', r_list)
@pytest.mark.parametrize('m', m_list)
def test_sylv_schur_V(n, r, m):
    np.random.seed(0)

    A = diff_conv_1d_fd(n, 1, 1)
    B = np.random.randn(n, m)

    Ar = np.random.randn(r, r) - r * np.eye(r)
    Br = np.random.randn(r, m)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)

    Arop = NumpyMatrixOperator(Ar)
    Brop = NumpyMatrixOperator(Br)

    Vva = solve_sylv_schur(Aop, Arop, B=Bop, Br=Brop)

    V = Vva.to_numpy().T

    AV = A.dot(V)
    VArT = V.dot(Ar.T)
    BBrT = B.dot(Br.T)
    assert fro_norm(AV + VArT + BBrT) / fro_norm(BBrT) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('r', r_list)
@pytest.mark.parametrize('m', m_list)
def test_sylv_schur_V_E(n, r, m):
    np.random.seed(0)

    A, E = diff_conv_1d_fem(n, 1, 1)
    B = np.random.randn(n, m)

    Ar = np.random.randn(r, r) - r * np.eye(r)
    Er = np.random.randn(r, r)
    Er = (Er + Er.T) / 2
    Er += r * np.eye(r)
    Br = np.random.randn(r, m)

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E)
    Bop = NumpyMatrixOperator(B)

    Arop = NumpyMatrixOperator(Ar)
    Erop = NumpyMatrixOperator(Er)
    Brop = NumpyMatrixOperator(Br)

    Vva = solve_sylv_schur(Aop, Arop, E=Eop, Er=Erop, B=Bop, Br=Brop)

    V = Vva.to_numpy().T

    AVErT = A.dot(V.dot(Er.T))
    EVArT = E.dot(V.dot(Ar.T))
    BBrT = B.dot(Br.T)
    assert fro_norm(AVErT + EVArT + BBrT) / fro_norm(BBrT) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('r', r_list)
@pytest.mark.parametrize('p', p_list)
def test_sylv_schur_W(n, r, p):
    np.random.seed(0)

    A = diff_conv_1d_fd(n, 1, 1)
    C = np.random.randn(p, n)

    Ar = np.random.randn(r, r) - r * np.eye(r)
    Cr = np.random.randn(p, r)

    Aop = NumpyMatrixOperator(A)
    Cop = NumpyMatrixOperator(C)

    Arop = NumpyMatrixOperator(Ar)
    Crop = NumpyMatrixOperator(Cr)

    Wva = solve_sylv_schur(Aop, Arop, C=Cop, Cr=Crop)

    W = Wva.to_numpy().T

    ATW = A.T.dot(W)
    WAr = W.dot(Ar)
    CTCr = C.T.dot(Cr)
    assert fro_norm(ATW + WAr + CTCr) / fro_norm(CTCr) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('r', r_list)
@pytest.mark.parametrize('p', p_list)
def test_sylv_schur_W_E(n, r, p):
    np.random.seed(0)

    A, E = diff_conv_1d_fem(n, 1, 1)
    C = np.random.randn(p, n)

    Ar = np.random.randn(r, r) - r * np.eye(r)
    Er = np.random.randn(r, r)
    Er = (Er + Er.T) / 2
    Er += r * np.eye(r)
    Cr = np.random.randn(p, r)

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E)
    Cop = NumpyMatrixOperator(C)

    Arop = NumpyMatrixOperator(Ar)
    Erop = NumpyMatrixOperator(Er)
    Crop = NumpyMatrixOperator(Cr)

    Wva = solve_sylv_schur(Aop, Arop, E=Eop, Er=Erop, C=Cop, Cr=Crop)

    W = Wva.to_numpy().T

    ATWEr = A.T.dot(W.dot(Er))
    ETWAr = E.T.dot(W.dot(Ar))
    CTCr = C.T.dot(Cr)
    assert fro_norm(ATWEr + ETWAr + CTCr) / fro_norm(CTCr) < 1e-10
