# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.core.config import config
from pymor.operators.numpy import NumpyMatrixOperator

import pytest


n_list_small = [10, 100]
n_list_big = [1000]
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


@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
def test_scipy(n, m, p):
    np.random.seed(0)
    A = np.random.randn(n, n) - n * np.eye(n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    R = np.random.randn(m, m)
    R = (R + R.T) / 2

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)
    Rop = NumpyMatrixOperator(R)

    from pymor.bindings.scipy import solve_ricc
    Z = solve_ricc(Aop, B=Bop, C=Cop, R=Rop)

    assert len(Z) <= n

    ATX = A.T.dot(Z.data.T).dot(Z.data)
    ZTB = Z.data.dot(B)
    XB = Z.data.T.dot(ZTB)
    RinvBTXT = spla.solve(R, XB.T)
    CTC = C.T.dot(C)
    assert fro_norm(ATX + ATX.T - XB.dot(RinvBTXT) + CTC) / fro_norm(CTC) < 1e-10


@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
def test_scipy_trans(n, m, p):
    np.random.seed(0)
    A = np.random.randn(n, n) - n * np.eye(n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    R = np.random.randn(p, p)
    R = (R + R.T) / 2

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)
    Rop = NumpyMatrixOperator(R)

    from pymor.bindings.scipy import solve_ricc
    Z = solve_ricc(Aop, B=Bop, C=Cop, R=Rop, trans=True)

    assert len(Z) <= n

    AX = A.dot(Z.data.T).dot(Z.data)
    ZTCT = Z.data.dot(C.T)
    XCT = Z.data.T.dot(ZTCT)
    RinvCXT = spla.solve(R, XCT.T)
    BBT = B.dot(B.T)
    assert fro_norm(AX + AX.T - XCT.dot(RinvCXT) + BBT) / fro_norm(BBT) < 1e-10


@pytest.mark.skipif(not config.HAVE_PYMESS, reason='pymess not available')
@pytest.mark.parametrize('n', n_list_big)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
@pytest.mark.parametrize('me_solver', ['pymess_care', 'pymess_lrnm'])
def test_pymess(n, m, p, me_solver):
    np.random.seed(0)
    A = diff_conv_1d_fd(n, 1, 1)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)

    from pymor.bindings.pymess import solve_ricc
    Z = solve_ricc(Aop, B=Bop, C=Cop, default_solver=me_solver)

    assert len(Z) <= n

    ATX = A.T.dot(Z.data.T).dot(Z.data)
    XB = Z.data.T.dot(Z.data.dot(B))
    CTC = C.T.dot(C)
    assert fro_norm(ATX + ATX.T - XB.dot(XB.T) + CTC) / fro_norm(CTC) < 1e-10


@pytest.mark.skipif(not config.HAVE_PYMESS, reason='pymess not available')
@pytest.mark.parametrize('n', n_list_big)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
@pytest.mark.parametrize('me_solver', ['pymess_care', 'pymess_lrnm'])
def test_pymess_trans(n, m, p, me_solver):
    np.random.seed(0)
    A = diff_conv_1d_fd(n, 1, 1)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)

    from pymor.bindings.pymess import solve_ricc
    Z = solve_ricc(Aop, B=Bop, C=Cop, trans=True, default_solver=me_solver)

    assert len(Z) <= n

    AX = A.dot(Z.data.T).dot(Z.data)
    XCT = Z.data.T.dot(Z.data.dot(C.T))
    BBT = B.dot(B.T)
    assert fro_norm(AX + AX.T - XCT.dot(XCT.T) + BBT) / fro_norm(BBT) < 1e-10


@pytest.mark.skipif(not config.HAVE_PYMESS, reason='pymess not available')
@pytest.mark.parametrize('n', n_list_big)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
@pytest.mark.parametrize('me_solver', ['pymess_care', 'pymess_lrnm'])
def test_pymess_E(n, m, p, me_solver):
    np.random.seed(0)
    A, E = diff_conv_1d_fem(n, 1, 1)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)
    Eop = NumpyMatrixOperator(E)

    from pymor.bindings.pymess import solve_ricc
    Z = solve_ricc(Aop, B=Bop, C=Cop, E=Eop, default_solver=me_solver)

    assert len(Z) <= n

    ATZ = A.T.dot(Z.data.T)
    ZTE = E.T.dot(Z.data.T).T
    ATXE = ATZ.dot(ZTE)
    ZTB = Z.data.dot(B)
    ETXB = E.T.dot(Z.data.T).dot(ZTB)
    CTC = C.T.dot(C)
    assert fro_norm(ATXE + ATXE.T - ETXB.dot(ETXB.T) + CTC) / fro_norm(CTC) < 1e-10


@pytest.mark.skipif(not config.HAVE_PYMESS, reason='pymess not available')
@pytest.mark.parametrize('n', n_list_big)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
@pytest.mark.parametrize('me_solver', ['pymess_care', 'pymess_lrnm'])
def test_pymess_E_trans(n, m, p, me_solver):
    np.random.seed(0)
    A, E = diff_conv_1d_fem(n, 1, 1)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)
    Eop = NumpyMatrixOperator(E)

    from pymor.bindings.pymess import solve_ricc
    Z = solve_ricc(Aop, B=Bop, C=Cop, E=Eop, trans=True, default_solver=me_solver)

    assert len(Z) <= n

    AZ = A.dot(Z.data.T)
    ZTET = E.dot(Z.data.T).T
    AXET = AZ.dot(ZTET)
    ZTCT = Z.data.dot(C.T)
    EXCT = E.dot(Z.data.T).dot(ZTCT)
    BBT = B.dot(B.T)
    assert fro_norm(AXET + AXET.T - EXCT.dot(EXCT.T) + BBT) / fro_norm(BBT) < 1e-10


@pytest.mark.skipif(not config.HAVE_SLYCOT, reason='slycot not available')
@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
def test_slycot(n, m, p):
    np.random.seed(0)
    A = diff_conv_1d_fd(n, 1, 1)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)

    from pymor.bindings.slycot import solve_ricc
    Z = solve_ricc(Aop, B=Bop, C=Cop)

    assert len(Z) <= n

    ATX = A.T.dot(Z.data.T).dot(Z.data)
    XB = Z.data.T.dot(Z.data.dot(B))
    CTC = C.T.dot(C)
    assert fro_norm(ATX + ATX.T - XB.dot(XB.T) + CTC) / fro_norm(CTC) < 1e-8


@pytest.mark.skipif(not config.HAVE_SLYCOT, reason='slycot not available')
@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
def test_slycot_trans(n, m, p):
    np.random.seed(0)
    A = diff_conv_1d_fd(n, 1, 1)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)

    from pymor.bindings.slycot import solve_ricc
    Z = solve_ricc(Aop, B=Bop, C=Cop, trans=True)

    assert len(Z) <= n

    AX = A.dot(Z.data.T).dot(Z.data)
    XCT = Z.data.T.dot(Z.data.dot(C.T))
    BBT = B.dot(B.T)
    assert fro_norm(AX + AX.T - XCT.dot(XCT.T) + BBT) / fro_norm(BBT) < 1e-8


@pytest.mark.skipif(not config.HAVE_SLYCOT, reason='slycot not available')
@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
def test_slycot_E(n, m, p):
    np.random.seed(0)
    A, E = diff_conv_1d_fem(n, 1, 1)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)
    Eop = NumpyMatrixOperator(E)

    from pymor.bindings.slycot import solve_ricc
    Z = solve_ricc(Aop, B=Bop, C=Cop, E=Eop)

    assert len(Z) <= n

    ATZ = A.T.dot(Z.data.T)
    ZTE = E.T.dot(Z.data.T).T
    ATXE = ATZ.dot(ZTE)
    ZTB = Z.data.dot(B)
    ETXB = E.T.dot(Z.data.T).dot(ZTB)
    CTC = C.T.dot(C)
    assert fro_norm(ATXE + ATXE.T - ETXB.dot(ETXB.T) + CTC) / fro_norm(CTC) < 1e-8


@pytest.mark.skipif(not config.HAVE_SLYCOT, reason='slycot not available')
@pytest.mark.parametrize('n', n_list_small)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
def test_slycot_E_trans(n, m, p):
    np.random.seed(0)
    A, E = diff_conv_1d_fem(n, 1, 1)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)
    Eop = NumpyMatrixOperator(E)

    from pymor.bindings.slycot import solve_ricc
    Z = solve_ricc(Aop, B=Bop, C=Cop, E=Eop, trans=True)

    assert len(Z) <= n

    AZ = A.dot(Z.data.T)
    ZTET = E.dot(Z.data.T).T
    AXET = AZ.dot(ZTET)
    ZTCT = Z.data.dot(C.T)
    EXCT = E.dot(Z.data.T).dot(ZTCT)
    BBT = B.dot(B.T)
    assert fro_norm(AXET + AXET.T - EXCT.dot(EXCT.T) + BBT) / fro_norm(BBT) < 1e-8
