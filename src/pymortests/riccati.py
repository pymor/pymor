# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.algorithms.riccati import solve_ricc

import pytest


n_list = [10, 100]
m_list = [1, 2]
p_list = [1, 2]


def fro_norm(A):
    if not sps.issparse(A):
        return sp.linalg.norm(A)
    else:
        return sps.linalg.norm(A)


@pytest.mark.parametrize('n', n_list)
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

    Z = solve_ricc(Aop, B=Bop, C=Cop, R=Rop, meth='scipy')

    assert len(Z) <= n

    ATX = A.T.dot(Z.data.T).dot(Z.data)
    ZTB = Z.data.dot(B)
    XB = Z.data.T.dot(ZTB)
    RinvBTXT = spla.solve(R, XB.T)
    CTC = C.T.dot(C)
    assert fro_norm(ATX + ATX.T - XB.dot(RinvBTXT) + CTC) / fro_norm(CTC) < 1e-10


@pytest.mark.parametrize('n', n_list)
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

    Z = solve_ricc(Aop, B=Bop, C=Cop, R=Rop, trans=True, meth='scipy')

    assert len(Z) <= n

    AX = A.dot(Z.data.T).dot(Z.data)
    ZTCT = Z.data.dot(C.T)
    XCT = Z.data.T.dot(ZTCT)
    RinvCXT = spla.solve(R, XCT.T)
    BBT = B.dot(B.T)
    assert fro_norm(AX + AX.T - XCT.dot(RinvCXT) + BBT) / fro_norm(BBT) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
def test_pymess(n, m, p):
    np.random.seed(0)
    A = np.random.randn(n, n) - n * np.eye(n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)

    Z = solve_ricc(Aop, B=Bop, C=Cop, meth='pymess_care')

    assert len(Z) <= n

    ATX = A.T.dot(Z.data.T).dot(Z.data)
    XB = Z.data.T.dot(Z.data.dot(B))
    CTC = C.T.dot(C)
    assert fro_norm(ATX + ATX.T - XB.dot(XB.T) + CTC) / fro_norm(CTC) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
def test_pymess_trans(n, m, p):
    np.random.seed(0)
    A = np.random.randn(n, n) - n * np.eye(n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)

    Z = solve_ricc(Aop, B=Bop, C=Cop, trans=True, meth='pymess_care')

    assert len(Z) <= n

    AX = A.dot(Z.data.T).dot(Z.data)
    XCT = Z.data.T.dot(Z.data.dot(C.T))
    BBT = B.dot(B.T)
    assert fro_norm(AX + AX.T - XCT.dot(XCT.T) + BBT) / fro_norm(BBT) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
def test_pymess_E(n, m, p):
    np.random.seed(0)
    A = np.random.randn(n, n) - n * np.eye(n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    E = np.random.randn(n, n) + n * np.eye(n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)
    Eop = NumpyMatrixOperator(E)

    Z = solve_ricc(Aop, B=Bop, C=Cop, E=Eop, meth='pymess_care')

    assert len(Z) <= n

    ATZ = A.T.dot(Z.data.T)
    ZTE = Z.data.dot(E)
    ATXE = ATZ.dot(ZTE)
    ZTB = Z.data.dot(B)
    ETXB = E.T.dot(Z.data.T).dot(ZTB)
    CTC = C.T.dot(C)
    assert fro_norm(ATXE + ATXE.T - ETXB.dot(ETXB.T) + CTC) / fro_norm(CTC) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
def test_pymess_E_trans(n, m, p):
    np.random.seed(0)
    A = np.random.randn(n, n) - n * np.eye(n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    E = np.random.randn(n, n) + n * np.eye(n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)
    Eop = NumpyMatrixOperator(E)

    Z = solve_ricc(Aop, B=Bop, C=Cop, E=Eop, trans=True, meth='pymess_care')

    assert len(Z) <= n

    AZ = A.dot(Z.data.T)
    ZTET = Z.data.dot(E.T)
    AXET = AZ.dot(ZTET)
    ZTCT = Z.data.dot(C.T)
    EXCT = E.dot(Z.data.T).dot(ZTCT)
    BBT = B.dot(B.T)
    assert fro_norm(AXET + AXET.T - EXCT.dot(EXCT.T) + BBT) / fro_norm(BBT) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
def test_slycot(n, m, p):
    np.random.seed(0)
    A = np.random.randn(n, n) - n * np.eye(n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)

    Z = solve_ricc(Aop, B=Bop, C=Cop, meth='slycot')

    assert len(Z) <= n

    ATX = A.T.dot(Z.data.T).dot(Z.data)
    XB = Z.data.T.dot(Z.data.dot(B))
    CTC = C.T.dot(C)
    assert fro_norm(ATX + ATX.T - XB.dot(XB.T) + CTC) / fro_norm(CTC) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
def test_slycot_trans(n, m, p):
    np.random.seed(0)
    A = np.random.randn(n, n) - n * np.eye(n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)

    Z = solve_ricc(Aop, B=Bop, C=Cop, trans=True, meth='slycot')

    assert len(Z) <= n

    AX = A.dot(Z.data.T).dot(Z.data)
    XCT = Z.data.T.dot(Z.data.dot(C.T))
    BBT = B.dot(B.T)
    assert fro_norm(AX + AX.T - XCT.dot(XCT.T) + BBT) / fro_norm(BBT) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
def test_slycot_E(n, m, p):
    np.random.seed(0)
    A = np.random.randn(n, n) - n * np.eye(n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    E = np.random.randn(n, n) + n * np.eye(n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)
    Eop = NumpyMatrixOperator(E)

    Z = solve_ricc(Aop, B=Bop, C=Cop, E=Eop, meth='slycot')

    assert len(Z) <= n

    ATZ = A.T.dot(Z.data.T)
    ZTE = Z.data.dot(E)
    ATXE = ATZ.dot(ZTE)
    ZTB = Z.data.dot(B)
    ETXB = E.T.dot(Z.data.T).dot(ZTB)
    CTC = C.T.dot(C)
    assert fro_norm(ATXE + ATXE.T - ETXB.dot(ETXB.T) + CTC) / fro_norm(CTC) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('p', p_list)
def test_slycot_E_trans(n, m, p):
    np.random.seed(0)
    A = np.random.randn(n, n) - n * np.eye(n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    E = np.random.randn(n, n) + n * np.eye(n)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)
    Eop = NumpyMatrixOperator(E)

    Z = solve_ricc(Aop, B=Bop, C=Cop, E=Eop, trans=True, meth='slycot')

    assert len(Z) <= n

    AZ = A.dot(Z.data.T)
    ZTET = Z.data.dot(E.T)
    AXET = AZ.dot(ZTET)
    ZTCT = Z.data.dot(C.T)
    EXCT = E.dot(Z.data.T).dot(ZTCT)
    BBT = B.dot(B.T)
    assert fro_norm(AXET + AXET.T - EXCT.dot(EXCT.T) + BBT) / fro_norm(BBT) < 1e-10
