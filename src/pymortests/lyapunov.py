# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy import stats

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.algorithms.lyapunov import solve_lyap

import pytest


n_list = [200, 300]
m_list = [1, 2]
me_solver_list = ['scipy', 'slycot', 'pymess_lyap', 'pymess_lradi']
me_solver_E_list = ['slycot', 'pymess_lyap', 'pymess_lradi']


def fro_norm(A):
    if not sps.issparse(A):
        return sp.linalg.norm(A)
    else:
        return sps.linalg.norm(A)


def relative_residual(A, E, B, Z, trans=False):
    if not trans:
        if E is None:
            AZZT = A.dot(Z).dot(Z.T)
            BBT = B.dot(B.T)
            res = fro_norm(AZZT + AZZT.T + BBT)
            rhs = fro_norm(BBT)
        else:
            AZZTET = A.dot(Z).dot(E.dot(Z).T)
            BBT = B.dot(B.T)
            res = fro_norm(AZZTET + AZZTET.T + BBT)
            rhs = fro_norm(BBT)
    else:
        if E is None:
            ATZZT = A.T.dot(Z).dot(Z.T)
            CTC = B.T.dot(B)
            res = fro_norm(ATZZT + ATZZT.T + CTC)
            rhs = fro_norm(CTC)
        else:
            ATZZTE = A.T.dot(Z).dot(E.T.dot(Z).T)
            CTC = B.T.dot(B)
            res = fro_norm(ATZZTE + ATZZTE.T + CTC)
            rhs = fro_norm(CTC)
    return res / rhs


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('me_solver', me_solver_list)
def test_cgf_dense(n, m, me_solver):
    np.random.seed(0)
    A = np.random.randn(n, n) - n * np.eye(n)
    B = np.random.randn(n, m)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)

    Zva = solve_lyap(Aop, None, Bop, me_solver=me_solver)
    Z = Zva.data.T

    assert len(Zva) <= n
    assert relative_residual(A, None, B, Z) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('me_solver', me_solver_E_list)
def test_cgf_dense_E(n, m, me_solver):
    np.random.seed(0)
    A = np.random.randn(n, n)
    A = (A + A.T) / 2
    A -= n * np.eye(n)

    E = np.random.randn(n, n)
    E = (E + E.T) / 2
    E += n * np.eye(n)

    B = np.random.randn(n, m)

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E)
    Bop = NumpyMatrixOperator(B)

    Zva = solve_lyap(Aop, Eop, Bop, me_solver=me_solver)
    Z = Zva.data.T

    assert len(Zva) <= n
    assert relative_residual(A, E, B, Z) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('me_solver', me_solver_list)
def test_cgf_sparse(n, m, me_solver):
    np.random.seed(0)
    A = sps.random(n, n, density=5 / n, format='csc', data_rvs=stats.norm().rvs)
    A -= n * sps.eye(n)
    B = sps.random(n, m, density=5 / n, format='csc', data_rvs=stats.norm().rvs)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)

    Zva = solve_lyap(Aop, None, Bop, me_solver=me_solver)
    Z = Zva.data.T

    assert len(Zva) <= n
    assert relative_residual(A, None, B, Z) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('me_solver', me_solver_E_list)
def test_cgf_sparse_E(n, m, me_solver):
    np.random.seed(0)
    A = sps.random(n, n, density=5 / n, format='csc', data_rvs=stats.norm().rvs)
    A = (A + A.T) / 2
    A -= n * sps.eye(n)

    E = sps.random(n, n, density=5 / n, format='csc', data_rvs=stats.norm().rvs)
    E = (E + E.T) / 2
    E += n * sps.eye(n)

    B = sps.random(n, m, density=5 / n, format='csc', data_rvs=stats.norm().rvs)

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E)
    Bop = NumpyMatrixOperator(B)

    Zva = solve_lyap(Aop, Eop, Bop, me_solver=me_solver)
    Z = Zva.data.T

    assert len(Zva) <= n
    assert relative_residual(A, E, B, Z) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('p', m_list)
@pytest.mark.parametrize('me_solver', me_solver_list)
def test_ogf_dense(n, p, me_solver):
    np.random.seed(0)
    A = np.random.randn(n, n) - n * np.eye(n)
    C = np.random.randn(p, n)

    Aop = NumpyMatrixOperator(A)
    Cop = NumpyMatrixOperator(C)

    Zva = solve_lyap(Aop, None, Cop, trans=True, me_solver=me_solver)
    Z = Zva.data.T

    assert len(Zva) <= n
    assert relative_residual(A, None, C, Z, trans=True) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('p', m_list)
@pytest.mark.parametrize('me_solver', me_solver_E_list)
def test_ogf_dense_E(n, p, me_solver):
    np.random.seed(0)
    A = np.random.randn(n, n)
    A = (A + A.T) / 2
    A -= n * np.eye(n)

    E = np.random.randn(n, n)
    E = (E + E.T) / 2
    E += n * np.eye(n)

    C = np.random.randn(p, n)

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E)
    Cop = NumpyMatrixOperator(C)

    Zva = solve_lyap(Aop, Eop, Cop, trans=True, me_solver=me_solver)
    Z = Zva.data.T

    assert len(Zva) <= n
    assert relative_residual(A, E, C, Z, trans=True) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('p', m_list)
@pytest.mark.parametrize('me_solver', me_solver_list)
def test_ogf_sparse(n, p, me_solver):
    np.random.seed(0)
    A = sps.random(n, n, density=5 / n, format='csc', data_rvs=stats.norm().rvs)
    A -= n * sps.eye(n)
    C = sps.random(p, n, density=5 / n, format='csc', data_rvs=stats.norm().rvs)

    Aop = NumpyMatrixOperator(A)
    Cop = NumpyMatrixOperator(C)

    Zva = solve_lyap(Aop, None, Cop, trans=True, me_solver=me_solver)
    Z = Zva.data.T

    assert len(Zva) <= n
    assert relative_residual(A, None, C, Z, trans=True) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('p', m_list)
@pytest.mark.parametrize('me_solver', me_solver_E_list)
def test_ogf_sparse_E(n, p, me_solver):
    np.random.seed(0)
    A = sps.random(n, n, density=5 / n, format='csc', data_rvs=stats.norm().rvs)
    A = (A + A.T) / 2
    A -= n * sps.eye(n)

    E = sps.random(n, n, density=5 / n, format='csc', data_rvs=stats.norm().rvs)
    E = (E + E.T) / 2
    E += n * sps.eye(n)

    C = sps.random(p, n, density=5 / n, format='csc', data_rvs=stats.norm().rvs)

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E)
    Cop = NumpyMatrixOperator(C)

    Zva = solve_lyap(Aop, Eop, Cop, trans=True, me_solver=me_solver)
    Z = Zva.data.T

    assert len(Zva) <= n
    assert relative_residual(A, E, C, Z, trans=True) < 1e-10
