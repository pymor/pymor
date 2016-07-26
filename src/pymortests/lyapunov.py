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


n_list = [50, 100]
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
            AZZT = A.apply(Z).data.T.dot(Z.data)
            BBT = B._matrix.dot(B._matrix.T)
            res = fro_norm(AZZT + AZZT.T + BBT)
            rhs = fro_norm(BBT)
        else:
            AZZTET = A.apply(Z).data.T.dot(E.apply(Z).data)
            BBT = B._matrix.dot(B._matrix.T)
            res = fro_norm(AZZTET + AZZTET.T + BBT)
            rhs = fro_norm(BBT)
    else:
        if E is None:
            ATZZT = A.apply_adjoint(Z).data.T.dot(Z.data)
            CTC = B._matrix.T.dot(B._matrix)
            res = fro_norm(ATZZT + ATZZT.T + CTC)
            rhs = fro_norm(CTC)
        else:
            ATZZTE = A.apply_adjoint(Z).data.T.dot(E.apply_adjoint(Z).data)
            CTC = B._matrix.T.dot(B._matrix)
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

    A = NumpyMatrixOperator(A)
    B = NumpyMatrixOperator(B)

    Z = solve_lyap(A, None, B, me_solver=me_solver)

    assert len(Z) <= n
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

    A = NumpyMatrixOperator(A)
    E = NumpyMatrixOperator(E)
    B = NumpyMatrixOperator(B)

    Z = solve_lyap(A, E, B, me_solver=me_solver)

    assert len(Z) <= n
    assert relative_residual(A, E, B, Z) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('me_solver', me_solver_list)
def test_cgf_sparse(n, m, me_solver):
    np.random.seed(0)
    A = sps.random(n, n, density=5 / n, format='csc', data_rvs=stats.norm().rvs)
    A -= n * sps.eye(n)
    B = sps.random(n, m, density=5 / n, format='csc', data_rvs=stats.norm().rvs)

    A = NumpyMatrixOperator(A)
    B = NumpyMatrixOperator(B)

    Z = solve_lyap(A, None, B, me_solver=me_solver)

    assert len(Z) <= n
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

    A = NumpyMatrixOperator(A)
    E = NumpyMatrixOperator(E)
    B = NumpyMatrixOperator(B)

    Z = solve_lyap(A, E, B, me_solver=me_solver)

    assert len(Z) <= n
    assert relative_residual(A, E, B, Z) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('p', m_list)
@pytest.mark.parametrize('me_solver', me_solver_list)
def test_ogf_dense(n, p, me_solver):
    np.random.seed(0)
    A = np.random.randn(n, n) - n * np.eye(n)
    C = np.random.randn(p, n)

    A = NumpyMatrixOperator(A)
    C = NumpyMatrixOperator(C)

    Z = solve_lyap(A, None, C, trans=True, me_solver=me_solver)

    assert len(Z) <= n
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

    A = NumpyMatrixOperator(A)
    E = NumpyMatrixOperator(E)
    C = NumpyMatrixOperator(C)

    Z = solve_lyap(A, E, C, trans=True, me_solver=me_solver)

    assert len(Z) <= n
    assert relative_residual(A, E, C, Z, trans=True) < 1e-10


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('p', m_list)
@pytest.mark.parametrize('me_solver', me_solver_list)
def test_ogf_sparse(n, p, me_solver):
    np.random.seed(0)
    A = sps.random(n, n, density=5 / n, format='csc', data_rvs=stats.norm().rvs)
    A -= n * sps.eye(n)
    C = sps.random(p, n, density=5 / n, format='csc', data_rvs=stats.norm().rvs)

    A = NumpyMatrixOperator(A)
    C = NumpyMatrixOperator(C)

    Z = solve_lyap(A, None, C, trans=True, me_solver=me_solver)

    assert len(Z) <= n
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

    A = NumpyMatrixOperator(A)
    E = NumpyMatrixOperator(E)
    C = NumpyMatrixOperator(C)

    Z = solve_lyap(A, E, C, trans=True, me_solver=me_solver)

    assert len(Z) <= n
    assert relative_residual(A, E, C, Z, trans=True) < 1e-10
