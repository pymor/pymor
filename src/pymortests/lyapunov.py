# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy import stats

from pymor.core.config import config
from pymor.operators.numpy import NumpyMatrixOperator

import pytest


n_list = [200, 300]
m_list = [1, 2]
me_solver_list = ['scipy', 'slycot', 'pymess_lyap', 'pymess_lradi', 'lradi']
me_solver_E_list = ['slycot', 'pymess_lyap', 'pymess_lradi', 'lradi']


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


def _get_solve_lyap(me_solver):
    if me_solver == 'scipy':
        from pymor.bindings.scipy import solve_lyap
    elif me_solver == 'slycot':
        if not config.HAVE_SLYCOT:
            pytest.skip('slycot not available')
        from pymor.bindings.slycot import solve_lyap
    elif me_solver.startswith('pymess'):
        if not config.HAVE_PYMESS:
            pytest.skip('pymess not available')
        from pymor.bindings.pymess import solve_lyap
    elif me_solver == 'lradi':
        from pymor.algorithms.lyapunov import solve_lyap
    return solve_lyap


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('m', m_list)
@pytest.mark.parametrize('me_solver', me_solver_list)
def test_cgf_dense(n, m, me_solver):
    np.random.seed(0)
    A = np.random.randn(n, n) - n * np.eye(n)
    B = np.random.randn(n, m)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)

    solve_lyap = _get_solve_lyap(me_solver)

    Zva = solve_lyap(Aop, None, Bop, options={'type': me_solver})
    Z = Zva.to_numpy().T

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

    solve_lyap = _get_solve_lyap(me_solver)

    Zva = solve_lyap(Aop, Eop, Bop, options={'type': me_solver})
    Z = Zva.to_numpy().T

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

    solve_lyap = _get_solve_lyap(me_solver)

    Zva = solve_lyap(Aop, None, Bop, options={'type': me_solver})
    Z = Zva.to_numpy().T

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

    solve_lyap = _get_solve_lyap(me_solver)

    Zva = solve_lyap(Aop, Eop, Bop, options={'type': me_solver})
    Z = Zva.to_numpy().T

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

    solve_lyap = _get_solve_lyap(me_solver)

    Zva = solve_lyap(Aop, None, Cop, trans=True, options={'type': me_solver})
    Z = Zva.to_numpy().T

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

    solve_lyap = _get_solve_lyap(me_solver)

    Zva = solve_lyap(Aop, Eop, Cop, trans=True, options={'type': me_solver})
    Z = Zva.to_numpy().T

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

    solve_lyap = _get_solve_lyap(me_solver)

    Zva = solve_lyap(Aop, None, Cop, trans=True, options={'type': me_solver})
    Z = Zva.to_numpy().T

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

    solve_lyap = _get_solve_lyap(me_solver)

    Zva = solve_lyap(Aop, Eop, Cop, trans=True, options={'type': me_solver})
    Z = Zva.to_numpy().T

    assert len(Zva) <= n
    assert relative_residual(A, E, C, Z, trans=True) < 1e-10
