# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy import stats

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.algorithms.lyapunov import solve_lyap


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


def test_cgf_dense():
    for n in (10, 500, 1200):
        for m in (1, 2):
            np.random.seed(1)
            A = np.random.randn(n, n) - n * np.eye(n)
            B = np.random.randn(n, m)

            A = NumpyMatrixOperator(A)
            B = NumpyMatrixOperator(B)

            Z = solve_lyap(A, None, B)

            assert relative_residual(A, None, B, Z) < 1e-10


def test_cgf_dense_E():
    np.random.seed(1)
    for n in (10, 500, 1200):
        for m in (1, 2):
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

            Z = solve_lyap(A, E, B)

            assert relative_residual(A, E, B, Z) < 1e-10


def test_cgf_sparse():
    np.random.seed(1)
    for n in (10, 500, 1200):
        for m in (1, 2):
            A = sps.random(n, n, density=5 / n, format='csc', data_rvs=stats.norm().rvs)
            A -= n * sps.eye(n)
            B = sps.random(n, m, density=5 / n, format='csc', data_rvs=stats.norm().rvs)

            A = NumpyMatrixOperator(A)
            B = NumpyMatrixOperator(B)

            Z = solve_lyap(A, None, B)

            assert relative_residual(A, None, B, Z) < 1e-10


def test_cgf_sparse_E():
    np.random.seed(1)
    for n in (10, 500, 1200):
        for m in (1, 2):
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

            Z = solve_lyap(A, E, B)

            assert relative_residual(A, E, B, Z) < 1e-10


def test_ogf_dense():
    np.random.seed(1)
    for n in (100, 500, 1200):
        for p in (1, 2):
            A = np.random.randn(n, n) - n * np.eye(n)
            C = np.random.randn(p, n)

            A = NumpyMatrixOperator(A)
            C = NumpyMatrixOperator(C)

            Z = solve_lyap(A, None, C, trans=True)

            assert relative_residual(A, None, C, Z, trans=True) < 1e-10


def test_ogf_dense_E():
    np.random.seed(1)
    for n in (10, 500, 1200):
        for p in (1, 2):
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

            Z = solve_lyap(A, E, C, trans=True)

            assert relative_residual(A, E, C, Z, trans=True) < 1e-10


def test_ogf_sparse():
    np.random.seed(1)
    for n in (10, 500, 1200):
        for p in (1, 2):
            A = sps.random(n, n, density=5 / n, format='csc', data_rvs=stats.norm().rvs)
            A -= n * sps.eye(n)
            C = sps.random(p, n, density=5 / n, format='csc', data_rvs=stats.norm().rvs)

            A = NumpyMatrixOperator(A)
            C = NumpyMatrixOperator(C)

            Z = solve_lyap(A, None, C, trans=True)

            assert relative_residual(A, None, C, Z, trans=True) < 1e-10


def test_ogf_sparse_E():
    np.random.seed(1)
    for n in (10, 500, 1200):
        for p in (1, 2):
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

            Z = solve_lyap(A, E, C, trans=True)

            assert relative_residual(A, E, C, Z, trans=True) < 1e-10
