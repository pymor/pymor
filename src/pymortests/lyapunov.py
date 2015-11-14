# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse as sps
from scipy import stats

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorArray

from pymor.algorithms.lyapunov import solve_lyap

def test_cgf_dense():
    np.random.seed(1)
    for n in (10, 500, 1200):
        for m in (1, 2):
            A = np.random.randn(n, n) - n * np.eye(n)
            B = np.random.randn(n, m)

            A = NumpyMatrixOperator(A)
            B = NumpyVectorArray(B.T)

            Z = solve_lyap(A, None, B)

            AZZT = A.apply(Z).data.T.dot(Z.data)
            BBT = B.data.T.dot(B.data)

            assert np.linalg.norm(AZZT + AZZT.T + BBT) / np.linalg.norm(BBT) < 1e-10

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
            B = NumpyVectorArray(B.T)

            Z = solve_lyap(A, E, B)

            AZZTET = A.apply(Z).data.T.dot(E.apply(Z).data)
            BBT = B.data.T.dot(B.data)

            assert np.linalg.norm(AZZTET + AZZTET.T + BBT) / np.linalg.norm(BBT) < 1e-10

def test_cgf_sparse():
    np.random.seed(1)
    for n in (10, 500, 1200):
        for m in (1, 2):
            A = sps.random(n, n, density=5/n, format='csc', data_rvs=stats.norm().rvs)
            A -= n * sps.eye(n)
            B = sps.random(n, m, density=5/n, format='csc', data_rvs=stats.norm().rvs)

            A = NumpyMatrixOperator(A)
            B = NumpyVectorArray(B.T)

            Z = solve_lyap(A, None, B)

            AZZT = A.apply(Z).data.T.dot(Z.data)
            BBT = B.data.T.dot(B.data)

            assert np.linalg.norm(AZZT + AZZT.T + BBT) / np.linalg.norm(BBT) < 1e-10

def test_cgf_sparse_E():
    np.random.seed(1)
    for n in (10, 500, 1200):
        for m in (1, 2):
            A = sps.random(n, n, density=5/n, format='csc', data_rvs=stats.norm().rvs)
            A = (A + A.T) / 2
            A -= n * sps.eye(n)

            E = sps.random(n, n, density=5/n, format='csc', data_rvs=stats.norm().rvs)
            E = (E + E.T) / 2
            E += n * sps.eye(n)

            B = sps.random(n, m, density=5/n, format='csc', data_rvs=stats.norm().rvs)

            A = NumpyMatrixOperator(A)
            E = NumpyMatrixOperator(E)
            B = NumpyVectorArray(B.T)

            Z = solve_lyap(A, E, B)

            AZZTET = A.apply(Z).data.T.dot(E.apply(Z).data)
            BBT = B.data.T.dot(B.data)

            assert np.linalg.norm(AZZTET + AZZTET.T + BBT) / np.linalg.norm(BBT) < 1e-10

def test_ogf_dense():
    np.random.seed(1)
    for n in (100, 500, 1200):
        for p in (1, 2):
            A = np.random.randn(n, n) - n * np.eye(n)
            C = np.random.randn(p, n)

            A = NumpyMatrixOperator(A)
            C = NumpyVectorArray(C)

            Z = solve_lyap(A, None, C, trans=True)

            ATZZT = A.apply_adjoint(Z).data.T.dot(Z.data)
            CTC = C.data.T.dot(C.data)

            assert np.linalg.norm(ATZZT + ATZZT.T + CTC) / np.linalg.norm(CTC) < 1e-10

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
            C = NumpyVectorArray(C)

            Z = solve_lyap(A, E, C, trans=True)

            ATZZTE = A.apply_adjoint(Z).data.T.dot(E.apply_adjoint(Z).data)
            CTC = C.data.T.dot(C.data)

            assert np.linalg.norm(ATZZTE + ATZZTE.T + CTC) / np.linalg.norm(CTC) < 1e-10

def test_ogf_sparse():
    np.random.seed(1)
    for n in (10, 500, 1200):
        for p in (1, 2):
            A = sps.random(n, n, density=5/n, format='csc', data_rvs=stats.norm().rvs)
            A -= n * sps.eye(n)
            C = sps.random(p, n, density=5/n, format='csc', data_rvs=stats.norm().rvs)

            A = NumpyMatrixOperator(A)
            C = NumpyVectorArray(C)

            Z = solve_lyap(A, None, C, trans=True)

            ATZZT = A.apply_adjoint(Z).data.T.dot(Z.data)
            CTC = C.data.T.dot(C.data)

            assert np.linalg.norm(ATZZT + ATZZT.T + CTC) / np.linalg.norm(CTC) < 1e-10

def test_cgf_sparse_E():
    np.random.seed(1)
    for n in (10, 500, 1200):
        for p in (1, 2):
            A = sps.random(n, n, density=5/n, format='csc', data_rvs=stats.norm().rvs)
            A = (A + A.T) / 2
            A -= n * sps.eye(n)

            E = sps.random(n, n, density=5/n, format='csc', data_rvs=stats.norm().rvs)
            E = (E + E.T) / 2
            E += n * sps.eye(n)

            C = sps.random(p, n, density=5/n, format='csc', data_rvs=stats.norm().rvs)

            A = NumpyMatrixOperator(A)
            E = NumpyMatrixOperator(E)
            C = NumpyVectorArray(C)

            Z = solve_lyap(A, E, C, trans=True)

            ATZZTE = A.apply_adjoint(Z).data.T.dot(E.apply_adjoint(Z).data)
            CTC = C.data.T.dot(C.data)

            assert np.linalg.norm(ATZZTE + ATZZTE.T + CTC) / np.linalg.norm(CTC) < 1e-10
