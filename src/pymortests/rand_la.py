# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy as sp
from numpy.random import uniform

from pymor.algorithms.rand_la import rrf, adaptive_rrf, random_ghep, random_generalized_svd
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import VectorArrayOperator


def test_adaptive_rrf():
    np.random.seed(0)
    A = uniform(low=-1.0, high=1.0, size=(100, 100))
    A = A @ A.T
    range_product = NumpyMatrixOperator(A)

    np.random.seed(1)
    B = uniform(low=-1.0, high=1.0, size=(10, 10))
    B = B.dot(B.T)
    source_product = NumpyMatrixOperator(B)

    C = range_product.range.random(10, seed=10)
    op = VectorArrayOperator(C)

    D = range_product.range.random(10, seed=11)+1j*range_product.range.random(10, seed=12)
    op_complex = VectorArrayOperator(D)

    Q1 = adaptive_rrf(op, source_product, range_product)
    assert Q1 in op.range

    Q2 = adaptive_rrf(op_complex, iscomplex=True)
    assert np.iscomplexobj(Q2.to_numpy())
    assert Q2 in op.range


def test_rrf():
    np.random.seed(2)
    A = uniform(low=-1.0, high=1.0, size=(100, 100))
    A = A @ A.T
    range_product = NumpyMatrixOperator(A)

    np.random.seed(3)
    B = uniform(low=-1.0, high=1.0, size=(10, 10))
    B = B @ B.T
    source_product = NumpyMatrixOperator(B)

    C = range_product.range.random(10, seed=10)
    op = VectorArrayOperator(C)

    D = range_product.range.random(10, seed=11)+1j*range_product.range.random(10, seed=12)
    op_complex = VectorArrayOperator(D)

    Q1 = rrf(op, source_product, range_product)
    assert Q1 in op.range
    assert len(Q1) == 8

    Q2 = rrf(op_complex, iscomplex=True)
    assert np.iscomplexobj(Q2.to_numpy())
    assert Q2 in op.range
    assert len(Q2) == 8


def test_random_generalized_svd():
    np.random.seed(4)
    E = uniform(low=-1.0, high=1.0, size=(5, 5))
    E_op = NumpyMatrixOperator(E)

    modes = 3
    U, s, Vh = random_generalized_svd(E_op, modes=modes, p=1)
    U_real, s_real, Vh_real = sp.linalg.svd(E)

    assert abs(np.linalg.norm(s-s_real[:modes])) <= 1e-2
    assert len(U) == modes
    assert len(Vh) == modes
    assert len(s) == modes
    assert U in E_op.range
    assert Vh in E_op.source


def test_random_ghep():
    np.random.seed(5)
    D = uniform(low=-1.0, high=1.0, size=(5, 5))
    D = D @ D.T
    D_op = NumpyMatrixOperator(D)

    modes = 3
    w1, V1 = random_ghep(D_op, modes=modes, p=1, single_pass=False)
    w2, V2 = random_ghep(D_op, modes=modes, p=1, single_pass=True)
    w_real, V_real = sp.linalg.eigh(D)
    w_real = w_real[::-1]
    V_real = V_real[:, ::-1]

    assert abs(np.linalg.norm(w1-w_real[:modes])) <= 1e-2
    assert abs(np.linalg.norm(w2-w_real[:modes])) <= 1

    for i in range(0, modes):
        assert np.linalg.norm(abs(V1.to_numpy()[i, :]) - abs(V_real[:, i])) <= 1
    for i in range(0, modes):
        assert np.linalg.norm(abs(V2.to_numpy()[i, :]) - abs(V_real[:, i])) <= 1

    assert len(w1) == modes
    assert len(V1) == modes
    assert V1.dim == D_op.source.dim
