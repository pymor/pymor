# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.linalg as spla

from pymor.algorithms.rand_la import RandomizedRangeFinder, randomized_ghep, randomized_svd
from pymor.operators.constructions import VectorArrayOperator
from pymor.operators.numpy import NumpyMatrixOperator

pytestmark = pytest.mark.builtin


@pytest.mark.parametrize('qr_method', ['gram_schmidt', 'shifted_chol_qr'])
@pytest.mark.parametrize('error_estimator', ['bs18', 'loo'])
def test_adaptive_rrf(rng, qr_method, error_estimator):
    A  = rng.uniform(low=-1.0, high=1.0, size=(100, 10))
    op = NumpyMatrixOperator(A)

    B = A + 1j*rng.uniform(low=-1.0, high=1.0, size=(100, 10))
    op_complex = NumpyMatrixOperator(B)

    Q1 = RandomizedRangeFinder(
        op,
        qr_method=qr_method,
        error_estimator=error_estimator
    ).find_range(tol=1e-5)
    assert Q1 in op.range

    Q2 = RandomizedRangeFinder(
        op_complex,
        iscomplex=True,
        qr_method=qr_method,
        error_estimator=error_estimator
    ).find_range(tol=1e-5)
    assert np.iscomplexobj(Q2.to_numpy_TP())
    assert Q2 in op.range


@pytest.mark.parametrize('qr_method', ['gram_schmidt', 'shifted_chol_qr'])
def test_adaptive_rrf_with_product(rng, qr_method):
    A = rng.uniform(low=-1.0, high=1.0, size=(100, 100))
    A = A @ A.T
    range_product = NumpyMatrixOperator(A)

    B = rng.uniform(low=-1.0, high=1.0, size=(10, 10))
    B = B.dot(B.T)
    source_product = NumpyMatrixOperator(B)

    C = range_product.range.random(10)
    op = VectorArrayOperator(C)

    D = range_product.range.random(10)
    D += 1j*range_product.range.random(10)
    op_complex = VectorArrayOperator(D)

    Q1 = RandomizedRangeFinder(
        op, range_product=range_product, source_product=source_product, qr_method=qr_method,
    ).find_range(tol=1e-5)
    assert Q1 in op.range

    Q2 = RandomizedRangeFinder(
        op_complex, range_product=range_product, source_product=source_product, qr_method=qr_method,
    ).find_range(tol=1e-5)
    assert np.iscomplexobj(Q2.to_numpy_TP())
    assert Q2 in op.range


def test_random_generalized_svd(rng):
    E = rng.uniform(low=-1.0, high=1.0, size=(5, 5))
    E_op = NumpyMatrixOperator(E)

    n = 3
    U, s, Vh = randomized_svd(E_op, n=n, oversampling=1, power_iterations=2)
    U_real, s_real, Vh_real = spla.svd(E)

    assert abs(np.linalg.norm(s-s_real[:n])) <= 1e-2
    assert len(U) == n
    assert len(Vh) == n
    assert len(s) == n
    assert U in E_op.range
    assert Vh in E_op.source


@pytest.mark.parametrize('return_evecs', [False, True])
@pytest.mark.parametrize('single_pass', [False, True])
def test_randomized_ghep(rng, return_evecs, single_pass):
    n = 3
    W = rng.uniform(low=-1.0, high=1.0, size=(5, 5))
    op = NumpyMatrixOperator(W @ W.T)

    w_real, V_real = spla.eigh(op.matrix)
    w_real = w_real[::-1]
    V_real = V_real[:, ::-1]

    w = randomized_ghep(op, n=n, power_iterations=1, single_pass=single_pass, return_evecs=return_evecs)
    if return_evecs:
        w, V = w[0], w[1]
        assert len(V) == n
        assert V.dim == op.source.dim
        for i in range(0, n):
            assert np.linalg.norm(abs(V.to_numpy_TP()[:, i]) - abs(V_real[:, i])) <= 1

    assert len(w) == n
    assert abs(np.linalg.norm(w - w_real[:n])) <= 1e-2
