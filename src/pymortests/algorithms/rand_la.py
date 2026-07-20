# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.linalg as spla

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.rand_la import RandomizedRangeFinder, RandomizedSVD, randomized_ghep, randomized_svd
from pymor.algorithms.svd_va import SVD_VA_METHODS
from pymor.operators.constructions import VectorArrayOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.tools.random import new_rng

pytestmark = pytest.mark.builtin


@pytest.mark.parametrize('qr_method', ['gram_schmidt', 'shifted_chol_qr'])
@pytest.mark.parametrize('error_estimator', ['bs18', 'loo'])
def test_adaptive_rrf(rng, qr_method, error_estimator):
    if qr_method == 'shifted_chol_qr' and error_estimator == 'loo':
        pytest.xfail('Keeps failing in windows CI builds.')
    A  = rng.uniform(low=-1.0, high=1.0, size=(100, 10))
    op = NumpyMatrixOperator(A)

    B = A + 1j*rng.uniform(low=-1.0, high=1.0, size=(100, 10))
    B *= (10**(np.linspace(-1, -10, 10)))[np.newaxis, :]
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
    assert np.iscomplexobj(Q2.to_numpy())
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
    assert np.iscomplexobj(Q2.to_numpy())
    assert Q2 in op.range


@pytest.mark.parametrize('low_rank_svd_method', SVD_VA_METHODS)
@pytest.mark.parametrize('power_iterations', [0, 1, 2])
@pytest.mark.parametrize('oversampling', [1, 10])
@pytest.mark.parametrize('range_product', [True, None])
@pytest.mark.parametrize('source_product', [True, None])
def test_RandomizedSVD_repeated_calls(low_rank_svd_method, power_iterations, oversampling,
                                      range_product, source_product):
    if source_product and low_rank_svd_method == 'scipy_svd':
        pytest.skip('scipy_svd does not support products.')
    A = NumpyMatrixOperator(np.diag(np.linspace(1, 10, 100)))
    if range_product:
        range_product = NumpyMatrixOperator(np.diag(np.linspace(2, 1, 100)))
    if source_product:
        source_product = NumpyMatrixOperator(np.diag(np.linspace(3, 1, 100)))
    with new_rng():
        svd = RandomizedSVD(A, range_product=range_product, source_product=source_product,
                            low_rank_svd_method=low_rank_svd_method, power_iterations=power_iterations)
        svd.compute_svd(3)
        U1, s1, V1 = svd.compute_svd(4, oversampling=oversampling)
    with new_rng():
        svd = RandomizedSVD(A, range_product=range_product, source_product=source_product,
                            low_rank_svd_method=low_rank_svd_method, power_iterations=power_iterations)
        U2, s2, V2 = svd.compute_svd(4, oversampling=oversampling)

    assert np.allclose(s1, s2)
    assert np.all(almost_equal(U1, U2))
    assert np.all(almost_equal(V1, V2))


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
            assert np.linalg.norm(abs(V.to_numpy()[:, i]) - abs(V_real[:, i])) <= 1

    assert len(w) == n
    assert abs(np.linalg.norm(w - w_real[:n])) <= 1e-2
