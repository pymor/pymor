# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.linalg as spla
from hypothesis import HealthCheck, assume, settings
from hypothesis.strategies import sampled_from

from pymor.algorithms.basic import almost_equal, contains_zero_vector
from pymor.algorithms.pod import pod
from pymor.core.logger import log_levels
from pymortests.strategies import given_vector_arrays

methods = ['method_of_snapshots', 'qr_svd']


@settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much,
          HealthCheck.too_slow, HealthCheck.data_too_large])
@given_vector_arrays(method=sampled_from(methods))
def test_pod(vector_array, method):
    A = vector_array
    # TODO assumption here masks a potential issue with the algorithm
    #      where it fails in internal lapack instead of a proper error
    # assumptions also necessitate the health check exemptions
    assume(len(A) > 1 or A.dim > 1)
    assume(not contains_zero_vector(A, rtol=1e-13, atol=1e-13))

    B = A.copy()
    orth_tol = 1e-10
    U, s = pod(A, method=method, orth_tol=orth_tol, return_reduced_coefficients=False)
    assert np.all(almost_equal(A, B))
    assert len(U) == len(s)
    assert np.allclose(U.gramian(), np.eye(len(s)), atol=orth_tol)


@settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much,
          HealthCheck.too_slow, HealthCheck.data_too_large])
@given_vector_arrays(method=sampled_from(methods))
def test_pod_with_coefficients(vector_array, method):
    A = vector_array
    # TODO assumption here masks a potential issue with the algorithm
    #      where it fails in internal lapack instead of a proper error
    # assumptions also necessitate the health check exemptions
    assume(len(A) > 1 or A.dim > 1)
    assume(not contains_zero_vector(A, rtol=1e-13, atol=1e-13))

    B = A.copy()
    orth_tol = 1e-10
    U, s, Vh = pod(A, method=method, orth_tol=orth_tol, return_reduced_coefficients=True)
    assert np.all(almost_equal(A, B))
    assert len(U) == len(s) == len(Vh)
    if len(s) > 0:
        U.scal(s)
        UsVh = U.lincomb(Vh.T)
        assert spla.norm((A - UsVh).norm()) / spla.norm(A.norm()) < 1e-7


@pytest.mark.parametrize('method', methods)
def test_pod_with_product(operator_with_arrays_and_products, method):
    _, _, A, _, p, _ = operator_with_arrays_and_products

    B = A.copy()
    with log_levels({'pymor.algorithms': 'ERROR'}):
        U, s = pod(A, product=p, method=method)
    assert np.all(almost_equal(A, B))
    assert len(U) == len(s)
    assert np.allclose(U.gramian(p), np.eye(len(s)))


@pytest.mark.parametrize('method', methods)
def test_pod_with_product_and_coefficients(operator_with_arrays_and_products, method):
    _, _, A, _, p, _ = operator_with_arrays_and_products

    B = A.copy()
    with log_levels({'pymor.algorithms': 'ERROR'}):
        U, s, Vh = pod(A, product=p, method=method, return_reduced_coefficients=True)
    assert np.all(almost_equal(A, B))
    assert len(U) == len(s) == len(Vh)
    if len(s) > 0:
        U.scal(s)
        UsVh = U.lincomb(Vh.T)
        assert spla.norm((A - UsVh).norm() / spla.norm(A.norm())) < 1e-7
