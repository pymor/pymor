# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from hypothesis import assume, settings, HealthCheck
from hypothesis.strategies import sampled_from

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.pod import pod
from pymor.algorithms.basic import contains_zero_vector
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
    U, s = pod(A, method=method, orth_tol=orth_tol)
    assert np.all(almost_equal(A, B))
    assert len(U) == len(s)
    assert np.allclose(U.gramian(), np.eye(len(s)), atol=orth_tol)


@pytest.mark.parametrize('method', methods)
def test_pod_with_product(operator_with_arrays_and_products, method):
    _, _, A, _, p, _ = operator_with_arrays_and_products

    B = A.copy()
    with log_levels({"pymor.algorithms": "ERROR"}):
        U, s = pod(A, product=p, method=method)
    assert np.all(almost_equal(A, B))
    assert len(U) == len(s)
    assert np.allclose(U.gramian(p), np.eye(len(s)))
