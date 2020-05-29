# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from hypothesis import given, assume, reproduce_failure, settings
from hypothesis.strategies import sampled_from

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.tools.floatcmp import contains_zero_vector
from pymortests.fixtures.operator import operator_with_arrays_and_products
from pymortests.strategies import vector_arrays

methods = ['method_of_snapshots', 'qr_svd']


@settings(deadline=None)
@given(vector_arrays(count=1), sampled_from(methods))
def test_pod(vector_array, method):
    A = vector_array[0]
    product = None
    print(type(A))
    print(A.dim, len(A))
    # TODO assumption here masks a potential issue with the algorithm
    #      where it fails in internal lapack instead of a proper error
    assume(len(A) > 1 or A.dim > 1)
    assume(not contains_zero_vector(A, rtol=1e-13, atol=1e-13))

    B = A.copy()
    # we run the pod with infinite tol so we can manually check gram_schidt and adjust assertions
    orth_tol = np.inf
    U, s = pod(A, method=method, orth_tol=orth_tol)
    assert np.all(almost_equal(A, B))
    if U.dim > 0 and len(U) > 0:
        orth_tol = 1e-10 # the default
        err = np.max(np.abs(U.inner(U, product) - np.eye(len(U))))
        U_orth = gram_schmidt(U, product=product, copy=True) if err >= orth_tol else U
        if len(U_orth) < len(U):
            assert len(U_orth) < len(s)
        else:
            assert len(U) == len(s)
    else:
        assert len(U) == len(s)
    assert np.allclose(U.gramian(), np.eye(len(s)))


@pytest.mark.parametrize('method', methods)
def test_pod_with_product(operator_with_arrays_and_products, method):
    _, _, A, _, p, _ = operator_with_arrays_and_products
    print(type(A))
    print(A.dim, len(A))

    B = A.copy()
    U, s = pod(A, product=p, method=method)
    assert np.all(almost_equal(A, B))
    assert len(U) == len(s)
    assert np.allclose(U.gramian(p), np.eye(len(s)))
