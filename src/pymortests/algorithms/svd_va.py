# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from hypothesis import assume, settings, HealthCheck
from hypothesis.strategies import sampled_from

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.svd_va import method_of_snapshots, qr_svd
from pymor.algorithms.basic import contains_zero_vector
from pymor.core.logger import log_levels
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.base import runmodule
from pymortests.strategies import given_vector_arrays

methods = [method_of_snapshots, qr_svd]


@given_vector_arrays(method=sampled_from(methods))
@settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large])
def test_method_of_snapshots(vector_array, method):
    A = vector_array

    # TODO assumption here masks a potential issue with the algorithm
    #      where it fails in internal lapack instead of a proper error
    assume(len(A) > 1 or A.dim > 1)
    assume(not contains_zero_vector(A, rtol=1e-13, atol=1e-13))

    B = A.copy()
    with log_levels({"pymor.algorithms": "ERROR"}):
        U, s, Vh = method(A, rtol=4e-8)  # default tolerance
    assert np.all(almost_equal(A, B))
    assert len(U) == len(s) == Vh.shape[0]
    assert Vh.shape[1] == len(A)
    assert np.allclose(Vh @ Vh.T.conj(), np.eye(len(s)))
    if len(s) > 0:
        U.scal(s)
        UsVh = U.lincomb(Vh.T)
        assert np.all(almost_equal(A, UsVh, atol=s[0]*4e-8*2))


@pytest.mark.parametrize('method', methods)
def test_method_of_snapshots_with_product(operator_with_arrays_and_products, method):
    _, _, A, _, p, _ = operator_with_arrays_and_products

    B = A.copy()
    with log_levels({"pymor.algorithms": "ERROR"}):
        U, s, Vh = method(A, product=p)
    assert np.all(almost_equal(A, B))
    assert len(U) == len(s) == Vh.shape[0]
    assert Vh.shape[1] == len(A)
    assert np.allclose(Vh @ Vh.T.conj(), np.eye(len(s)))
    U.scal(s)
    UsVh = U.lincomb(Vh.T)
    assert np.all(almost_equal(A, UsVh, rtol=4e-8))


@pytest.mark.parametrize('method', methods)
def test_not_too_many_modes(method):
    vec_array = NumpyVectorSpace.from_numpy(np.logspace(-5, 0, 10).reshape((-1, 1)))
    U, s, V = method(vec_array, atol=0, rtol=0)
    assert len(U) == len(s) == len(V) == 1


if __name__ == "__main__":
    runmodule(filename=__file__)
