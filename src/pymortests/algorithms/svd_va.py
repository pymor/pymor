# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from hypothesis import given, reproduce_failure, assume
from hypothesis.strategies import sampled_from

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.svd_va import method_of_snapshots, qr_svd
from pymor.tools.floatcmp import contains_zero_vector
from pymortests.base import runmodule
from pymortests.fixtures.operator import operator_with_arrays_and_products
from pymortests.strategies import vector_arrays

methods = [method_of_snapshots, qr_svd]


@given(vector_arrays(count=1), sampled_from(methods))
def test_method_of_snapshots(vector_array, method):
    A = vector_array[0]
    print(type(A))
    print(A.dim, len(A))

    # TODO
    assume(not contains_zero_vector(A))

    B = A.copy()
    U, s, Vh = method(A)
    assert np.all(almost_equal(A, B))
    assert len(U) == len(s) == Vh.shape[0]
    assert Vh.shape[1] == len(A)
    assert np.allclose(Vh @ Vh.T.conj(), np.eye(len(s)))
    U.scal(s)
    UsVh = U.lincomb(Vh.T)
    assert np.all(almost_equal(A, UsVh, rtol=4e-8))


@pytest.mark.parametrize('method', methods)
def test_method_of_snapshots_with_product(operator_with_arrays_and_products, method):
    _, _, A, _, p, _ = operator_with_arrays_and_products
    print(type(A))
    print(A.dim, len(A))

    B = A.copy()
    U, s, Vh = method(A, product=p)
    assert np.all(almost_equal(A, B))
    assert len(U) == len(s) == Vh.shape[0]
    assert Vh.shape[1] == len(A)
    assert np.allclose(Vh @ Vh.T.conj(), np.eye(len(s)))
    U.scal(s)
    UsVh = U.lincomb(Vh.T)
    assert np.all(almost_equal(A, UsVh, rtol=4e-8))


if __name__ == "__main__":
    runmodule(filename=__file__)
