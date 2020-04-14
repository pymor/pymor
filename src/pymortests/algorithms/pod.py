# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import sampled_from

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.pod import pod
from pymortests.fixtures.operator import operator_with_arrays_and_products
from pymortests.strategies import vector_arrays

methods = ['method_of_snapshots', 'qr_svd']


@given(vector_arrays(count=1), sampled_from(methods))
def test_pod(vector_array, method):
    A = vector_array[0]
    print(type(A))
    print(A.dim, len(A))

    B = A.copy()
    U, s = pod(A, method=method)
    assert np.all(almost_equal(A, B))
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
