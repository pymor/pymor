# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.basic import project_array
from pymor.algorithms.krylov import arnoldi
from pymor.operators.constructions import IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator


@pytest.mark.parametrize('block_size', [1,3])
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('r', [1, 2])
def test_arnoldi(block_size, with_E, r, rng):
    A = NumpyMatrixOperator(rng.standard_normal((10,10)))
    E = NumpyMatrixOperator(rng.standard_normal((10,10))) if with_E else None
    b = A.source.from_numpy_TP(rng.standard_normal((10, block_size)))

    V = arnoldi(A, E, b, r)
    assert len(V) == r * block_size
    assert np.linalg.norm(V.gramian() - np.eye(len(V))) < 1e-10
    E = IdentityOperator(A.source) if E is None else E
    U = E.apply_inverse(b)
    for _ in range(1, r):
        U = E.apply_inverse(A.apply(U))
    assert np.max((project_array(U, V) - U).norm()) < 1e-10
