# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from hypothesis import assume, settings

import pymortests.strategies as pyst
from pymor.algorithms.basic import almost_equal, contains_zero_vector
from pymor.algorithms.chol_qr import shifted_chol_qr


@pyst.given_vector_arrays()
@settings(deadline=None)
def test_shifted_chol_qr(vector_array):
    U = vector_array

    assume(len(U) > 1 or not contains_zero_vector(U))

    V = U.copy()

    onb, R = shifted_chol_qr(U, copy=True)
    assert np.all(almost_equal(U, V))
    assert np.allclose(onb.inner(onb), np.eye(len(onb)))
    lc = onb.lincomb(onb.inner(U).T)
    rtol = atol = 1e-13
    assert np.all(almost_equal(U, lc, rtol=rtol, atol=atol))
    assert np.all(almost_equal(V, onb.lincomb(R.T), rtol=rtol, atol=atol))

    onb2, R2 = shifted_chol_qr(U, copy=False)
    assert np.all(almost_equal(onb, onb2))
    assert np.all(R == R2)
    assert np.all(almost_equal(onb, U))


def test_shifted_chol_qr_with_product(operator_with_arrays_and_products):
    _, _, U, _, p, _ = operator_with_arrays_and_products

    assume(len(U) > 1 or not contains_zero_vector(U))

    if U.dim < len(U):
        return

    V = U.copy()

    onb, R = shifted_chol_qr(U, product=p, copy=True)
    assert np.all(almost_equal(U, V))
    assert np.allclose(p.apply2(onb, onb), np.eye(len(onb)))
    assert np.all(almost_equal(U, onb.lincomb(p.apply2(onb, U).T), rtol=1e-13))
    assert np.all(almost_equal(U, onb.lincomb(R.T)))

    onb2, R2 = shifted_chol_qr(U, product=p, copy=False)
    assert np.all(almost_equal(onb, onb2))
    assert np.all(R == R2)
    assert np.all(almost_equal(onb, U))
