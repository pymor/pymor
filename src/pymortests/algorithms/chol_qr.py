# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import warnings

import numpy as np
import pytest
from hypothesis import assume, settings

import pymortests.strategies as pyst
from pymor.algorithms.basic import almost_equal, contains_zero_vector
from pymor.algorithms.chol_qr import shifted_chol_qr
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.vectorarrays.numpy import NumpyVectorSpace


@pyst.given_vector_arrays()
@settings(deadline=None)
def test_shifted_chol_qr(vector_array):
    U = vector_array

    assume(len(U) > 1 or not contains_zero_vector(U))

    V = U.copy()

    onbgs, Rgs = gram_schmidt(U, copy=True, return_R=True)
    if len(onbgs) < len(V):
        warnings.warn('Linearly dependent vectors detected! Skipping ...')
        return

    onb, R = shifted_chol_qr(U, copy=True)
    assert np.all(almost_equal(U, V))
    assert np.allclose(onb.inner(onb), np.eye(len(onb)))
    lc = onb.lincomb(onb.inner(U).T)
    rtol = atol = 1e-13

    assert np.all(almost_equal(U, lc, rtol=rtol, atol=atol))
    try:
        assert np.all(almost_equal(V, onb.lincomb(R.T), rtol=rtol, atol=atol))
    except AssertionError:
        assert 0

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


@pytest.mark.parametrize('copy', [False, True])
def test_chol_qr_zeros(copy):
    n, m = 5, 2
    V = NumpyVectorSpace(n).zeros(m)
    Q, R = shifted_chol_qr(V, copy=copy)
    if copy:
        assert len(V) == m
    else:
        assert len(V) == 0
    assert len(Q) == 0


@pytest.mark.parametrize('copy', [False, True])
def test_chol_qr_empty(copy):
    n = 5
    V = NumpyVectorSpace(n).empty(0)
    Q, R = shifted_chol_qr(V, copy=copy)
    assert len(V) == len(Q) == 0
