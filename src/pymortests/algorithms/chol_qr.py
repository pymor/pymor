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
from pymor.core.config import is_scipy_mkl
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.base import runmodule


@pyst.given_vector_arrays()
@settings(deadline=None)
def test_shifted_chol_qr(vector_array):
    U = vector_array

    assume(len(U) >= 1 and not contains_zero_vector(U))

    V = U.copy()

    onbgs, Rgs = gram_schmidt(U, copy=True, return_R=True)
    if len(onbgs) < len(V):
        warnings.warn('Linearly dependent vectors detected! Skipping ...')
        return

    onb, R = shifted_chol_qr(U, return_R=True, copy=True)
    assert np.all(almost_equal(U, V))
    assert np.allclose(onb.inner(onb), np.eye(len(onb)))
    lc = onb.lincomb(onb.inner(U).T)
    rtol = atol = 1e-10

    assert np.all(almost_equal(U, lc, rtol=rtol, atol=atol))
    try:
        assert np.all(almost_equal(V, onb.lincomb(R.T), rtol=rtol, atol=atol))
    except AssertionError:
        assert 0

    onb2, R2 = shifted_chol_qr(U, return_R=True, copy=False)
    assert np.all(almost_equal(onb, onb2))
    assert np.all(R == R2)
    assert np.all(almost_equal(onb, U))


def test_shifted_chol_qr_with_product(operator_with_arrays_and_products):
    if is_scipy_mkl():
        pytest.xfail('fails with mkl')
    _, _, U, _, p, _ = operator_with_arrays_and_products

    assume(len(U) >= 1 and not contains_zero_vector(U))

    if U.dim < len(U):
        return

    V = U.copy()

    onb, R = shifted_chol_qr(U, product=p, return_R=True, copy=True)
    assert np.all(almost_equal(U, V))
    assert np.allclose(p.apply2(onb, onb), np.eye(len(onb)), atol=1e-7)
    assert np.all(almost_equal(U, onb.lincomb(p.apply2(onb, U).T), rtol=1e-11))
    assert np.all(almost_equal(U, onb.lincomb(R.T)))

    onb2, R2 = shifted_chol_qr(U, product=p, return_R=True, copy=False)
    assert np.all(almost_equal(onb, onb2))
    assert np.all(R == R2)
    assert np.all(almost_equal(onb, U))


@pytest.mark.parametrize('copy', [False, True])
def test_chol_qr_empty(copy):
    n = 5
    V = NumpyVectorSpace(n).empty(0)
    Q, R = shifted_chol_qr(V, return_R=True, copy=copy)
    assert len(V) == len(Q) == 0

@pyst.given_vector_arrays()
def test_shifted_chol_qr_with_offset(vector_array):
    U = vector_array

    assume(len(U) >= 2 and not contains_zero_vector(U))

    # check if the test VectorArray contains linearly dependent vectors
    # Q and R are later used for comparison
    onbgs, Rgs = gram_schmidt(U, copy=True, return_R=True)
    if len(onbgs) < len(U):
        warnings.warn('Linearly dependent vectors detected! Skipping ...')
        return

    # orthogonalize first half
    offset = round(len(U) / 2)
    onb, Ro = shifted_chol_qr(U[:offset], return_R=True, copy=True)
    assert np.allclose(onb.inner(onb), np.eye(len(onb)))

    # orthogonalize second half
    onb.append(U[offset:])
    _, R = shifted_chol_qr(onb, return_R=True, copy=False)

    # insert R of first orthogonalization step into R of second step
    # overwritten matrix block only contains a unit matrix
    R[:offset,:offset] = Ro

    # compare Q and R of shifted Cholesky QR with offset to the gram_schmidt implementation
    rtol = atol = 1e-13
    assert np.all(almost_equal(onb, onbgs, rtol=rtol, atol=atol))
    assert np.allclose(Rgs, R)


@pyst.given_vector_arrays()
@settings(deadline=None)
def test_recalculated_shifted_chol_qr(vector_array):
    U = vector_array

    assume(len(U) >= 1 and not contains_zero_vector(U))

    V = U.copy()

    onbgs, Rgs = gram_schmidt(U, copy=True, return_R=True, atol=0, rtol=0)
    if len(onbgs) < len(V):
        warnings.warn('Linearly dependent vectors detected! Skipping ...')
        return

    onb, R = shifted_chol_qr(U, return_R=True, copy=True, recompute_shift=True, maxiter=10, orth_tol=1e-13)
    assert np.all(almost_equal(U, V))
    assert np.allclose(onb.inner(onb), np.eye(len(onb)))
    lc = onb.lincomb(onb.inner(U).T)
    rtol = atol = 1e-9

    assert np.all(almost_equal(U, lc, rtol=rtol, atol=atol))
    assert np.all(almost_equal(V, onb.lincomb(R.T), rtol=rtol, atol=atol))

    onb2, R2 = shifted_chol_qr(U, return_R=True, copy=False, recompute_shift=True, maxiter=10, orth_tol=1e-13)
    assert np.all(almost_equal(onb, onb2))
    assert np.all(R == R2)
    assert np.all(almost_equal(onb, U))


def test_recalculated_shifted_chol_qr_with_product(operator_with_arrays_and_products):
    _, _, U, _, p, _ = operator_with_arrays_and_products

    assume(len(U) >= 1 and not contains_zero_vector(U))

    if U.dim < len(U):
        return

    V = U.copy()

    onb, R = shifted_chol_qr(U, return_R=True, product=p, copy=True, recompute_shift=True, maxiter=10, orth_tol=1e-13)
    assert np.all(almost_equal(U, V))
    assert np.allclose(p.apply2(onb, onb), np.eye(len(onb)))
    assert np.all(almost_equal(U, onb.lincomb(p.apply2(onb, U).T), rtol=1e-11))
    assert np.all(almost_equal(U, onb.lincomb(R.T)))

    onb2, R2 = shifted_chol_qr(U, return_R=True, product=p, copy=False,
                               recompute_shift=True, maxiter=10, orth_tol=1e-13)
    assert np.all(almost_equal(onb, onb2))
    assert np.all(R == R2)
    assert np.all(almost_equal(onb, U))

if __name__ == '__main__':
    runmodule(filename=__file__)
