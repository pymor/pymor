# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from hypothesis import settings, assume, given

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.core.logger import log_levels
from pymor.algorithms.basic import contains_zero_vector
from pymortests.base import runmodule
import pymortests.strategies as pyst


@pyst.given_vector_arrays()
@settings(deadline=20000)
def test_gram_schmidt(vector_array):
    U = vector_array
    # TODO assumption here masks a potential issue with the algorithm
    #      where it fails in del instead of a proper error
    assume(len(U) > 1 or not contains_zero_vector(U))

    V = U.copy()
    onb = gram_schmidt(U, copy=True)
    assert np.all(almost_equal(U, V))
    assert np.allclose(onb.inner(onb), np.eye(len(onb)))
    # TODO maybe raise tolerances again
    assert np.all(almost_equal(U, onb.lincomb(onb.inner(U).T), atol=1e-13, rtol=1e-13))

    onb2 = gram_schmidt(U, copy=False)
    assert np.all(almost_equal(onb, onb2))
    assert np.all(almost_equal(onb, U))


@pyst.given_vector_arrays()
@settings(deadline=None)
def test_gram_schmidt_with_R(vector_array):
    U = vector_array
    # TODO assumption here masks a potential issue with the algorithm
    #      where it fails in del instead of a proper error
    assume(len(U) > 1 or not contains_zero_vector(U))

    V = U.copy()
    onb, R = gram_schmidt(U, return_R=True, copy=True)
    assert np.all(almost_equal(U, V))
    assert np.allclose(onb.inner(onb), np.eye(len(onb)))
    lc = onb.lincomb(onb.inner(U).T)
    rtol = atol = 1e-13
    assert np.all(almost_equal(U, lc, rtol=rtol, atol=atol))
    assert np.all(almost_equal(V, onb.lincomb(R.T), rtol=rtol, atol=atol))

    onb2, R2 = gram_schmidt(U, return_R=True, copy=False)
    assert np.all(almost_equal(onb, onb2))
    assert np.all(R == R2)
    assert np.all(almost_equal(onb, U))


def test_gram_schmidt_with_product(operator_with_arrays_and_products):
    _, _, U, _, p, _ = operator_with_arrays_and_products

    V = U.copy()
    onb = gram_schmidt(U, product=p, copy=True)
    assert np.all(almost_equal(U, V))
    assert np.allclose(p.apply2(onb, onb), np.eye(len(onb)))
    assert np.all(almost_equal(U, onb.lincomb(p.apply2(onb, U).T), rtol=1e-13))

    onb2 = gram_schmidt(U, product=p, copy=False)
    assert np.all(almost_equal(onb, onb2))
    assert np.all(almost_equal(onb, U))


def test_gram_schmidt_with_product_and_R(operator_with_arrays_and_products):
    _, _, U, _, p, _ = operator_with_arrays_and_products

    V = U.copy()
    onb, R = gram_schmidt(U, product=p, return_R=True, copy=True)
    assert np.all(almost_equal(U, V))
    assert np.allclose(p.apply2(onb, onb), np.eye(len(onb)))
    assert np.all(almost_equal(U, onb.lincomb(p.apply2(onb, U).T), rtol=1e-13))
    assert np.all(almost_equal(U, onb.lincomb(R.T)))

    onb2, R2 = gram_schmidt(U, product=p, return_R=True, copy=False)
    assert np.all(almost_equal(onb, onb2))
    assert np.all(R == R2)
    assert np.all(almost_equal(onb, U))


@given(pyst.base_vector_arrays(count=2))
@settings(deadline=None)
def test_gram_schmidt_biorth(vector_arrays):
    U1, U2 = vector_arrays

    V1 = U1.copy()
    V2 = U2.copy()

    # this is the default used in gram_schmidt_biorth
    check_tol = 1e-3
    with log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt_biorth': 'ERROR'}):
        A1, A2 = gram_schmidt_biorth(U1, U2, copy=True, check_tol=check_tol)
    assert np.all(almost_equal(U1, V1))
    assert np.all(almost_equal(U2, V2))
    assert np.allclose(A2.inner(A1), np.eye(len(A1)), atol=check_tol)
    c = np.linalg.cond(A1.to_numpy()) * np.linalg.cond(A2.to_numpy())
    assert np.all(almost_equal(U1, A1.lincomb(A2.inner(U1).T), rtol=c * 1e-14))
    assert np.all(almost_equal(U2, A2.lincomb(A1.inner(U2).T), rtol=c * 1e-14))

    with log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt_biorth': 'ERROR'}):
        B1, B2 = gram_schmidt_biorth(U1, U2, copy=False)
    assert np.all(almost_equal(A1, B1))
    assert np.all(almost_equal(A2, B2))
    assert np.all(almost_equal(A1, U1))
    assert np.all(almost_equal(A2, U2))


def test_gram_schmidt_biorth_with_product(operator_with_arrays_and_products):
    _, _, U, _, p, _ = operator_with_arrays_and_products
    if U.dim < 2:
        return
    l = len(U) // 2
    l = min((l, U.dim - 1))
    if l < 1:
        return
    U1 = U[:l].copy()
    U2 = U[l:2 * l].copy()

    V1 = U1.copy()
    V2 = U2.copy()
    A1, A2 = gram_schmidt_biorth(U1, U2, product=p, copy=True)
    assert np.all(almost_equal(U1, V1))
    assert np.all(almost_equal(U2, V2))
    assert np.allclose(p.apply2(A2, A1), np.eye(len(A1)))
    c = np.linalg.cond(A1.to_numpy()) * np.linalg.cond(p.apply(A2).to_numpy())
    assert np.all(almost_equal(U1, A1.lincomb(p.apply2(A2, U1).T), rtol=c * 1e-14))
    assert np.all(almost_equal(U2, A2.lincomb(p.apply2(A1, U2).T), rtol=c * 1e-14))

    B1, B2 = gram_schmidt_biorth(U1, U2, product=p, copy=False)
    assert np.all(almost_equal(A1, B1))
    assert np.all(almost_equal(A2, B2))
    assert np.all(almost_equal(A1, U1))
    assert np.all(almost_equal(A2, U2))


if __name__ == "__main__":
    runmodule(filename=__file__)
