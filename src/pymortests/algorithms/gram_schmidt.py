# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from hypothesis import settings
from scipy.linalg import hilbert

import pymortests.strategies as pyst
from pymor.algorithms.basic import almost_equal
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.core.logger import log_levels
from pymor.vectorarrays.list import NumpyListVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.algorithms.qr_test_util import evaluate_qr, evaluate_qr_offset
from pymortests.base import runmodule


@pytest.mark.builtin
@pytest.mark.parametrize('copy', [False, True])
def test_gram_schmidt_empty(copy):
    n = 5
    V = NumpyVectorSpace(n).empty(0)
    Q, _ = gram_schmidt(V, return_R=True, copy=copy)
    assert len(V) == len(Q) == 0


@pytest.mark.parametrize('va_space', [NumpyVectorSpace, NumpyListVectorSpace])
@pytest.mark.parametrize('n', [5])
@pytest.mark.parametrize('return_R', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_gram_schmidt_parameters(va_space, n, return_R, copy):
    # larger hilbert matrices are more ill-conditioned,
    # but are never rank-deficient in exact arithmetic
    A = hilbert(n)
    A = va_space(n).from_numpy(A)
    evaluate_qr(gram_schmidt, A, None, return_R, copy, {})


@pyst.given_vector_arrays()
@settings(deadline=None)
def test_gram_schmidt(vector_array):
    A = vector_array
    evaluate_qr(gram_schmidt, A, None, True, True, {})


@pyst.given_vector_arrays()
# into how many blocks the matrix should be split; 0 for n blocks/ single vectors
@pytest.mark.parametrize('num_blocks', [1, 2, 5, 0])
def test_gram_schmidt_with_offset(vector_array, num_blocks):
    A = vector_array
    evaluate_qr_offset(gram_schmidt, A, num_blocks, {})


def test_gram_schmidt_with_product(operator_with_arrays_and_products):
    _, _, A, _, product, _ = operator_with_arrays_and_products
    evaluate_qr(gram_schmidt, A, product, True, True, {})


@settings(deadline=None)
@pyst.given_vector_arrays(count=2)
def test_gram_schmidt_biorth(vector_arrays):
    U1, U2 = vector_arrays

    if len(U1) != len(U2):
        with pytest.raises(Exception):
            A1, A2 = gram_schmidt_biorth(U1, U2, copy=True)
        return

    onb1 = gram_schmidt(U1)
    if len(onb1) < len(U1):
        return
    onb2 = gram_schmidt(U2)
    if len(onb2) < len(U2):
        return

    V1 = U1.copy()
    V2 = U2.copy()

    # this is the default used in gram_schmidt_biorth
    check_tol = 1e-3
    with log_levels({'pymor.algorithms.gram_schmidt.gram_schmidt_biorth': 'ERROR'}):
        A1, A2 = gram_schmidt_biorth(U1, U2, copy=True, check_tol=check_tol)
    assert np.all(almost_equal(U1, V1))
    assert np.all(almost_equal(U2, V2))
    assert np.allclose(A2.inner(A1), np.eye(len(A1)), atol=check_tol)
    c = (1 if len(A1) == 0 else np.linalg.cond(A1.to_numpy())) \
        * (1 if len(A2) == 0 else np.linalg.cond(A2.to_numpy()))
    assert np.all(almost_equal(U1, A1.lincomb(A2.inner(U1)), rtol=c * 1e-14))
    assert np.all(almost_equal(U2, A2.lincomb(A1.inner(U2)), rtol=c * 1e-14))

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
    assert np.all(almost_equal(U1, A1.lincomb(p.apply2(A2, U1)), rtol=c * 1e-14))
    assert np.all(almost_equal(U2, A2.lincomb(p.apply2(A1, U2)), rtol=c * 1e-14))

    B1, B2 = gram_schmidt_biorth(U1, U2, product=p, copy=False)
    assert np.all(almost_equal(A1, B1))
    assert np.all(almost_equal(A2, B2))
    assert np.all(almost_equal(A1, U1))
    assert np.all(almost_equal(A2, U2))


if __name__ == '__main__':
    runmodule(filename=__file__)
