# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.qr import qr, rrqr
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

pytestmark = pytest.mark.builtin


def random_spd_operator(n, complex):
    rng = np.random.default_rng(0)
    P = rng.normal(size=(n, n))
    if complex:
        P = P + 1j * rng.normal(size=(n, n))
    P = P @ P.conj().T
    return NumpyMatrixOperator(P)


@pytest.mark.parametrize('with_product', [False, True])
@pytest.mark.parametrize('complex', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_qr_random(copy, complex, with_product):
    n, m = 5, 3
    V = NumpyVectorSpace(n).random(m)
    if complex:
        V += 1j * NumpyVectorSpace(n).random(m)
    product = random_spd_operator(n, complex) if with_product else None
    Q, R = qr(V, product=product, copy=copy)
    assert len(V) == len(Q) == m


@pytest.mark.parametrize('with_product', [False, True])
@pytest.mark.parametrize('complex', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_qr_random_offset(copy, complex, with_product):
    n, m = 5, 3
    V = NumpyVectorSpace(n).random(m)
    if complex:
        V += 1j * NumpyVectorSpace(n).random(m)
    product = random_spd_operator(n, complex) if with_product else None
    V[0].scal(1 / V[0].norm(product=product))
    Q, R = qr(V, product=product, offset=1, copy=copy)
    assert len(V) == len(Q) == m


@pytest.mark.parametrize('complex', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_qr_ones(copy, complex):
    n, m = 5, 3
    V = NumpyVectorSpace(n).ones(m)
    if complex:
        V += 1j * NumpyVectorSpace(n).ones(m)
    with pytest.raises(RuntimeError):
        Q, R = qr(V, copy=copy)


@pytest.mark.parametrize('copy', [False, True])
def test_qr_zeros(copy):
    n, m = 5, 3
    V = NumpyVectorSpace(n).zeros(m)
    with pytest.raises(RuntimeError):
        Q, R = qr(V, copy=copy)


@pytest.mark.parametrize('copy', [False, True])
def test_qr_empty(copy):
    n = 5
    V = NumpyVectorSpace(n).empty(0)
    Q, R = qr(V, copy=copy)
    assert len(V) == len(Q) == 0


@pytest.mark.parametrize('with_product', [False, True])
@pytest.mark.parametrize('complex', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_rrqr_random(copy, complex, with_product):
    n, m = 5, 3
    V = NumpyVectorSpace(n).random(m)
    if complex:
        V += 1j * NumpyVectorSpace(n).random(m)
    product = random_spd_operator(n, complex) if with_product else None
    Q, R = rrqr(V, product=product, copy=copy)
    assert len(V) == len(Q) == m


@pytest.mark.parametrize('with_product', [False, True])
@pytest.mark.parametrize('complex', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_rrqr_random_offset(copy, complex, with_product):
    n, m = 5, 3
    V = NumpyVectorSpace(n).random(m)
    if complex:
        V += 1j * NumpyVectorSpace(n).random(m)
    product = random_spd_operator(n, complex) if with_product else None
    V[0].scal(1 / V[0].norm(product=product))
    Q, R = rrqr(V, product=product, offset=1, copy=copy)
    assert len(V) == len(Q) == m


@pytest.mark.parametrize('complex', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_rrqr_ones(copy, complex):
    n, m = 5, 2
    V = NumpyVectorSpace(n).ones(m)
    if complex:
        V += 1j * NumpyVectorSpace(n).ones(m)
    Q, R = rrqr(V, copy=copy)
    if copy:
        assert len(V) == m
    else:
        assert len(V) == 1
    assert len(Q) == 1


@pytest.mark.parametrize('copy', [False, True])
def test_rrqr_zeros(copy):
    n, m = 5, 2
    V = NumpyVectorSpace(n).zeros(m)
    Q, R = rrqr(V, copy=copy)
    if copy:
        assert len(V) == m
    else:
        assert len(V) == 0
    assert len(Q) == 0


@pytest.mark.parametrize('copy', [False, True])
def test_rrqr_empty(copy):
    n = 5
    V = NumpyVectorSpace(n).empty(0)
    Q, R = rrqr(V, copy=copy)
    assert len(V) == len(Q) == 0
