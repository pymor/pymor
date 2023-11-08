# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import pytest

from pymor.algorithms.qr import qr, rrqr
from pymor.vectorarrays.numpy import NumpyVectorSpace

pytestmark = pytest.mark.builtin


@pytest.mark.parametrize('complex', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_qr_random(copy, complex):
    V = NumpyVectorSpace(5).random(3)
    if complex:
        V += 1j * NumpyVectorSpace(5).random(3)
    Q, R = qr(V, copy=copy)
    assert len(V) == len(Q) == 3


@pytest.mark.parametrize('complex', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_qr_random_offset(copy, complex):
    V = NumpyVectorSpace(5).random(3)
    if complex:
        V += 1j * NumpyVectorSpace(5).random(3)
    V[0].scal(1 / V[0].norm())
    Q, R = qr(V, offset=1, copy=copy)
    assert len(V) == len(Q) == 3


@pytest.mark.parametrize('complex', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_qr_ones(copy, complex):
    V = NumpyVectorSpace(5).ones(2)
    if complex:
        V += 1j * NumpyVectorSpace(5).ones(2)
    with pytest.raises(RuntimeError):
        Q, R = qr(V, copy=copy)


@pytest.mark.parametrize('copy', [False, True])
def test_qr_zeros(copy):
    V = NumpyVectorSpace(5).zeros(2)
    with pytest.raises(RuntimeError):
        Q, R = qr(V, copy=copy)


@pytest.mark.parametrize('copy', [False, True])
def test_qr_empty(copy):
    V = NumpyVectorSpace(5).empty(0)
    Q, R = qr(V, copy=copy)
    assert len(V) == len(Q) == 0


@pytest.mark.parametrize('complex', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_rrqr_random(copy, complex):
    V = NumpyVectorSpace(5).random(3)
    if complex:
        V += 1j * NumpyVectorSpace(5).random(3)
    Q, R = rrqr(V, copy=copy)
    assert len(V) == len(Q) == 3


@pytest.mark.parametrize('complex', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_rrqr_random_offset(copy, complex):
    V = NumpyVectorSpace(5).random(3)
    if complex:
        V += 1j * NumpyVectorSpace(5).random(3)
    V[0].scal(1 / V[0].norm())
    Q, R = rrqr(V, offset=1, copy=copy)
    assert len(V) == len(Q) == 3


@pytest.mark.parametrize('complex', [False, True])
@pytest.mark.parametrize('copy', [False, True])
def test_rrqr_ones(copy, complex):
    V = NumpyVectorSpace(5).ones(2)
    if complex:
        V += 1j * NumpyVectorSpace(5).ones(2)
    Q, R = rrqr(V, copy=copy)
    if copy:
        assert len(V) == 2
    else:
        assert len(V) == 1
    assert len(Q) == 1


@pytest.mark.parametrize('copy', [False, True])
def test_rrqr_zeros(copy):
    V = NumpyVectorSpace(5).zeros(2)
    Q, R = rrqr(V, copy=copy)
    if copy:
        assert len(V) == 2
    else:
        assert len(V) == 0
    assert len(Q) == 0


@pytest.mark.parametrize('copy', [False, True])
def test_rrqr_empty(copy):
    V = NumpyVectorSpace(5).empty(0)
    Q, R = rrqr(V, copy=copy)
    assert len(V) == len(Q) == 0
