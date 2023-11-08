# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import pytest

from pymor.algorithms.qr import qr, rrqr
from pymor.vectorarrays.numpy import NumpyVectorSpace


def test_qr_random():
    A = NumpyVectorSpace(5).random(3)
    Q, R = qr(A)
    assert len(A) == len(Q)


def test_qr_zeros():
    A = NumpyVectorSpace(5).zeros(2)
    with pytest.raises(RuntimeError):
        Q, R = qr(A)


def test_rrqr_random():
    A = NumpyVectorSpace(5).random(3)
    Q, R = rrqr(A)
    assert len(A) == len(Q)


def test_rrqr_zeros():
    A = NumpyVectorSpace(5).zeros(2)
    Q, R = rrqr(A)
    assert len(Q) == 0
