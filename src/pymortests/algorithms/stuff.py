# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymortests.base import runmodule, MonomOperator
from pymor.algorithms.newton import newton, NewtonError
from pymor.tools.floatcmp import float_cmp
from pymor.vectorarrays.numpy import NumpyVectorSpace


def _newton(order, **kwargs):
    mop = MonomOperator(order)
    rhs = NumpyVectorSpace.from_numpy([0.0])
    guess = NumpyVectorSpace.from_numpy([1.0])
    return newton(mop, rhs, initial_guess=guess, **kwargs)


@pytest.mark.parametrize("order", list(range(1, 8)))
def test_newton(order):
    U, _ = _newton(order, atol=1e-15)
    assert float_cmp(U.to_numpy(), 0.0)


def test_newton_fail():
    with pytest.raises(NewtonError):
        _ = _newton(0, maxiter=10, stagnation_threshold=np.inf)


if __name__ == "__main__":
    runmodule(filename=__file__)
