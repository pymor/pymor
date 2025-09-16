# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.solvers.newton import NewtonError, NewtonSolver
from pymor.tools.floatcmp import float_cmp
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.base import runmodule
from pymortests.fixtures.operator import MonomOperator

pytestmark = pytest.mark.builtin


def _newton(mop, initial_value=1.0, return_info=False, **kwargs):
    rhs = NumpyVectorSpace.from_numpy([0.0])
    guess = NumpyVectorSpace.from_numpy([initial_value])
    return NewtonSolver(**kwargs).solve(mop, rhs, initial_guess=guess)


@pytest.mark.parametrize('order', list(range(1, 8)))
@pytest.mark.parametrize('error_measure', ['update', 'residual'])
def test_newton(order, error_measure):
    mop = MonomOperator(order)
    U = _newton(mop,
                atol=1e-15 if error_measure == 'residual' else 1e-7,
                rtol=0.,
                error_measure=error_measure)
    assert float_cmp(mop.apply(U).to_numpy(), 0.0)


def test_newton_fail():
    mop = MonomOperator(0)
    with pytest.raises(NewtonError):
        _ = _newton(mop, maxiter=10, stagnation_threshold=np.inf)


def test_newton_with_line_search():
    mop = MonomOperator(3) - 2 * MonomOperator(1) + 2 * MonomOperator(0)
    U = _newton(mop, initial_value=0.0, atol=1e-15, relax='armijo')
    assert float_cmp(mop.apply(U).to_numpy(), 0.0)


def test_newton_fail_without_line_search():
    mop = MonomOperator(3) - 2 * MonomOperator(1) + 2 * MonomOperator(0)
    with pytest.raises(NewtonError):
        _ = _newton(mop, initial_value=0.0, atol=1e-15, relax=1.)


def test_newton_unknown_line_search():
    mop = MonomOperator(1)
    with pytest.raises(ValueError):
        _ = _newton(mop, relax='armo')


def test_newton_residual_is_zero(order=5):  # noqa: PT028
    mop = MonomOperator(order)
    U = _newton(mop, initial_value=0.0)
    assert float_cmp(mop.apply(U).to_numpy(), 0.0)


if __name__ == '__main__':
    runmodule(filename=__file__)
