# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
from math import factorial

from pymor.la import NumpyVectorArray
from pymor.operators.basic import OperatorBase
from pymortests.base import polynomials,runmodule
from pymor.algorithms.newton import newton, NewtonError

class MonomOperator(OperatorBase):

    dim_source = dim_range = 1

    def __init__(self, order, monom):
        self.monom = monom
        self.order = order
        self.derivative = list(polynomials(order))[-1][2]

    def apply(self, U, ind=None, mu=None):
        return NumpyVectorArray(self.monom(U.data))

    def jacobian(self, U, mu=None):
        return MonomOperator(self.order, self.derivative)

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        return NumpyVectorArray(1. / U.data)

def test_newton():
    def _newton(order):
        monom = list(polynomials(order))[-1][1]
        mop = MonomOperator(1, monom)
        rhs = NumpyVectorArray([0.0])
        guess = NumpyVectorArray([1.0])
        return newton(mop, rhs, initial_guess=guess)

    for order in range(1, 8, 1):
        U, _ = _newton(order)
        assert np.allclose(U.data, 0.0)

    with pytest.raises(NewtonError):
        _newton(0)

if __name__ == "__main__":
    runmodule(filename=__file__)