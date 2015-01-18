# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from pymor.la.numpyvectorarray import NumpyVectorArray
from pymortests.base import runmodule, MonomOperator
from pymor.algorithms.newton import newton, NewtonError
import pymor.algorithms.basisextension as bxt
from pymor.tools.floatcmp import float_cmp


def _newton(order):
    mop = MonomOperator(order)
    rhs = NumpyVectorArray([0.0])
    guess = NumpyVectorArray([1.0])
    return newton(mop, rhs, initial_guess=guess)


@pytest.mark.parametrize("order", range(1, 8))
def test_newton(order):
    U, _ = _newton(order)
    assert float_cmp(U.data, 0.0)


def test_newton_fail():
    with pytest.raises(NewtonError):
        _ = _newton(0)


@pytest.fixture(params=('pod_basis_extension', 'gram_schmidt_basis_extension', 'trivial_basis_extension'))
def extension_alg(request):
    return getattr(bxt, request.param)


def test_ext(extension_alg):
    size = 5
    ident = np.identity(size)
    current = ident[0]
    for i in range(1, size):
        c = NumpyVectorArray(current)
        n, _ = extension_alg(c, NumpyVectorArray(ident[i]))
        assert np.allclose(n.data, ident[0:i+1])
        current = ident[0:i+1]

if __name__ == "__main__":
    runmodule(filename=__file__)
