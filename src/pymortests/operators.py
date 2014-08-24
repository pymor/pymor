# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from pymor.la import NumpyVectorArray
from pymortests.algorithms import MonomOperator
from pymortests.base import runmodule


def test_lincomb_op():
    p1 = MonomOperator(1)
    p2 = MonomOperator(2)
    p12 = p1 + p2
    p0 = p1 - p1
    x = np.linspace(-1., 1., num=3)
    vx = NumpyVectorArray(x[:, np.newaxis])
    assert np.allclose(p0.apply(vx).data, [0.])
    assert np.allclose(p12.apply(vx).data, (x * x + x)[:, np.newaxis])
    assert np.allclose((p1 * 2.).apply(vx).data, (x * 2.)[:, np.newaxis])
    assert p2.jacobian(vx).apply(vx).almost_equal(p1.apply(vx) * 2.).all()
    assert p0.jacobian(vx).apply(vx).almost_equal(vx * 0.).all()
    with pytest.raises(TypeError):
        p2.as_vector()
    p1.as_vector()
    assert p1.as_vector().almost_equal(p1.apply(NumpyVectorArray(1.)))

    basis = NumpyVectorArray([1.])
    for p in (p1, p2, p12):
        projected = p.projected(basis, basis)
        pa = projected.apply(vx)
        assert pa.almost_equal(p.apply(vx)).all()


if __name__ == "__main__":
    runmodule(filename=__file__)
