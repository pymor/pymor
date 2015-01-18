# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor import la
from pymor.la.basic import induced_norm
from pymor.la.gram_schmidt import gram_schmidt
from pymor.la.numpyvectorarray import NumpyVectorArray
from pymor.operators.cg import L2ProductP1
from pymortests.base import runmodule
from pymor.grids.tria import TriaGrid
from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo


def test_induced():
    grid = TriaGrid(num_intervals=(10, 10))
    boundary_info = AllDirichletBoundaryInfo(grid)
    product = L2ProductP1(grid, boundary_info)
    zero = NumpyVectorArray(np.zeros(grid.size(2)))
    norm = induced_norm(product)
    value = norm(zero)
    np.testing.assert_almost_equal(value, 0.0)

def test_gram_schmidt():
    for i in (1, 32):
        b = NumpyVectorArray(np.identity(i, dtype=np.float))
        a = gram_schmidt(b)
        assert b == a
    c = NumpyVectorArray([[1.0, 0], [0., 0]])
    a = gram_schmidt(c)
    assert (a.data == np.array([[1.0, 0]])).all()

if __name__ == "__main__":
    runmodule(filename=__file__)
