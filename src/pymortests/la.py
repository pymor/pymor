# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo
from pymor.grids.tria import TriaGrid
from pymor.operators.cg import L2ProductP1
from pymor.operators.constructions import induced_norm
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.base import runmodule


def test_induced():
    grid = TriaGrid(num_intervals=(10, 10))
    boundary_info = AllDirichletBoundaryInfo(grid)
    product = L2ProductP1(grid, boundary_info)
    zero = product.source.zeros()
    norm = induced_norm(product)
    value = norm(zero)
    np.testing.assert_almost_equal(value, 0.0)


def test_gram_schmidt():
    for i in (1, 32):
        b = NumpyVectorSpace.from_data(np.identity(i, dtype=np.float))
        a = gram_schmidt(b)
        assert np.all(almost_equal(b, a))
    c = NumpyVectorSpace.from_data([[1.0, 0], [0., 0]])
    a = gram_schmidt(c)
    assert (a.data == np.array([[1.0, 0]])).all()

if __name__ == "__main__":
    runmodule(filename=__file__)
