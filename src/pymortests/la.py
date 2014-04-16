# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor import la
from pymor.operators.cg import L2ProductP1
from pymortests.base import runmodule
from pymor.grids.tria import TriaGrid
from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo


def test_induced():
    grid = TriaGrid(num_intervals=(10, 10))
    boundary_info = AllDirichletBoundaryInfo(grid)
    product = L2ProductP1(grid, boundary_info)
    zero = la.NumpyVectorArray(np.zeros(grid.size(2)))
    norm = la.induced_norm(product)
    value = norm(zero)
    np.testing.assert_almost_equal(value, 0.0)


if __name__ == "__main__":
    runmodule(filename=__file__)
