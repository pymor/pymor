# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.operators.constructions import induced_norm
from pymor.operators.cg import L2ProductP1
from pymortests.base import runmodule
from pymor.grids.tria import TriaGrid
from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo


def test_induced():
    grid = TriaGrid(num_intervals=(10, 10))
    boundary_info = AllDirichletBoundaryInfo(grid)
    product = L2ProductP1(grid, boundary_info)
    zero = product.source.zeros()
    norm = induced_norm(product)
    value = norm(zero)
    np.testing.assert_almost_equal(value, 0.0)


if __name__ == "__main__":
    runmodule(filename=__file__)
