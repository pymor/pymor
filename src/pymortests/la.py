# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.discretizers.builtin.grids.boundaryinfos import AllDirichletBoundaryInfo
from pymor.discretizers.builtin.grids.tria import TriaGrid
from pymor.discretizers.builtin.cg import L2ProductP1
from pymor.operators.constructions import induced_norm
from pymortests.base import runmodule


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
