# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from mock import Mock

from pymor import la
from pymor import discretizations
from pymor.operators.cg import L2ProductP1
from pymortests.base import TestBase, runmodule
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid


class TestBasicParameterspace(TestBase):

    def test_induced(self):
        grid = TriaGrid(num_intervals=(10, 10))
        product = L2ProductP1(grid)
        zero = np.zeros(10 * 10 * 3 * 2)
        norm = la.induced_norm(product)
        value = norm(zero)
        self.assertAlmostEqual(value, 0.0)


if __name__ == "__main__":
    runmodule(filename=__file__)
