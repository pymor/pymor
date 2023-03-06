# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import pytest

from pymor.analyticalproblems.domaindescriptions import PolygonalDomain
from pymortests.base import runmodule

pytestmark = pytest.mark.builtin


def _determine_boundary_type(point):
    y = point[1]

    if y == 0:
        return 'neumann'

    return 'dirichlet'


def test_polygonal_chain_boundary_function():
    chain = [
        [2, 0], [4, 0], [3, 2]
    ]

    reference_boundary_types = {
        'neumann':
            [0, 1],
        'dirichlet':
            [2]
    }

    domain = PolygonalDomain(points=chain, boundary_types=_determine_boundary_type)

    print(domain.boundary_types)

    assert domain.boundary_types == reference_boundary_types


if __name__ == '__main__':
    runmodule(filename=__file__)
