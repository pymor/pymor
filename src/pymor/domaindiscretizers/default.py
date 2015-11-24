# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import math as m
import numpy as np

from pymor.domaindescriptions.basic import RectDomain, CylindricalDomain, TorusDomain, LineDomain, CircleDomain
from pymor.domaindescriptions.polygonal import PolygonalDomain
from pymor.grids.boundaryinfos import BoundaryInfoFromIndicators, EmptyBoundaryInfo
from pymor.grids.oned import OnedGrid
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid
from pymor.tools.floatcmp import float_cmp


def discretize_domain_default(domain_description, diameter=1 / 100, grid_type=None):
    """Discretize a |DomainDescription| using a sensible default implementation.

    This method can discretize the following |DomainDescriptions|:

        +----------------------+-------------+---------+
        | DomainDescription    | grid_type   | default |
        +======================+=============+=========+
        | |RectDomain|         | |TriaGrid|  |    X    |
        |                      +-------------+---------+
        |                      | |RectGrid|  |         |
        +----------------------+-------------+---------+
        | |CylindricalDomain|  | |TriaGrid|  |    X    |
        |                      +-------------+---------+
        |                      | |RectGrid|  |         |
        +----------------------+-------------+---------+
        | |TorusDomain|        | |TriaGrid|  |    X    |
        |                      +-------------+---------+
        |                      | |RectGrid|  |         |
        +----------------------+-------------+---------+
        | |LineDomain|         | |OnedGrid|  |    X    |
        +----------------------+-------------+---------+
        | |CircleDomain|       | |OnedGrid|  |    X    |
        +----------------------+-------------+---------+
        | |PolygonalDomain     | |GmshGrid|  |    X    |
        +----------------------+-------------+---------+

    Parameters
    ----------
    domain_description
        A |DomainDescription| of the domain to discretize.
    diameter
        Maximal diameter of the codim-0 entities of the generated |Grid|.
    grid_type
        The class of the |Grid| which is to be constructed. If `None`, a default
        choice is made according to the table above.

    Returns
    -------
    grid
        The generated |Grid|.
    boundary_info
        The generated |BoundaryInfo|.
    """

    def discretize_RectDomain():
        if grid_type == RectGrid:
            x0i = int(m.ceil(domain_description.width * m.sqrt(2) / diameter))
            x1i = int(m.ceil(domain_description.height * m.sqrt(2) / diameter))
        elif grid_type == TriaGrid:
            x0i = int(m.ceil(domain_description.width / diameter))
            x1i = int(m.ceil(domain_description.height / diameter))
        else:
            raise NotImplementedError
        grid = grid_type(domain=domain_description.domain, num_intervals=(x0i, x1i))

        def indicator_factory(dd, bt):
            def indicator(X):
                L = np.logical_and(float_cmp(X[:, 0], dd.domain[0, 0]), dd.left == bt)
                R = np.logical_and(float_cmp(X[:, 0], dd.domain[1, 0]), dd.right == bt)
                T = np.logical_and(float_cmp(X[:, 1], dd.domain[1, 1]), dd.top == bt)
                B = np.logical_and(float_cmp(X[:, 1], dd.domain[0, 1]), dd.bottom == bt)
                LR = np.logical_or(L, R)
                TB = np.logical_or(T, B)
                return np.logical_or(LR, TB)
            return indicator

        indicators = {bt: indicator_factory(domain_description, bt)
                      for bt in domain_description.boundary_types}
        bi = BoundaryInfoFromIndicators(grid, indicators)

        return grid, bi

    def discretize_CylindricalDomain():
        if grid_type == RectGrid:
            x0i = int(m.ceil(domain_description.width * m.sqrt(2) / diameter))
            x1i = int(m.ceil(domain_description.height * m.sqrt(2) / diameter))
        elif grid_type == TriaGrid:
            x0i = int(m.ceil(domain_description.width / diameter))
            x1i = int(m.ceil(domain_description.height / diameter))
        else:
            raise NotImplementedError
        grid = grid_type(domain=domain_description.domain, num_intervals=(x0i, x1i),
                         identify_left_right=True)

        def indicator_factory(dd, bt):
            def indicator(X):
                T = np.logical_and(float_cmp(X[:, 1], dd.domain[1, 1]), dd.top == bt)
                B = np.logical_and(float_cmp(X[:, 1], dd.domain[0, 1]), dd.bottom == bt)
                TB = np.logical_or(T, B)
                return TB
            return indicator

        indicators = {bt: indicator_factory(domain_description, bt)
                      for bt in domain_description.boundary_types}
        bi = BoundaryInfoFromIndicators(grid, indicators)

        return grid, bi

    def discretize_TorusDomain():
        if grid_type == RectGrid:
            x0i = int(m.ceil(domain_description.width * m.sqrt(2) / diameter))
            x1i = int(m.ceil(domain_description.height * m.sqrt(2) / diameter))
        elif grid_type == TriaGrid:
            x0i = int(m.ceil(domain_description.width / diameter))
            x1i = int(m.ceil(domain_description.height / diameter))
        else:
            raise NotImplementedError
        grid = grid_type(domain=domain_description.domain, num_intervals=(x0i, x1i),
                         identify_left_right=True, identify_bottom_top=True)

        bi = EmptyBoundaryInfo(grid)

        return grid, bi

    def discretize_LineDomain():
        ni = int(m.ceil(domain_description.width / diameter))
        grid = OnedGrid(domain=domain_description.domain, num_intervals=ni)

        def indicator_factory(dd, bt):
            def indicator(X):
                L = np.logical_and(float_cmp(X[:, 0], dd.domain[0]), dd.left == bt)
                R = np.logical_and(float_cmp(X[:, 0], dd.domain[1]), dd.right == bt)
                return np.logical_or(L, R)
            return indicator

        indicators = {bt: indicator_factory(domain_description, bt)
                      for bt in domain_description.boundary_types}
        bi = BoundaryInfoFromIndicators(grid, indicators)

        return grid, bi

    def discretize_CircleDomain():
        ni = int(m.ceil(domain_description.width / diameter))
        grid = OnedGrid(domain=domain_description.domain, num_intervals=ni, identify_left_right=True)
        bi = EmptyBoundaryInfo(grid)

        return grid, bi

    if not isinstance(domain_description,
                      (RectDomain, CylindricalDomain, TorusDomain, LineDomain, CircleDomain, PolygonalDomain)):
        raise NotImplementedError('I do not know how to discretize {}'.format(domain_description))
    if isinstance(domain_description, RectDomain):
        grid_type = grid_type or TriaGrid
        if grid_type not in (TriaGrid, RectGrid):
            raise NotImplementedError('I do not know how to discretize {} with {}'.format('RectDomain', grid_type))
        return discretize_RectDomain()
    elif isinstance(domain_description, (CylindricalDomain, TorusDomain)):
        grid_type = grid_type or TriaGrid
        if grid_type not in (TriaGrid, RectGrid):
            raise NotImplementedError('I do not know how to discretize {} with {}'
                                      .format(type(domain_description), grid_type))
        if isinstance(domain_description, CylindricalDomain):
            return discretize_CylindricalDomain()
        else:
            return discretize_TorusDomain()
    elif isinstance(domain_description, PolygonalDomain):
        from pymor.grids.gmsh import GmshGrid
        from pymor.domaindiscretizers.gmsh import discretize_gmsh
        assert grid_type is None or grid_type is GmshGrid
        return discretize_gmsh(domain_description, clscale=diameter)
    else:
        grid_type = grid_type or OnedGrid
        if grid_type is not OnedGrid:
            raise NotImplementedError('I do not know hot to discretize {} with {}'
                                      .format(str(type(domain_description)), grid_type))
        return discretize_LineDomain() if isinstance(domain_description, LineDomain) else discretize_CircleDomain()
