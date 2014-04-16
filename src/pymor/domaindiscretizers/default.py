# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import math as m
import numpy as np

from pymor.core import inject_sid
from pymor.domaindescriptions import RectDomain, CylindricalDomain, TorusDomain, LineDomain, CircleDomain
from pymor.grids import RectGrid, TriaGrid, OnedGrid, BoundaryInfoFromIndicators, EmptyBoundaryInfo
from pymor.tools import float_cmp


def discretize_domain_default(domain_description, diameter=1 / 100, grid_type=None):
    '''Discretize a |DomainDescription| using a sensible default implementation.

    This method can discretize the following |DomainDescriptions|:

        +----------------------+-------------+---------+
        | DomainDescription    | grid_type   | default |
        +======================+=============+=========+
        | |RectDomain|         | |TriaGrid|  |    X    |
        |                      +-------------+---------+
        |                      | |RectGrid|  |         |
        +----------------------+-------------+---------+
        | |CylindricalDomain|  | |TriaGrid|  |   n.a.  |
        |                      +-------------+---------+
        |                      | |RectGrid|  |    X    |
        +----------------------+-------------+---------+
        | |TorusDomain|        | |TriaGrid|  |   n.a.  |
        |                      +-------------+---------+
        |                      | |RectGrid|  |    X    |
        +----------------------+-------------+---------+
        | |LineDomain|         | |OnedGrid|  |    X    |
        +----------------------+-------------+---------+
        | |CircleDomain|       | |OnedGrid|  |    X    |
        +----------------------+-------------+---------+

    Parameters
    ----------
    domain_description
        A |DomainDescription| of the domain to discretize.
    diameter
        Maximal diameter of the codim-0 entities of the generated |Grid|.
    grid_type
        The class of the grid which is to be constructed. If `None`, a default choice
        is made according to the table above.

    Returns
    -------
    grid
        The generated |Grid|.
    boundary_info
        The generated |BoundaryInfo|.
    '''

    def discretize_RectDomain():
        x0i = int(m.ceil(domain_description.width * m.sqrt(2) / diameter))
        x1i = int(m.ceil(domain_description.height * m.sqrt(2) / diameter))
        if grid_type == TriaGrid:
            grid = TriaGrid(domain=domain_description.domain, num_intervals=(x0i, x1i))
        else:
            grid = RectGrid(domain=domain_description.domain, num_intervals=(x0i, x1i))

        indicator_id = __name__ + '.discretize_domain_default.discretize_RectDomain'

        def indicator_factory(dd, bt):
            def indicator(X):
                L = np.logical_and(float_cmp(X[:, 0], dd.domain[0, 0]), dd.left == bt)
                R = np.logical_and(float_cmp(X[:, 0], dd.domain[1, 0]), dd.right == bt)
                T = np.logical_and(float_cmp(X[:, 1], dd.domain[1, 1]), dd.top == bt)
                B = np.logical_and(float_cmp(X[:, 1], dd.domain[0, 1]), dd.bottom == bt)
                LR = np.logical_or(L, R)
                TB = np.logical_or(T, B)
                return np.logical_or(LR, TB)
            inject_sid(indicator, indicator_id, dd, bt)
            return indicator

        indicators = {bt: indicator_factory(domain_description, bt)
                      for bt in domain_description.boundary_types}
        bi = BoundaryInfoFromIndicators(grid, indicators)

        return grid, bi

    def discretize_CylindricalDomain():
        x0i = int(m.ceil(domain_description.width * m.sqrt(2) / diameter))
        x1i = int(m.ceil(domain_description.height * m.sqrt(2) / diameter))
        assert grid_type == RectGrid
        grid = RectGrid(domain=domain_description.domain, num_intervals=(x0i, x1i),
                        identify_left_right=True)

        indicator_id = __name__ + '.discretize_domain_default.discretize_CylindricalDomain'

        def indicator_factory(dd, bt):
            def indicator(X):
                T = np.logical_and(float_cmp(X[:, 1], dd.domain[1, 1]), dd.top == bt)
                B = np.logical_and(float_cmp(X[:, 1], dd.domain[0, 1]), dd.bottom == bt)
                TB = np.logical_or(T, B)
                return TB
            inject_sid(indicator, indicator_id, dd, bt)
            return indicator

        indicators = {bt: indicator_factory(domain_description, bt)
                      for bt in domain_description.boundary_types}
        bi = BoundaryInfoFromIndicators(grid, indicators)

        return grid, bi

    def discretize_TorusDomain():
        x0i = int(m.ceil(domain_description.width * m.sqrt(2) / diameter))
        x1i = int(m.ceil(domain_description.height * m.sqrt(2) / diameter))
        assert grid_type == RectGrid
        grid = RectGrid(domain=domain_description.domain, num_intervals=(x0i, x1i),
                        identify_left_right=True, identify_bottom_top=True)

        bi = EmptyBoundaryInfo(grid)

        return grid, bi

    def discretize_LineDomain():
        ni = int(m.ceil(domain_description.width / diameter))
        grid = OnedGrid(domain=domain_description.domain, num_intervals=ni)

        indicator_id = __name__ + '.discretize_domain_default.discretize_LineDomain'

        def indicator_factory(dd, bt):
            def indicator(X):
                L = np.logical_and(float_cmp(X[:, 0], dd.domain[0]), dd.left == bt)
                R = np.logical_and(float_cmp(X[:, 0], dd.domain[1]), dd.right == bt)
                return np.logical_or(L, R)
            inject_sid(indicator, indicator_id, dd, bt)
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

    if not isinstance(domain_description, (RectDomain, CylindricalDomain, TorusDomain, LineDomain, CircleDomain)):
        raise NotImplementedError('I do not know how to discretize {}'.format(domain_description))
    if isinstance(domain_description, RectDomain):
        grid_type = grid_type or TriaGrid
        if grid_type not in (TriaGrid, RectGrid):
            raise NotImplementedError('I do not know how to discretize {} with {}'.format('RectDomain', grid_type))
        return discretize_RectDomain()
    elif isinstance(domain_description, (CylindricalDomain, TorusDomain)):
        grid_type = grid_type or RectGrid
        if grid_type not in (RectGrid, ):
            raise NotImplementedError('I do not know how to discretize {} with {}'
                                      .format(type(domain_description), grid_type))
        if isinstance(domain_description, CylindricalDomain):
            return discretize_CylindricalDomain()
        else:
            return discretize_TorusDomain()
    else:
        grid_type = grid_type or OnedGrid
        if grid_type is not OnedGrid:
            raise NotImplementedError('I do not know hot to discretize {} with {}'
                                      .format(str(type(domain_description)), grid_type))
        return discretize_LineDomain() if isinstance(domain_description, LineDomain) else discretize_CircleDomain()
