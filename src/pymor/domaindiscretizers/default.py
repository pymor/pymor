from __future__ import absolute_import, division, print_function, unicode_literals

import math as m
import numpy as np

from .interfaces import DomainDiscretizerInterface
from pymor.domaindescriptions import RectDomain
from pymor.grids import RectGrid, TriaGrid, BoundaryInfoFromIndicators


class DefaultDomainDiscretizer(DomainDiscretizerInterface):

    def __init__(self, domain_description, grid_type=TriaGrid):
        if not isinstance(domain_description, RectDomain):
            raise NotImplementedError('I do not know how to discretize {}'.format(domain_description))
        if grid_type not in (TriaGrid, RectGrid):
            raise NotImplementedError('I do not know how to discretize {} with {}'.format('RectDomain', grid_type))

        self.domain_description = domain_description
        self.grid_type = grid_type

    def discretize(self, diameter= 1 / 100):
        x0i = int(m.ceil(self.domain_description.width * m.sqrt(2) / diameter))
        x1i = int(m.ceil(self.domain_description.height * m.sqrt(2) / diameter))
        if self.grid_type == TriaGrid:
            grid = TriaGrid(domain=self.domain_description.domain, num_intervals=(x0i, x1i))
        else:
            grid = RectGrid(domain=self.domain_description.domain, num_intervals=(x0i, x1i))

        def indicator_factory(dd, bt):
            def indicator(X):
                L = np.logical_and(np.abs(X[:, 0] - dd.domain[0, 0]) < 10e-14, dd.left == bt)
                R = np.logical_and(np.abs(X[:, 0] - dd.domain[1, 0]) < 10e-14, dd.right == bt)
                T = np.logical_and(np.abs(X[:, 1] - dd.domain[1, 1]) < 10e-14, dd.top == bt)
                B = np.logical_and(np.abs(X[:, 1] - dd.domain[0, 1]) < 10e-14, dd.bottom == bt)
                LR = np.logical_or(L, R)
                TB = np.logical_or(T, B)
                return np.logical_or(LR, TB)
            return indicator

        indicators = {bt: indicator_factory(self.domain_description, bt)
                      for bt in self.domain_description.boundary_types}
        bi = BoundaryInfoFromIndicators(grid, indicators)

        return grid, bi
