from __future__ import absolute_import, division, print_function

import math as m
import numpy as np

from .interfaces import DomainDiscretizerInterface
from pymor.domaindescriptions import RectDomain, LineDomain
from pymor.grids import RectGrid, TriaGrid, OnedGrid, BoundaryInfoFromIndicators
from pymor.tools import float_cmp


class DefaultDomainDiscretizer(DomainDiscretizerInterface):

    def __init__(self, domain_description, grid_type=None):
        if not isinstance(domain_description, (RectDomain, LineDomain)):
            raise NotImplementedError('I do not know how to discretize {}'.format(domain_description))
        if isinstance(domain_description, RectDomain):
            grid_type = grid_type or TriaGrid
            if grid_type not in (TriaGrid, RectGrid):
                raise NotImplementedError('I do not know how to discretize {} with {}'.format('RectDomain', grid_type))
        else:
            grid_type = grid_type or OnedGrid
            if grid_type is not OnedGrid:
                raise NotImplementedError('I do not know hot to discretize {} with {}'.format('LineDomain', grid_type))

        self.domain_description = domain_description
        self.grid_type = grid_type

    def _discretize_RectDomain(self, diameter):
        x0i = int(m.ceil(self.domain_description.width * m.sqrt(2) / diameter))
        x1i = int(m.ceil(self.domain_description.height * m.sqrt(2) / diameter))
        if self.grid_type == TriaGrid:
            grid = TriaGrid(domain=self.domain_description.domain, num_intervals=(x0i, x1i))
        else:
            grid = RectGrid(domain=self.domain_description.domain, num_intervals=(x0i, x1i))

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

        indicators = {bt: indicator_factory(self.domain_description, bt)
                      for bt in self.domain_description.boundary_types}
        bi = BoundaryInfoFromIndicators(grid, indicators)

        return grid, bi

    def _discretize_LineDomain(self, diameter):
        ni = int(m.ceil(self.domain_description.width / diameter))
        grid = OnedGrid(domain=self.domain_description.domain, num_intervals=ni)

        def indicator_factory(dd, bt):
            def indicator(X):
                L = np.logical_and(np.abs(X[:, 0] - dd.domain[0]) < 10e-14, dd.left == bt)
                R = np.logical_and(np.abs(X[:, 0] - dd.domain[1]) < 10e-14, dd.right == bt)
                return np.logical_or(L, R)
            return indicator

        indicators = {bt: indicator_factory(self.domain_description, bt)
                      for bt in self.domain_description.boundary_types}
        bi = BoundaryInfoFromIndicators(grid, indicators)

        return grid, bi


    def discretize(self, diameter= 1 / 100):
        if isinstance(self.domain_description, RectDomain):
            return self._discretize_RectDomain(diameter)
        else:
            return self._discretize_LineDomain(diameter)
