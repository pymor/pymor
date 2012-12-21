from __future__ import absolute_import, division, print_function, unicode_literals

import math as m
import numpy as np
import pymor.core as core
from .interfaces import DomainDiscretizerInterface
from pymor.common.domaindescription import RectDomain as DRect
from pymor.grid.rect import RectGrid
from pymor.grid.tria import TriaGrid
from pymor.common.boundaryinfo import FromIndicators


class Default(DomainDiscretizerInterface):

    def __init__(self, grid_type=None, diameter=1 / 100):
        self.grid_type = grid_type
        self.diameter = diameter

    def discretize(self, domain_description):
        if isinstance(domain_description, DRect):
            x0i = int(m.ceil(domain_description.width * m.sqrt(2) / self.diameter))
            x1i = int(m.ceil(domain_description.height * m.sqrt(2) / self.diameter))
            grid_type = self.grid_type or TriaGrid
            if grid_type == TriaGrid:
                grid = TriaGrid(domain=domain_description.domain, num_intervals=(x0i, x1i))
            elif grid_type == RectGrid:
                grid = RectGrid(domain=domain_description.domain, num_intervals=(x0i, x1i))
            else:
                raise NotImplementedError('I do not know how to discretize {} with {}'.format('domaindescription.RectDomain',
                                     grid_type))

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

            indicators = {bt: indicator_factory(domain_description, bt) for bt in domain_description.boundary_types}
            bi = FromIndicators(grid, indicators)
            return grid, bi
        else:
            raise NotImplementedError('I do not know how to discretize {}'.format(domain_description))
