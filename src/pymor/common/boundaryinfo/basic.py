from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .interfaces import IBoundaryInfo


class FromIndicators(IBoundaryInfo):

    def __init__(self, grid, indicators, data):
        self.grid = grid
        self.condition_types = set(indicators.keys())
        assert self.condition_types <= set(('dirichlet', 'neumann', 'robin')), ValueError('Unknown boundary type')
        self._data = data
        self._masks = {condition_type:[np.zeros(grid.size(codim), dtype='bool') for codim in xrange(grid.dim + 1)]
                       for condition_type in self.condition_types}
        for condition_type, codims in self._masks.iteritems():
            for codim, mask in enumerate(codims):
                mask[grid.boundaries(codim)] = indicators[condition_type](grid.centers(codim)[grid.boundaries(codim)])

    def mask(self, condition_type, codim):
        return self._masks[condition_type][codim]

    def data(self, condition_type):
        return self.data[condition_type]


class AllDirichlet(IBoundaryInfo):

    def __init__(self, grid, data):
        self.grid = grid
        self.condition_types = set(('dirichlet',))
        self._data = data

    def mask(self, condition_type, codim):
        assert condition_type == 'dirichlet', ValueError('Has no condition_type "{}"'.format(condition_type))
        return np.ones(self.grid.size(codim), dtype='bool') * self.grid.boundary_mask(codim)

    def data(self, condition_type):
        assert condition_type == 'dirichlet', ValueError('Has no condition_type "{}"'.format(condition_type))
        return self._data


class AllDirichletZero(AllDirichlet):

    def __init__(self, grid):
        super(AllDirichletZero, self).__init__(grid, lambda points: np.zeros(points.shape[0]))

