# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.domaindescriptions.boundarytypes import BoundaryType
from pymor.grids.interfaces import BoundaryInfoInterface


class EmptyBoundaryInfo(BoundaryInfoInterface):
    """|BoundaryInfo| with no |BoundaryTypes| attached to any boundary.
    """

    def __init__(self, grid):
        self.grid = grid
        self.boundary_types = frozenset()

    def mask(self, boundary_type, codim):
        assert False, 'Has no boundary_type "{}"'.format(boundary_type)


class BoundaryInfoFromIndicators(BoundaryInfoInterface):
    """|BoundaryInfo| where the |BoundaryTypes| are determined by indicator functions.

    Parameters
    ----------
    grid
        The grid to which the |BoundaryInfo| is associated.
    indicators
        Dict where each key is a |BoundaryType| and the corresponding value is a boolean
        valued function defined on the analytical domain which indicates if a point belongs
        to a boundary of the given |BoundaryType|. (The indicator functions must be vectorized.)
    """

    def __init__(self, grid, indicators, assert_unique_type=None, assert_some_type=None):
        self.grid = grid
        assert_unique_type = assert_unique_type if assert_unique_type else [1]
        assert_some_type = assert_some_type if assert_some_type else []
        self.boundary_types = indicators.keys()
        self._masks = {boundary_type: [np.zeros(grid.size(codim), dtype='bool') for codim in xrange(1, grid.dim + 1)]
                       for boundary_type in self.boundary_types}
        for boundary_type, codims in self._masks.iteritems():
            for c, mask in enumerate(codims):
                mask[grid.boundaries(c + 1)] = indicators[boundary_type](grid.centers(c + 1)[grid.boundaries(c + 1)])
        self.check_boundary_types(assert_unique_type=assert_unique_type, assert_some_type=assert_some_type)

    def mask(self, boundary_type, codim):
        assert 1 <= codim <= self.grid.dim
        return self._masks[boundary_type][codim - 1]


class AllDirichletBoundaryInfo(BoundaryInfoInterface):
    """|BoundaryInfo| where `BoundaryType('dirichlet')` is attached to each boundary entity."""

    def __init__(self, grid):
        self.grid = grid
        self.boundary_types = frozenset({BoundaryType('dirichlet')})

    def mask(self, boundary_type, codim):
        assert boundary_type == BoundaryType('dirichlet'), 'Has no boundary_type "{}"'.format(boundary_type)
        assert 1 <= codim <= self.grid.dim
        return np.ones(self.grid.size(codim), dtype='bool') * self.grid.boundary_mask(codim)


class SubGridBoundaryInfo(BoundaryInfoInterface):
    """Derives a |BoundaryInfo| for a :class:`~pymor.grids.subgrid.SubGrid`.

    Parameters
    ----------
    subrid
        The :class:`~pymor.grids.subgrid.SubGrid` for which a |BoundaryInfo| is created.
    grid
        The parent |Grid|.
    grid_boundary_info
        The |BoundaryInfo| of the parent |Grid| from which to derive the |BoundaryInfo|
    new_boundary_type
        The |BoundaryType| which is assigned to the new boundaries of `subgrid`. If
        `None`, no |BoundaryType| is assigned.
    """

    def __init__(self, subgrid, grid, grid_boundary_info, new_boundary_type=None):
        assert new_boundary_type is None or isinstance(new_boundary_type, BoundaryType)

        boundary_types = grid_boundary_info.boundary_types
        has_new_boundaries = False
        masks = []
        for codim in xrange(1, subgrid.dim + 1):
            parent_indices = subgrid.parent_indices(codim)[subgrid.boundaries(codim)]
            new_boundaries = np.where(np.logical_not(grid.boundary_mask(codim)[parent_indices]))
            if len(new_boundaries) > 0:
                has_new_boundaries = True
            m = {}
            for t in boundary_types:
                m[t] = grid_boundary_info.mask(t, codim)[subgrid.parent_indices(codim)]
                if t == new_boundary_type:
                    m[t][new_boundaries] = True
            if new_boundary_type is not None and new_boundary_type not in boundary_types:
                m[new_boundary_type] = np.zeros(subgrid.size(codim), dtype=np.bool)
                m[new_boundary_type][new_boundaries] = True
            masks.append(m)
        self.__masks = masks

        self.boundary_types = grid_boundary_info.boundary_types
        if has_new_boundaries and new_boundary_type is not None:
            self.boundary_types = self.boundary_types.union({new_boundary_type})

    def mask(self, boundary_type, codim):
        assert 1 <= codim < len(self.__masks) + 1, 'Invalid codimension'
        assert boundary_type in self.boundary_types
        return self.__masks[codim - 1][boundary_type]
