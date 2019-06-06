# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.grids.interfaces import BoundaryInfoInterface


class EmptyBoundaryInfo(BoundaryInfoInterface):
    """|BoundaryInfo| with no boundary types attached to any boundary."""

    def __init__(self, grid):
        self.grid = grid
        self.boundary_types = frozenset()

    def mask(self, boundary_type, codim):
        assert False, f'Has no boundary_type "{boundary_type}"'


class GenericBoundaryInfo(BoundaryInfoInterface):
    """Generic |BoundaryInfo| storing entity masks per boundary type."""

    def __init__(self, grid, masks, assert_unique_type=None, assert_some_type=None):
        self.assert_unique_type = assert_unique_type if assert_unique_type else [1]
        self.assert_some_type = assert_some_type if assert_some_type else []
        self.grid = grid
        self.masks = masks
        self.boundary_types = frozenset(masks)
        self.check_boundary_types(
            assert_unique_type=self.assert_unique_type,
            assert_some_type=self.assert_some_type,
        )

    @classmethod
    def from_indicators(
        cls, grid, indicators, assert_unique_type=None, assert_some_type=None
    ):
        """Create |BoundaryInfo| from indicator functions.

        Parameters
        ----------
        grid
            The |Grid| to which the |BoundaryInfo| is associated.
        indicators
            Dict where each key is a boundary type and the corresponding value is a boolean
            valued function defined on the analytical domain which indicates if a point belongs
            to a boundary of the given boundary type (the indicator functions must be vectorized).
        """
        masks = {
            boundary_type: [
                np.zeros(grid.size(codim), dtype="bool")
                for codim in range(1, grid.dim + 1)
            ]
            for boundary_type in indicators
        }
        for boundary_type, codims in masks.items():
            for c, mask in enumerate(codims):
                mask[grid.boundaries(c + 1)] = indicators[boundary_type](
                    grid.centers(c + 1)[grid.boundaries(c + 1)]
                )
        return cls(
            grid,
            masks,
            assert_unique_type=assert_unique_type,
            assert_some_type=assert_some_type,
        )

    def mask(self, boundary_type, codim):
        assert 1 <= codim <= self.grid.dim
        return self.masks[boundary_type][codim - 1]


class AllDirichletBoundaryInfo(BoundaryInfoInterface):
    """|BoundaryInfo| where the boundary type 'dirichlet' is attached to each boundary entity."""

    def __init__(self, grid):
        self.grid = grid
        self.boundary_types = frozenset({"dirichlet"})

    def mask(self, boundary_type, codim):
        assert boundary_type == "dirichlet", f'Has no boundary_type "{boundary_type}"'
        assert 1 <= codim <= self.grid.dim
        return np.ones(self.grid.size(codim), dtype="bool") * self.grid.boundary_mask(
            codim
        )
