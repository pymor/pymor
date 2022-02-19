# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import weakref

import numpy as np

from pymor.discretizers.builtin.grids.boundaryinfos import GenericBoundaryInfo
from pymor.discretizers.builtin.grids.interfaces import Grid


class SubGrid(Grid):
    """A subgrid of a |Grid|.

    Given a |Grid| and a list of codim-0 entities we construct the minimal
    subgrid of the grid, containing all the given entities.

    Parameters
    ----------
    parent_grid
        |Grid| of which a subgrid is to be created.
    parent_entities
        |NumPy array| of global indices of the codim-0 entities which
        are to be contained in the subgrid.

    Attributes
    ----------
    parent_grid
        The |Grid| from which the subgrid was constructed. :class:`Subgrid`
        only stores a :mod:`weakref` to the grid, so accessing this property
        might return `None` if the original grid has been destroyed.
    """

    reference_element = None

    def __init__(self, parent_grid, parent_entities):
        assert parent_grid is not None, \
            'parent_grid is None. Maybe you have called sub_grid.with(parent_entities=e)\n' \
            'on a SubGrid for which the parent grid has been destroyed?'
        assert isinstance(parent_grid, Grid)
        self.dim = parent_grid.dim
        self.reference_element = parent_grid.reference_element

        parent_indices = [np.array(np.unique(parent_entities), dtype=np.int32)]
        assert len(parent_indices[0] == len(parent_entities))

        subentities = [np.arange(len(parent_indices[0]), dtype=np.int32).reshape((-1, 1))]

        for codim in range(1, self.dim + 1):
            SUBE = parent_grid.subentities(0, codim)[parent_indices[0]]
            if np.any(SUBE < 0):
                raise NotImplementedError
            UI, UI_inv = np.unique(SUBE, return_inverse=True)
            subentities.append(np.array(UI_inv.reshape(SUBE.shape), dtype=np.int32))
            parent_indices.append(np.array(UI, dtype=np.int32))

        self.parent_entities = parent_entities
        self.__parent_grid = weakref.ref(parent_grid)
        self.__parent_indices = parent_indices
        self.__subentities = subentities
        embeddings = parent_grid.embeddings(0)
        self.__embeddings = (embeddings[0][parent_indices[0]], embeddings[1][parent_indices[0]])

    @property
    def parent_grid(self):
        return self.__parent_grid()

    def parent_indices(self, codim):
        """`retval[e]` is the index of the `e`-th codim-`codim` entity in the parent grid."""
        assert 0 <= codim <= self.dim, 'Invalid codimension'
        return self.__parent_indices[codim]

    def indices_from_parent_indices(self, ind, codim):
        """Maps indices of codim-`codim` entities of the parent grid to indices of the subgrid.

        Parameters
        ----------
        ind
            |NumPy array| of indices of codim-`codim` entities of the parent grid
        codim
            codim of the entities indicated by `ind`

        Raises
        ------
        ValueError
            Not all provided indices correspond to entities contained in the subgrid.
        """
        assert 0 <= codim <= self.dim, 'Invalid codimension'
        ind = ind.ravel()
        # TODO Find better implementation of the following
        R = np.argmax(ind[:, np.newaxis] - self.__parent_indices[codim][np.newaxis, :] == 0, axis=1)
        if not np.all(self.__parent_indices[codim][R] == ind):
            raise ValueError('Not all parent indices found')
        return np.array(R, dtype=np.int32)

    def size(self, codim):
        assert 0 <= codim <= self.dim, 'Invalid codimension'
        return len(self.__parent_indices[codim])

    def subentities(self, codim, subentity_codim):
        if codim == 0:
            assert codim <= subentity_codim <= self.dim, 'Invalid subentity codimension'
            return self.__subentities[subentity_codim]
        else:
            return super().subentities(codim, subentity_codim)

    def embeddings(self, codim):
        if codim == 0:
            return self.__embeddings
        else:
            return super().embeddings(codim)

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_SubGrid__parent_grid']
        return d


def make_sub_grid_boundary_info(sub_grid, parent_grid, parent_grid_boundary_info, new_boundary_type=None):
    """Derives a |BoundaryInfo| for a :class:`~pymor.discretizers.builtin.grids.subgrid.SubGrid`.

    Parameters
    ----------
    sub_grid
        The :class:`~pymor.discretizers.builtin.grids.subgrid.SubGrid` for which a
        |BoundaryInfo| is created.
    parent_grid
        The parent |Grid|.
    parent_grid_boundary_info
        The |BoundaryInfo| of the parent |Grid| from which to derive the |BoundaryInfo|
    new_boundary_type
        The boundary type which is assigned to the new boundaries of `subgrid`. If
        `None`, no boundary type is assigned.

    Returns
    -------
    |BoundaryInfo| associated with sub_grid.
    """
    boundary_types = parent_grid_boundary_info.boundary_types

    masks = {}
    for codim in range(1, sub_grid.dim + 1):
        parent_indices = sub_grid.parent_indices(codim)[sub_grid.boundaries(codim)]
        new_boundaries = np.where(np.logical_not(parent_grid.boundary_mask(codim)[parent_indices]))
        new_boundaries_sg_indices = sub_grid.boundaries(codim)[new_boundaries]
        for t in boundary_types:
            m = parent_grid_boundary_info.mask(t, codim)[sub_grid.parent_indices(codim)]
            if t == new_boundary_type:
                m[new_boundaries_sg_indices] = True
            if codim == 1:
                masks[t] = [m]
            else:
                masks[t].append(m)
        if new_boundary_type is not None and new_boundary_type not in boundary_types:
            m = np.zeros(sub_grid.size(codim), dtype=np.bool)
            m[new_boundaries_sg_indices] = True
            if codim == 1:
                masks[new_boundary_type] = [m]
            else:
                masks[new_boundary_type].append(m)

    return GenericBoundaryInfo(sub_grid, masks)
