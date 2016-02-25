# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import weakref

import numpy as np

from pymor.grids.interfaces import AffineGridInterface


class SubGrid(AffineGridInterface):
    """A subgrid of a |Grid|.

    Given a |Grid| and a list of codim-0 entities we construct the minimal
    subgrid of the grid, containing all the given entities.

    Parameters
    ----------
    grid
        |Grid| of which a subgrid is to be created.
    entities
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

    def __init__(self, grid, entities):
        assert isinstance(grid, AffineGridInterface)
        self.dim = grid.dim
        self.dim_outer = grid.dim_outer
        self.reference_element = grid.reference_element

        parent_indices = [np.array(np.unique(entities), dtype=np.int32)]
        assert len(parent_indices[0] == len(entities))

        subentities = [np.arange(len(parent_indices[0]), dtype=np.int32).reshape((-1, 1))]

        for codim in range(1, self.dim + 1):
            SUBE = grid.subentities(0, codim)[parent_indices[0]]
            if np.any(SUBE < 0):
                raise NotImplementedError
            UI, UI_inv = np.unique(SUBE, return_inverse=True)
            subentities.append(np.array(UI_inv.reshape(SUBE.shape), dtype=np.int32))
            parent_indices.append(np.array(UI, dtype=np.int32))

        self.__parent_grid = weakref.ref(grid)
        self.__parent_indices = parent_indices
        self.__subentities = subentities
        embeddings = grid.embeddings(0)
        self.__embeddings = (embeddings[0][parent_indices[0]], embeddings[1][parent_indices[0]])

    @property
    def parent_grid(self):
        return self.__parent_grid()

    def parent_indices(self, codim):
        """`retval[e]` is the index of the `e`-th codim-`codim` entity in the parent grid."""
        assert 0 <= codim <= self.dim, 'Invalid codimension'
        return self.__parent_indices[codim]

    def indices_from_parent_indices(self, ind, codim):
        """Maps a |NumPy array| of indicies of codim-`codim` entites of the parent grid to indicies of the subgrid.

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
            return super(SubGrid, self).subentities(codim, subentity_codim)

    def embeddings(self, codim):
        if codim == 0:
            return self.__embeddings
        else:
            return super(SubGrid, self).embeddings(codim)

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_SubGrid__parent_grid']
        return d
