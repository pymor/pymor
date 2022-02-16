# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.tools.floatcmp import float_cmp
from pymor.discretizers.builtin.grids.rect import RectGrid
from pymor.discretizers.builtin.grids.tria import TriaGrid
from pymor.discretizers.builtin.relations import inverse_relation


def flatten_grid(grid):
    """Flatten a |Grid|.

    This method is used by our visualizers to render n-dimensional grids which cannot
    be embedded into R^n by duplicating vertices which would have to be mapped to multiple
    points at once (think of grids on rectangular domains with identified edges).

    Parameters
    ----------
    grid
        The |Grid| to flatten.

    Returns
    -------
    subentities
        The `subentities(0, grid.dim)` relation for the flattened grid.
    coordinates
        The coordinates of the codim-`grid.dim` entities.
    entity_map
        Maps the indices of the codim-`grid.dim` entities of the flattened
        grid to the indices of the corresponding entities in the original grid.
    """
    # special handling of known flat grids
    if isinstance(grid, (RectGrid, TriaGrid)) and not grid.identify_left_right and not grid.identify_bottom_top:
        subentities = grid.subentities(0, grid.dim)
        coordinates = grid.centers(grid.dim)
        entity_map = np.arange(grid.size(grid.dim), dtype=np.int32)
        return subentities, coordinates, entity_map

    # first we determine which vertices are mapped to different coordinates when using the
    # embeddings of their codim-0 superentities
    dim = grid.dim
    global_coordinates = grid.embeddings(dim)[1]
    subentities = grid.subentities(0, dim)
    super_entities = grid.superentities(dim, 0)
    superentity_indices = grid.superentity_indices(dim, 0)
    A, B = grid.embeddings(0)
    ref_el_coordinates = grid.reference_element.subentity_embedding(dim)[1]
    local_coordinates = np.einsum('eij,vj->evi', A, ref_el_coordinates) + B[:, np.newaxis, :]
    critical_vertices = np.unique(subentities[np.logical_not(np.all(float_cmp(global_coordinates[subentities],
                                                                              local_coordinates), axis=2))])
    del A
    del B

    # when there are critical vertices, we have to create additional vertices
    if len(critical_vertices) > 0:
        subentities = subentities.copy()
        supe = super_entities[critical_vertices]
        supi = superentity_indices[critical_vertices]
        coord = local_coordinates[supe, supi]

        new_points = np.ones_like(supe, dtype=np.int32) * -1
        new_points[:, 0] = critical_vertices
        num_points = grid.size(dim)
        entity_map = np.empty((0,), dtype=np.int32)
        for i in range(new_points.shape[1]):
            for j in range(i):
                new_points[:, i] = np.where(supe[:, i] == -1, new_points[:, i],
                                            np.where(np.all(float_cmp(coord[:, i], coord[:, j]), axis=1),
                                                     new_points[:, j], new_points[:, i]))
            new_point_inds = np.where(np.logical_and(new_points[:, i] == -1, supe[:, i] != -1))[0]
            new_points[new_point_inds, i] = np.arange(num_points, num_points + len(new_point_inds))
            num_points += len(new_point_inds)
            entity_map = np.hstack((entity_map, critical_vertices[new_point_inds]))

        entity_map = np.hstack((np.arange(grid.size(dim), dtype=np.int32), entity_map))

        # handle -1 entries in supe/supi correctly ...
        ci = np.where(critical_vertices == subentities[-1, -1])[0]
        if len(ci) > 0:
            assert len(ci) == 1
            ci = ci[0]
            i = np.where(supe[ci] == (grid.size(0) - 1))[0]
            if len(i) > 0:
                assert len(i) == 1
                i = i[0]
                new_points[supe == -1] = new_points[ci, i]
            else:
                new_points[supe == -1] = subentities[-1, -1]
        else:
            new_points[supe == -1] = subentities[-1, -1]
        subentities[supe, supi] = new_points
        super_entities, superentity_indices = inverse_relation(subentities, size_rhs=num_points, with_indices=True)
        coordinates = local_coordinates[super_entities[:, 0], superentity_indices[:, 0]]
    else:
        coordinates = global_coordinates
        entity_map = np.arange(grid.size(dim), dtype=np.int32)

    return subentities, coordinates, entity_map
