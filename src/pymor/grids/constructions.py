# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.tools.relations import inverse_relation


def flatten_grid(grid):
    '''This method is used by our visualizers to render n-dimensional grids which cannot
    be embedded into R^n by duplicating verticies which would have to be mapped to multiple
    points at once. (Think of grids on rectangular domains with identified edges.)

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
        Maps the indicies of the codim-`grid.dim` entities of the flattened
        grid to the indicies of the corresponding entities in the original grid.
    '''
    # first we determine which verticies are mapped to different coordinates when using the
    # embeddings of their codim-0 superentities
    dim = grid.dim
    global_coordinates = grid.embeddings(dim)[1]
    subentities = grid.subentities(0, dim)
    super_entities = grid.superentities(dim, 0)
    superentity_indices = grid.superentity_indices(dim, 0)
    A, B = grid.embeddings(0)
    ref_el_coordinates  = grid.reference_element.subentity_embedding(dim)[1]
    local_coordinates = np.einsum('eij,vj->evi', A, ref_el_coordinates) + B[:, np.newaxis, :]
    critical_verticies = np.unique(subentities[np.logical_not(np.all(np.isclose(global_coordinates[subentities],
                                                                                local_coordinates), axis=2))])
    del A
    del B

    # when there are critical verticies, we have to create additional verticies
    if len(critical_verticies) > 0:
        subentities = subentities.copy()
        supe = super_entities[critical_verticies]
        supi = superentity_indices[critical_verticies]
        coord = local_coordinates[supe, supi]

        new_points = np.ones_like(supe, dtype=np.int32) * -1
        new_points[:, 0] = critical_verticies
        num_points = grid.size(dim)
        entity_map = np.empty((0,), dtype=np.int32)
        for i in xrange(new_points.shape[1]):
            for j in xrange(i):
                new_points[:, i] = np.where(supe[:, i] == -1, new_points[:, i],
                                            np.where(np.all(np.isclose(coord[:, i], coord[:, j]), axis=1),
                                                     new_points[:, j], new_points[:, i]))
            new_point_inds = np.where(np.logical_and(new_points[:, i] == -1, supe[:, i] != -1))[0]
            new_points[new_point_inds, i] = np.arange(num_points, num_points + len(new_point_inds))
            num_points += len(new_point_inds)
            entity_map = np.hstack((entity_map, critical_verticies[new_point_inds]))

        entity_map = np.hstack((np.arange(grid.size(dim), dtype=np.int32), entity_map))

        # handle -1 entries in supe/supi correctly ...
        ci = np.where(critical_verticies == subentities[-1, -1])[0]
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
