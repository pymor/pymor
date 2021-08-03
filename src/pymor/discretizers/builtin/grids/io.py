# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.config import config
from pymor.discretizers.builtin.grids import referenceelements
from pymor.discretizers.builtin.grids.constructions import flatten_grid


def to_meshio(grid, codim, data=None, scalar_name='Data'):
    """Transform given |Grid| and |VectorArray| data into a sequence of meshio.Mesh objects

    Parameters
    ----------
    grid
        A |Grid| with triangular or rectilinear reference element.
    data
        |VectorArray| with either cell (ie one datapoint per codim 0 entity)
        or vertex (ie one datapoint per codim 2 entity) data in each array element.
        If data is None, a meshio.Mesh with a dummy array of associated data
    codim
        the codimension associated with the data
    scalar_name
        optional string acting as a name in the meshio data structure's dict

    Returns
    -------
    list of meshio.Mesh objects if data is a |VectorArray|
    tuple of list of meshio.Mesh objects if data is a tuple of |VectorArray|
    """
    if not config.HAVE_MESHIO:
        raise ImportError('Missing meshio')
    import meshio

    if isinstance(data, tuple):
        return tuple(to_meshio(grid, data=d, codim=codim) for d in data)

    subentities, coordinates, entity_map = flatten_grid(grid)
    is_cell_data = (codim == 0)

    ref = grid.reference_element
    if ref is referenceelements.triangle:
        cells = [("triangle", subentities), ]
    elif ref is referenceelements.square:
        cells = [("quad", subentities), ]
    else:
        raise NotImplementedError("Meshio conversion only available for grids with triangle "
                                  "or rectangle reference elements")

    if data is None:
        if is_cell_data:
            data = np.zeros((1, grid.size(0)))
        else:
            data = np.zeros((1, grid.size(2)))
    else:
        data = data.to_numpy() if codim == 0 else data.to_numpy()[:, entity_map].copy()
    meshes = []
    for i in range(len(data)):
        if is_cell_data:
            meshes.append(meshio.Mesh(coordinates, cells, cell_data={scalar_name: [data[i, :]]}))
        else:
            meshes.append(meshio.Mesh(coordinates, cells, point_data={scalar_name: data[i, :]}))
    return meshes
