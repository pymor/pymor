# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
from pymor.discretizers.builtin.grids import referenceelements
from pymor.discretizers.builtin.grids.constructions import flatten_grid


def to_meshio(grid, data, codim=2):
    """Transform given |Grid| and |VectorArray| data into a sequence of meshio.Mesh objects

    Parameters
    ----------
    grid
        A |Grid| with triangular or rectilinear reference element.
    data
        |VectorArray| with either cell (ie one datapoint per codim 0 entity)
        or vertex (ie one datapoint per codim 2 entity) data in each array element.
    codim
        the codimension associated with the data

    Returns
    -------
    list of meshio.Mesh objects
    """
    if not config.HAVE_MESHIO:
        raise ImportError('Missing meshio')
    import meshio

    subentities, coordinates, entity_map = flatten_grid(grid)
    data = data.to_numpy() if codim == 0 else data.to_numpy()[:, entity_map].copy()
    is_cell_data = (codim == 0)

    ref = grid.reference_element
    if ref is referenceelements.triangle:
        cells = [("triangle", subentities), ]
    elif ref is referenceelements.square:
        cells = [("quad", subentities), ]
    else:
        raise NotImplementedError("Meshio conversion restricted to grid with triangle or rectangle reference elements")

    meshes = []
    for i in range(len(data)):
        if is_cell_data:
            meshes.append(meshio.Mesh(coordinates, cells, cell_data={"Data": [data[i, :]]}))
        else:
            meshes.append(meshio.Mesh(coordinates, cells, point_data={"Data": data[i, :]}))
    return meshes
