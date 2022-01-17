# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import time

from pymor.core.config import config
from pymor.core.exceptions import MeshioMissing
from pymor.core.logger import getLogger
from pymor.discretizers.builtin.grids.boundaryinfos import GenericBoundaryInfo, EmptyBoundaryInfo
from pymor.discretizers.builtin.grids.unstructured import UnstructuredTriangleGrid


def load_gmsh(filename):
    """Parse a Gmsh file and create a corresponding :class:`GmshGrid` and :class:`GmshBoundaryInfo`.

    Parameters
    ----------
    filename
        Path of the Gmsh MSH-file.

    Returns
    -------
    grid
        The generated :class:`GmshGrid`.
    boundary_info
        The generated :class:`GmshBoundaryInfo`.
    """
    if not config.HAVE_MESHIO:
        raise MeshioMissing('meshio (>=4) is required for reading Gmsh files.')
    import meshio

    logger = getLogger('pymor.discretizers.builtin.grids.gmsh.load_gmsh')

    logger.info('Parsing Gmsh file ...')
    tic = time.perf_counter()
    data = meshio.read(filename)
    toc = time.perf_counter()
    t_parse = toc - tic

    if data.gmsh_periodic:
        raise NotImplementedError
    cells_dict = data.cells_dict
    if not cells_dict.keys() <= {'line', 'triangle'}:
        raise NotImplementedError
    if not np.all(data.points[:, 2] == 0):
        raise NotImplementedError

    logger.info('Create Grid ...')
    tic = time.perf_counter()

    vertices = data.points[:, :2]
    faces = cells_dict['triangle']

    grid = UnstructuredTriangleGrid.from_vertices(vertices, faces)
    toc = time.perf_counter()
    t_grid = toc - tic

    logger.info('Create GmshBoundaryInfo ...')
    tic = time.perf_counter()

    boundary_types = {k: v[0] for k, v in data.field_data.items() if v[1] == 1}

    cell_data_dict = data.cell_data_dict
    if 'line' in cells_dict and 'gmsh:physical' in cell_data_dict and 'line' in cell_data_dict['gmsh:physical']:
        superentities = grid.superentities(2, 1)

        # find the edge for given vertices.
        def find_edge(vertices):
            edge_set = set(superentities[vertices[0]]).intersection(superentities[vertices[1]]) - {-1}
            if len(edge_set) != 1:
                raise ValueError
            return next(iter(edge_set))

        line_ids = np.array([find_edge(l) for l in cells_dict['line']])

        # compute boundary masks for all boundary types.
        masks = {}
        for bt, bt_id in boundary_types.items():
            masks[bt] = [np.zeros(grid.size(1), dtype=bool), np.zeros(grid.size(2), dtype=bool)]
            masks[bt][0][line_ids] = cell_data_dict['gmsh:physical']['line'] == bt_id
            vtx = np.sort(grid.subentities(1, 2)[np.where(masks[bt][0])])
            masks[bt][1][vtx] = True

        bi = GenericBoundaryInfo(grid, masks)
    else:
        logger.warning('Boundary data not found. Creating empty BoundaryInfo ...')
        bi = EmptyBoundaryInfo(grid)

    toc = time.perf_counter()
    t_bi = toc - tic

    logger.info(f'Parsing took {t_parse}s; Grid creation took {t_grid}s; BoundaryInfo creation took {t_bi}s')

    return grid, bi
