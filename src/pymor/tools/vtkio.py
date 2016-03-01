# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

try:
    from evtk.hl import gridToVTK, pointsToVTK, _addDataToFile, _appendDataToFile
    from evtk.vtk import VtkGroup, VtkFile, VtkUnstructuredGrid, VtkVertex, VtkTriangle, VtkQuad

    HAVE_PYVTK = True
except ImportError:
    HAVE_PYVTK = False

import numpy as np

from pymor.grids import referenceelements
from pymor.grids.constructions import flatten_grid


def _write_vtu_series(grid, coordinates, connectivity, data, filename_base, last_step, is_cell_data):
    steps = last_step + 1 if last_step is not None else len(data)
    fn_tpl = "{}_{:08d}"

    npoints = grid.size(2)
    ncells = grid.size(0)

    ref = grid.reference_element
    if ref is ref is referenceelements.triangle:
        points_per_cell = 3
        vtk_el_type = VtkTriangle.tid
    elif ref is referenceelements.square:
        points_per_cell = 4
        vtk_el_type = VtkQuad.tid
    else:
        raise NotImplementedError("vtk output only available for grids with triangle or rectangle reference elments")

    connectivity = connectivity.reshape(-1)
    cell_types = np.empty(ncells, dtype='uint8')
    cell_types[:] = vtk_el_type
    offsets = np.arange(start=points_per_cell, stop=ncells*points_per_cell+1, step=points_per_cell, dtype='int32')

    group = VtkGroup(filename_base)
    for i in range(steps):
        fn = fn_tpl.format(filename_base, i)
        vtk_data = data[i, :]
        w = VtkFile(fn, VtkUnstructuredGrid)
        w.openGrid()
        w.openPiece(ncells=ncells, npoints=npoints)

        w.openElement("Points")
        w.addData("Coordinates", coordinates)
        w.closeElement("Points")
        w.openElement("Cells")
        w.addData("connectivity", connectivity)
        w.addData("offsets", offsets)
        w.addData("types", cell_types)
        w.closeElement("Cells")
        if is_cell_data:
            _addDataToFile(w, cellData={"Data": vtk_data}, pointData=None)
        else:
            _addDataToFile(w, cellData=None, pointData={"Data": vtk_data})

        w.closePiece()
        w.closeGrid()
        w.appendData(coordinates)
        w.appendData(connectivity).appendData(offsets).appendData(cell_types)
        if is_cell_data:
            _appendDataToFile(w, cellData={"Data": vtk_data}, pointData=None)
        else:
            _appendDataToFile(w, cellData=None, pointData={"Data": vtk_data})

        w.save()
        group.addFile(filepath=fn, sim_time=i)
    group.save()

def write_vtk(grid, data, filename_base, codim=2, binary_vtk=True, last_step=None):
    """Output grid-associated data in (legacy) vtk format

    Parameters
    ----------
    grid
        a |Grid| with triangular or rectilinear reference element

    data
        VectorArrayInterface instance with either cell (ie one datapoint per codim 0 entity)
        or vertex (ie one datapoint per codim 2 entity) data in each array element

    filename_base
        common component for output files in timeseries

    last_step
        if set must be <= len(data) to restrict output of timeseries
    """
    if not HAVE_PYVTK:
        raise ImportError('could not import pyevtk')
    if grid.dim != 2 or grid.dim_outer != 2:
        raise NotImplementedError
    if codim not in (0, 2):
        raise NotImplementedError

    subentities, coordinates, entity_map = flatten_grid(grid)
    x, y, z = coordinates[:, 0].copy(), coordinates[:, 1].copy(), np.zeros(coordinates[:, 1].size)
    _write_vtu_series(grid, coordinates=(x, y, z), connectivity=subentities, data=data.data,
                      filename_base=filename_base, last_step=last_step, is_cell_data=(codim == 0))

