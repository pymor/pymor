# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
from pymor.discretizers.builtin.grids import referenceelements
from pymor.discretizers.builtin.grids.constructions import flatten_grid


def _write_vtu_series(grid, coordinates, connectivity, data, filename_base, last_step, is_cell_data):
    from pyevtk.vtk import VtkGroup
    import meshio
    steps = last_step + 1 if last_step is not None else len(data)
    fn_tpl = '{}_{:08d}.vtu'

    ref = grid.reference_element
    if ref is referenceelements.triangle:
        cells = [("triangle", connectivity), ]
    elif ref is referenceelements.square:
        cells = [("quad", connectivity), ]
    else:
        raise NotImplementedError("vtk output only available for grids with triangle or rectangle reference elments")

    group = VtkGroup(filename_base)
    for i in range(steps):
        fn = fn_tpl.format(filename_base, i)
        if is_cell_data:
            mesh = meshio.Mesh(coordinates, cells, cell_data={"Data": [data[i, :]]})
        else:
            mesh = meshio.Mesh(coordinates, cells, point_data={"Data": data[i, :]})
        mesh.write(fn)
        group.addFile(filepath=fn, sim_time=i)
    group.save()
    return f'{filename_base}.pvd'


def write_vtk(grid, data, filename_base, codim=2, binary_vtk=True, last_step=None):
    """Output grid-associated data in vtk format

    Parameters
    ----------
    grid
        A |Grid| with triangular or rectilinear reference element.
    data
        |VectorArray| with either cell (ie one datapoint per codim 0 entity)
        or vertex (ie one datapoint per codim 2 entity) data in each array element.
    codim
        the codimension associated with the data
    filename_base
        common component for output files in timeseries
    binary_vtk
        if false, output files contain human readable inline ascii data, else appended binary
    last_step
        if set must be <= len(data) to restrict output of timeseries

    Returns
    -------
    full filename of saved file
    """
    if not config.HAVE_VTKIO:
        raise ImportError('pyevtk, meshio, lxml and xmljson needed for vtk output')
    if grid.dim != 2:
        raise NotImplementedError
    if codim not in (0, 2):
        raise NotImplementedError
    if not binary_vtk:
        raise NotImplementedError

    subentities, coordinates, entity_map = flatten_grid(grid)
    data = data.to_numpy() if codim == 0 else data.to_numpy()[:, entity_map].copy()
    return _write_vtu_series(grid, coordinates=coordinates, connectivity=subentities, data=data,
                             filename_base=filename_base, last_step=last_step, is_cell_data=(codim == 0))
