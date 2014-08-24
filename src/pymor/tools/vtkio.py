# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

try:
    from pyvtk import (VtkData, UnstructuredGrid, PointData, CellData, Scalars)
    HAVE_PYVTK = True
except ImportError:
    HAVE_PYVTK = False

import numpy as np

from pymor.grids import referenceelements
from pymor.grids.constructions import flatten_grid


def _write_meta_file(filename_base, steps, fn_tpl):
    """Outputs a collection file for a series of vtu files

    This DOES NOT WORK for the currently used legacy vtk format below
    """

    pvd_header = '''<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
    <Collection>
'''
    pvd_footer = '''
    </Collection>
</VTKFile>'''

    fn_tpl += '.vtu'
    with open('{}.pvd'.format(filename_base), 'wb') as pvd:
        pvd.write(pvd_header)
        for step in xrange(steps):
            fn = fn_tpl.format(filename_base, step)
            pvd.write('\t\t<DataSet timestep="{}" group="" part="0" file="{}" />\n'.format(step, fn))
        pvd.write(pvd_footer)


def _vtk_grid(reference_element, subentities, coords):
    if reference_element not in (referenceelements.triangle, referenceelements.square):
        raise NotImplementedError
    subentity_ordering = subentities.tolist()
    num_points = len(coords[0])
    points = [[coords[0][i], coords[1][i], coords[2][i]] for i in xrange(num_points)]
    if reference_element == referenceelements.triangle:
        return UnstructuredGrid(points, triangle=subentity_ordering)
    else:
        return UnstructuredGrid(points=points, quad=subentity_ordering)


def _data_item(is_cell_data, data, step):
    sd = data[step, :]
    if is_cell_data:
        return CellData(Scalars(sd, 'cell_data', lookup_table='default'))
    return PointData(Scalars(sd, 'vertex_data'))


def _write_vtu_series(us_grid, data, filename_base, binary_vtk, last_step, is_cell_data):
    steps = last_step + 1 if last_step is not None else len(data)
    fn_tpl = "{}_{:08d}"
    _write_meta_file(filename_base, steps, fn_tpl)
    for i in xrange(steps):
        fn = fn_tpl.format(filename_base, i)
        pd = _data_item(is_cell_data, data, i)
        vtk = VtkData(us_grid, pd, 'Unstructured Grid Example')
        if binary_vtk:
            vtk.tofile(fn, 'binary')
        else:
            vtk.tofile(fn)


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
        raise ImportError('could not import pyvtk')
    if grid.dim != 2 or grid.dim_outer != 2:
        raise NotImplementedError
    if codim not in (0, 2):
        raise NotImplementedError

    subentities, coordinates, entity_map = flatten_grid(grid)

    x, y = coordinates[:, 0], coordinates[:, 1]
    z = np.zeros(len(x))
    coords = (x, y, z)
    us_grid = _vtk_grid(grid.reference_element, subentities, coords)
    if codim == 0:
        _write_vtu_series(us_grid, data.data, filename_base, binary_vtk, last_step, True)
    else:
        _write_vtu_series(us_grid, data.data[:, entity_map], filename_base, binary_vtk, last_step, False)
