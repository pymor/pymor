# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pyvtk import (VtkData, UnstructuredGrid, PointData, CellData, Scalars)
import numpy as np

from pymor.grids import referenceelements

def _write_meta_file(filename_base, steps, fn_tpl):
    '''Outputs a collection file for a series of vtu files
    
    This DOES NOT WORK for the currently used legacy vtk format below
    '''

    pvd_header = '''<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
    <Collection>
'''
    pvd_footer = '''
    </Collection>
</VTKFile>'''

    fn_tpl += '.vtu'
    with open('{}.pvd'.format(filename_base), 'wb') as pvd:
        pvd.write( pvd_header )
        for step in xrange(steps):
            fn = fn_tpl.format(filename_base, step)
            pvd.write('\t\t<DataSet timestep="{}" group="" part="0" file="{}" />\n'.format(step, fn))
        pvd.write( pvd_footer )

def _vtk_grid(grid, coords):
    subentity_ordering = grid.subentities(0, 2).tolist()
    num_points = len(coords[0])
    points = [[coords[0][i], coords[1][i], coords[2][i]] for i in xrange(num_points)]
    if grid.reference_element == referenceelements.triangle:
        return UnstructuredGrid(points, triangle=subentity_ordering)
    else:
        return UnstructuredGrid(points=points, quad=subentity_ordering)


def _data_item(is_cell_data, data, step):
    sd = data.data[step, :]
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


def write_vtk(grid, data, filename_base, binary_vtk=True, last_step=None):
    '''Output grid-associated data in (legacy) vtk format
    
    Parameters
    ---------
    grid
        a pymor grid with triangular or rectilinear reference element
        
    data
        VectorArrayInterface instance with either cell (ie one datapoint per codim 0 entity) 
        or vertex (ie one datapoint per codim 2 entity) data in each array element
        
    filename_base
        common component for output files in timeseries
        
    last_step
        if set must be <= len(data) to restrict output of timeseries
    '''
    x, y = grid.centers(2)[:, 0], grid.centers(2)[:, 1]
    z = np.zeros(len(x))
    coords = (x, y, z)
    us_grid = _vtk_grid(grid, coords)
    shape = data.data[0, :].shape
    if shape[0] == grid.size(0):
        _write_vtu_series(us_grid, data, filename_base, binary_vtk, last_step, True)
    elif shape[0] == grid.size(2):
        _write_vtu_series(us_grid, data, filename_base, binary_vtk, last_step, False)
    else:
        raise Exception()
