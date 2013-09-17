# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pyvtk import (VtkData, UnstructuredGrid, Vectors, PointData, Scalars)
import numpy as np

from pymor.grids import referenceelements

def _triangle_data_to_vtk(subentity_ordering, coords, data, filename_base, binary_vtk, last_step):
    num_points = len(coords[0])
    points = [[coords[0][i], coords[1][i], coords[2][i]] for i in xrange(num_points)]
    dummy = Vectors([[1, 1, 1] for _ in xrange(num_points)])
    us_grid = UnstructuredGrid(points, triangle=subentity_ordering)

    steps = last_step + 1 if last_step is not None else data.shape[0]
    for i in xrange(steps):
        fn = "{}_{:08d}".format(filename_base, i)
        pd = PointData(dummy, Scalars(data))
        vtk = VtkData(us_grid, pd, 'Unstructured Grid Example')
        if binary_vtk:
            vtk.tofile(fn, 'binary')
        else:
            vtk.tofile(fn)


def write_vtk(grid, data, filename_base, binary_vtk=True, last_step=None):
    if grid.reference_element == referenceelements.triangle:
        x, y = grid.centers(2)[:, 0], grid.centers(2)[:, 1]
        z = np.zeros(len(x))
        _triangle_data_to_vtk(grid.subentities(0, 2).tolist(), (x, y, z), data.data[0, :],
                              filename_base, binary_vtk, last_step)
    else:
        raise Exception()
