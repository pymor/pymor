# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
config.require('VTKIO')


from pymor.discretizers.builtin.grids.io import to_meshio
from pymor.tools.io.vtk import write_vtk_collection


def write_vtk(grid, data, filename_base, codim=2, metadata=None):
    """Convenience function around write_vtk_collection(to_meshio(...))"""
    return write_vtk_collection(meshes=to_meshio(grid, data, codim=codim),
                                filename_base=filename_base, metadata=metadata)
