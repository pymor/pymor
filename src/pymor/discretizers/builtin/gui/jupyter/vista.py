# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from math import ceil
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from pymor.core.config import is_jupyter
from pymor.discretizers.builtin.grids.io import to_meshio
from pymor.discretizers.builtin.grids.referenceelements import triangle, square
from pymor.vectorarrays.interface import VectorArray


def _normalize(u, vmin=None, vmax=None):
    # rescale to be in [max(0,vmin), min(1,vmax)], scale nan to be the smallest value
    vmin = np.nanmin(u) if vmin is None else vmin
    vmax = np.nanmax(u) if vmax is None else vmax
    u -= vmin
    if (vmax - vmin) > 0:
        u /= float(vmax - vmin)
    return np.nan_to_num(u)


def visualize_vista(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, columns=2, color_map='viridis'):
    """Generate a pyvista Plot and associated controls for grid-based data

    The grid's |ReferenceElement| must be the triangle or square. The data can either
    be attached to the faces or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    U
        |VectorArray| of the data to visualize. If `len(U) 1`, the data is visualized
        as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
        provided, in which case a subplot is created for each entry of the tuple. The
        lengths of all arrays have to agree.
    bounding_box
        A bounding box in which the grid is contained.
    codim
        The codimension of the entities the data in `U` is attached to (either 0 or 2).
    title
        Title of the plot.
    legend
        Description of the data that is plotted. Most useful if `U` is a tuple in which
        case `legend` has to be a tuple of strings of the same length.
    separate_colorbars
        If `True`, use separate colorbars for each subplot.
    rescale_colorbars
        If `True`, rescale colorbars to data in each frame.
    columns
        The number of columns in the visualizer GUI in case multiple plots are displayed
        at the same time.
    color_map
        name of a Matplotlib Colormap (default: viridis)
    """
    assert isinstance(U, VectorArray) \
           or (isinstance(U, tuple) and all(isinstance(u, VectorArray) for u in U)
               and all(len(u) == len(U[0]) for u in U))
    meshes = to_meshio(grid, data=U, codim=codim)
    return visualize_vista_mesh(meshes, bounding_box, codim, title, legend, separate_colorbars, rescale_colorbars,
                                columns, color_map)


class _JupyterMultiPlotter:
    """Jupyter Analog for the pyvistqt.plotting.MultiPlotter

    We do not actually have multiple Plotters here, but fake the same interface with
    a single instance using the plotter's subplot mechanism
    """

    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        **kwargs
    ):
        self._nrows = nrows
        self._ncols = ncols
        self._plotter = pv.Plotter(shape=(nrows, ncols), notebook=True, **kwargs)
        self._meshes = [None] * (self._nrows * self._ncols)

    def show(self, **kwargs):
        return self._plotter.show(**kwargs)

    def close(self):
        pass

    def __setitem__(self, idx, plotter):
        raise NotImplementedError

    def __getitem__(self, idx):
        row, col = idx
        print(f'SUbplot {idx}')
        self._plotter.subplot(row, col)
        return self._plotter


def visualize_vista_mesh(meshes, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                         separate_colorbars=False, rescale_colorbars=False, columns=2, color_map='viridis'):
    from pyvista.utilities.fileio import from_meshio
    from pyvistaqt import MultiPlotter

    render_size = (300, 300)
    # themes are loadable from json files, would make for a nice customization point for users
    my_theme = pv.themes.DocumentTheme()
    my_theme.cmap = color_map
    my_theme.cpos = 'xy'
    my_theme.show_edges = True
    my_theme.interactive = True
    my_theme.transparent_background = True
    my_theme.jupyter_backend = 'ipygany'
    my_theme.title = title
    my_theme.axes.show = False
    # apply it globally
    pv.global_theme.load_theme(my_theme)
    if is_jupyter():
        MultiPlotterType = _JupyterMultiPlotter
    else:
        MultiPlotterType = MultiPlotter
    if isinstance(meshes, tuple):
        rows = ceil(len(meshes)/2)
        plotter = MultiPlotterType(nrows=rows, ncols=columns, window_size=render_size)
        for meshlist, ind in zip(meshes, np.unravel_index(list(range(len(meshes))), shape=(rows, columns))):
            plotter[ind].add_mesh(from_meshio(meshlist[0]), show_edges=True)
    else:
        plotter = pv.Plotter()
        plotter.add_mesh(from_meshio(mesh=meshes[0]))

    plotter.camera_position = 'xy'
    return plotter.show()


class PyVistaPatchWidget(QtInteractor):

    def __init__(self, parent, grid, vmin=None, vmax=None, bounding_box=([0, 0], [1, 1]), codim=2):
        from qtpy.QtWidgets import QSizePolicy
        assert grid.reference_element in (triangle, square)
        assert grid.dim == 2
        assert codim in (0, 2)
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.scalar_name = 'Data'
        self.mesh = to_meshio(grid, data=None, scalar_name=self.scalar_name, codim=codim)[0]
        self.add_mesh(self.mesh)
        self.view_xy()
        self.grid = grid
        self.reference_element = grid.reference_element
        self.vmin = vmin
        self.vmax = vmax
        self.codim = codim

    def set(self, U, vmin=None, vmax=None):
        self.vmin = self.vmin if vmin is None else vmin
        self.vmax = self.vmax if vmax is None else vmax

        mesh = to_meshio(self.grid, U)[0]
        if self.codim == 2:
            arrays = self.mesh.point_arrays
            arrays[self.scalar_name] = mesh.point_data[self.scalar_name]
        else:
            arrays = self.mesh.cell_arrays
            arrays[self.scalar_name] = mesh.cell_data[self.scalar_name]

        _normalize(arrays[self.scalar_name], vmin, vmax)
        self.render()
