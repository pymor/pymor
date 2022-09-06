# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import warnings

import IPython
import numpy as np

from pymor.core.config import config

config.require('K3D')
config.require('MATPLOTLIB')

import k3d
from ipywidgets import IntSlider, Play, interact, widgets
from k3d.plot import Plot as k3dPlot
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap

from pymor.discretizers.builtin.grids.constructions import flatten_grid


class VectorArrayPlot(k3dPlot):
    def __init__(self, U, grid, codim, color_map, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'transform' in kwargs.keys():
            raise RuntimeError('supplying transforms is currently not supported for time series Data')

        self.subentities, self.coordinates, entity_map = flatten_grid(grid)
        self.data = (U.to_numpy() if codim == 0 else U.to_numpy()[:, entity_map].copy()).astype(np.float32)

        if grid.dim == 2:
            # pad 0 in z dimension
            self.vertices = np.zeros((len(self.coordinates), 3))
            self.vertices[:, :-1] = self.coordinates

        self.idx = 0
        self.mesh = k3d.mesh(vertices=np.array(self.vertices, np.float32),
                             indices=np.array(self.subentities, np.uint32),
                             color=0x0000FF,
                             opacity=1.0,
                             attribute=self.data[self.idx],
                             color_range=(np.nanmin(self.data), np.nanmax(self.data)),
                             color_map=np.array(color_map, np.float32),
                             wireframe=False,
                             compression_level=0)

        self += self.mesh
        self.lock = True
        self.camera_no_pan = self.lock
        self.camera_no_rotate = self.lock
        self.camera_no_zoom = self.lock

    def _goto_idx(self, idx):
        if idx > len(self.data) or idx < 0:
            warnings.warn(f'Index {idx} outside data range for VectorArrayPlot', RuntimeWarning)
            return
        self.idx = idx
        self.mesh.attribute = self.data[self.idx]

    def dec(self):
        self._goto_idx(self.idx - 1)

    def inc(self):
        self._goto_idx(self.idx + 1)


def visualize_k3d(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                  separate_colorbars=False, rescale_colorbars=False, columns=2,
                  color_map=get_cmap('viridis')):
    """Generate a k3d Plot for scalar data associated to a two-dimensional |Grid|.

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
        a Matplotlib Colormap object or a K3D array((step, r, g, b))
    """
    assert len(bounding_box) == 2
    assert all(1 < len(bounding_box[i]) < 4 for i in range(2))
    assert len(bounding_box[0]) == len(bounding_box[1])
    if isinstance(color_map, Colormap):
        color_map = [(x, *color_map(x)[:3]) for x in np.linspace(0, 1, 256)]

    if len(bounding_box[0]) == 2:
        lower = np.array([bounding_box[0][0], bounding_box[0][1], 0])
        upper = np.array([bounding_box[1][0], bounding_box[1][1], 0])
        bounding_box = (lower, upper)
    combined_bounds = np.hstack(bounding_box)

    plot = VectorArrayPlot(U=U, grid=grid, codim=codim,
                           color_attribute_name='None',
                           grid_auto_fit=False,
                           camera_auto_fit=False, color_map=color_map)
    size = len(U)
    plot.grid_visible = True
    plot.menu_visibility = size > 1
    plot.camera = plot.get_auto_camera(yaw=0, pitch=0, bounds=combined_bounds, factor=0.5)

    if size > 1:
        play = Play(min=0, max=size - 1, step=1, value=0, description='Timestep:')
        interact(idx=play).widget(plot._goto_idx)
        slider = IntSlider(min=0, max=size - 1, step=1, value=0, description='Timestep:')
        interact(idx=slider).widget(plot._goto_idx)
        widgets.jslink((play, 'value'), (slider, 'value'))
        hbox = widgets.HBox([play, slider])
        IPython.display.display(hbox)

    return plot
