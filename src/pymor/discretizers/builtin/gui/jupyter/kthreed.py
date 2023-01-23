# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)


import numpy as np
from ipywidgets import GridBox, jslink

from pymor.core.config import config
from pymor.discretizers.builtin.grids.referenceelements import triangle

config.require('K3D')
config.require('MATPLOTLIB')

import k3d
from k3d.plot import Plot as k3dPlot
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap

from pymor.discretizers.builtin.grids.constructions import flatten_grid


class VectorArrayPlot(k3dPlot):
    def __init__(self, U, grid, codim, color_map, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'transform' in kwargs.keys():
            raise RuntimeError('supplying transforms is currently not supported for time series Data')

        subentities, coordinates, entity_map = flatten_grid(grid)

        from pymor.discretizers.builtin.gui.jupyter import _transform_vertex_index_data
        self.indices, vertices = _transform_vertex_index_data(codim, coordinates, grid, subentities)
        vertices = np.array(vertices, dtype=np.float32)

        size = len(U)
        if size > 0:
            self.data = {}
            self.vertices = {}
            self.time = 0
            # TODO fix hardcoded color range
            # color_range = {}
            for idx, u in enumerate(U.to_numpy()):
                u = u.astype(np.float32)
                if codim == 2:
                    data =u[entity_map]
                elif grid.reference_element == triangle:
                    data = np.repeat(u, 3)
                else:
                    data = np.tile(np.repeat(u, 3), 2)
                self.data[str(idx)] = data
                self.vertices[str(idx)] = vertices


        else:
            u = U.to_numpy()[0].astype(np.float32)
            self.vertices = vertices

            if codim == 2:
                self.data = u[entity_map]
            elif grid.reference_element == triangle:
                self.data = np.repeat(u, 3).astype(np.float32)
            else:
                self.data = np.tile(np.repeat(u, 3), 2).astype(np.float32)

        self.idx = 0
        self.mesh = k3d.mesh(vertices=self.vertices,
                             indices=np.array(self.indices, np.uint32),
                             color=0x0000FF,
                             opacity=1.0,
                             attribute=self.data,
                             color_range=(0, 1),
                             color_map=np.array(color_map, np.float32),
                             wireframe=False,
                             compression_level=0)

        self += self.mesh
        self.time = 0
        self.lock = False
        self.camera_no_pan = self.lock
        self.camera_no_rotate = self.lock
        self.camera_no_zoom = self.lock

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
    if isinstance(U, (tuple, list)):
        plots = [visualize_k3d(grid, u, bounding_box, codim, color_map=color_map, title=None,
                               legend=legend, separate_colorbars=separate_colorbars,
                               rescale_colorbars=rescale_colorbars, columns=None) for u in U]
        first_plot = plots[0]
        for p in plots[1:]:
            jslink((first_plot, 'time'), (p, 'time'))
        return GridBox(plots, columns=columns)
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


    return plot
