# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('IPYWIDGETS')
config.require('K3D')
config.require('MATPLOTLIB')

import k3d
import numpy as np
from ipywidgets import GridspecLayout, IntSlider, Label, Layout, VBox, jslink
from k3d.plot import Plot as K3DPlot
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap

from pymor.core.defaults import defaults
from pymor.discretizers.builtin.grids.constructions import flatten_grid
from pymor.discretizers.builtin.grids.referenceelements import triangle
from pymor.vectorarrays.interface import VectorArray


class VectorArrayPlot(K3DPlot):
    def __init__(self, grid, U, codim, color_range, color_map, warp, bounding_box, *args, **kwargs):
        bounding_box = grid.bounding_box() if bounding_box is None else np.asarray(bounding_box)
        assert bounding_box.shape == (2, 2)

        if isinstance(color_map, Colormap):
            color_map = [(x, *color_map(x)[:3]) for x in np.linspace(0, 1, 256)]

        super().__init__(*args, **kwargs)
        if 'transform' in kwargs.keys():
            raise RuntimeError('supplying transforms is currently not supported for time series Data')

        subentities, coordinates, entity_map = flatten_grid(grid)

        if grid.reference_element == triangle:
            if codim == 2:
                vertices = np.zeros((len(coordinates), 3))
                vertices[:, :-1] = coordinates
                indices = subentities
            else:
                vertices = np.zeros((len(subentities) * 3, 3))
                VERTEX_POS = coordinates[subentities]
                vertices[:, 0:2] = VERTEX_POS.reshape((-1, 2))
                indices = np.arange(len(subentities) * 3, dtype=np.uint32)
        else:
            if codim == 2:
                vertices = np.zeros((len(coordinates), 3))
                vertices[:, :-1] = coordinates
                indices = np.vstack((subentities[:, 0:3], subentities[:, [0, 2, 3]]))
            else:
                num_entities = len(subentities)
                vertices = np.zeros((num_entities * 6, 3))
                VERTEX_POS = coordinates[subentities]
                vertices[0:num_entities * 3, 0:2] = VERTEX_POS[:, 0:3, :].reshape((-1, 2))
                vertices[num_entities * 3:, 0:2] = VERTEX_POS[:, [0, 2, 3], :].reshape((-1, 2))
                indices = np.arange(len(subentities) * 6, dtype=np.uint32)


        vertices = np.array(vertices, dtype=np.float32)

        self.data = {}
        self.vertices = {} if warp else vertices
        for idx, u in enumerate(U):
            u = u.astype(np.float32)
            if codim == 2:
                data = u[entity_map]
            elif grid.reference_element == triangle:
                data = u
            else:
                data = np.tile(u, 2)
            if warp:
                self.vertices[str(idx)] = vertices.copy()
                if codim == 2:
                    self.vertices[str(idx)][:,-1] = u[entity_map] * warp
                elif grid.reference_element == triangle:
                    self.vertices[str(idx)][:,-1] = np.repeat(u, 3) * warp
                else:
                    self.vertices[str(idx)][:,-1] = np.tile(np.repeat(u, 3), 2) * warp
            self.data[str(idx)] = data

        self.idx = 0
        self.mesh = k3d.mesh(vertices=self.vertices,
                             indices=np.array(indices, np.uint32),
                             color=0x0000FF,
                             opacity=1.0,
                             color_range=color_range,
                             color_map=np.array(color_map, np.float32),
                             wireframe=False,
                             compression_level=0,
                             **{('attribute' if codim == 2 else 'triangles_attribute'): self.data})

        self += self.mesh
        self.time = 0

        if not warp:
            self.camera_no_pan = True
            self.camera_no_rotate = True
            self.camera_no_zoom = True

        self.grid_visible = True
        self.menu_visibility = False
        self.axes_helper = 0
        center = np.hstack([(bounding_box[1] + bounding_box[0]) / 2, 0.])
        camera_pos = center + np.array([0., 0., 1.])
        up = np.array([0, 1, 0])
        self.camera = np.hstack([camera_pos, center, up])

        radius = np.max(bounding_box[1] - bounding_box[0]) / 2
        self.camera_fov = np.rad2deg(np.arcsin(radius)) * 2


@defaults('warp_by_scalar', 'scale_factor', 'background_color')
def visualize_k3d(grid, U, bounding_box=None, codim=2, title=None, legend=None,
                  separate_colorbars=False, rescale_colorbars=False, columns=2,
                  warp_by_scalar=True, scale_factor='auto', height=300,
                  color_map=get_cmap('viridis'), background_color=0xffffff):
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
    if rescale_colorbars:
        raise NotImplementedError

    assert isinstance(U, VectorArray) \
        or (isinstance(U, tuple) and all(isinstance(u, VectorArray) for u in U)
            and all(len(u) == len(U[0]) for u in U))
    if isinstance(U, VectorArray):
        U = (U,)
    U = tuple(u.to_numpy() for u in U)

    legend = (legend,) if isinstance(legend, str) else legend
    assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)
    if legend is None:
        legend = (None,) * len(U)

    if separate_colorbars:
        color_ranges = [[np.min(u), np.max(u)] for u in U]
    else:
        color_ranges = [[min(np.min(u) for u in U), max(np.max(u) for u in U)]] * len(U)

    if warp_by_scalar:
        if scale_factor == 'auto':
            scale_factors = np.max(np.abs(np.array(color_ranges)), axis=1) + 1e-15  # prevent division by zero
            if not separate_colorbars:
                scale_factors = np.full(len(U), np.max(scale_factors))
            bb = grid.bounding_box()
            scale_factors = np.max(bb[1] - bb[0]) / scale_factors
        else:
            scale_factors = [scale_factor] * len(U)
    else:
        scale_factors = [0] * len(U)

    plots = [VectorArrayPlot(grid, u,
                             codim=codim,
                             grid_auto_fit=False,
                             camera_auto_fit=False,
                             color_range=cr,
                             color_map=color_map,
                             warp=sf,
                             bounding_box=bounding_box,
                             height=height,
                             background_color=background_color)
             for u, cr, sf in zip(U, color_ranges, scale_factors)]

    for p in plots[1:]:
        jslink((plots[0], 'camera'), (p, 'camera'))

    if len(plots) > 1:
        rows = int(np.ceil(len(plots) / columns))
        plot_widget = GridspecLayout(rows, columns, width='100%')
        for (i, j), p, l in zip(np.ndindex(rows, columns), plots, legend):
            if l is None:
                w = p
            else:
                p.layout.width = '100%'
                w = VBox([Label(l), p], layout=Layout(align_items='center'))
            plot_widget[i, j] = w
    else:
        plot_widget = plots[0]
        plot_widget.layout.width = '100%'

    main_widget = []
    if title is not None:
        main_widget.append(Label(title))

    main_widget.append(plot_widget)

    if len(U[0]) > 1:
        slider = IntSlider(0, 0, len(U[0])-1)
        for p in plots:
            jslink((p, 'time'), (slider, 'value'))
        main_widget.append(slider)

    main_widget = VBox(main_widget, layout=Layout(align_items='center'))

    return main_widget
