# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('IPYWIDGETS')
config.require('K3D')
config.require('MATPLOTLIB')

import k3d
import matplotlib as mpl
import numpy as np
from ipywidgets import GridspecLayout, Label, Layout, VBox, jslink
from k3d.plot import Plot as K3DPlot
from matplotlib.colors import Colormap

from pymor.core.defaults import defaults
from pymor.discretizers.builtin.grids.constructions import flatten_grid
from pymor.discretizers.builtin.grids.referenceelements import triangle
from pymor.discretizers.builtin.gui.jupyter.animation_widget import AnimationWidget
from pymor.vectorarrays.interface import VectorArray


class VectorArrayPlot(K3DPlot):
    def __init__(self, grid, U, codim, color_range, color_map, warp, bounding_box, show_mesh, *args, **kwargs):
        bounding_box = grid.bounding_box() if bounding_box is None else np.asarray(bounding_box)
        assert bounding_box.shape == (2, 2)

        if isinstance(color_map, Colormap):
            color_map = [(x, *color_map(x)[:3]) for x in np.linspace(0, 1, 256)]

        super().__init__(*args, **kwargs)
        if 'transform' in kwargs.keys():
            raise RuntimeError('supplying transforms is currently not supported for time series Data')

        self.codim, self.warp, self.reference_element = codim, warp, grid.reference_element

        subentities, coordinates, entity_map = flatten_grid(grid)
        self.entity_map = entity_map

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
        indices = np.array(indices, dtype=np.uint32)

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
                             indices=indices,
                             color=0x0000FF,
                             opacity=1.0,
                             color_range=color_range,
                             color_map=np.array(color_map, np.float32),
                             wireframe=False,
                             compression_level=0,
                             side='double',
                             **{('attribute' if codim == 2 else 'triangles_attribute'): self.data})
        self += self.mesh

        if warp and show_mesh:
            line_vertices = np.hstack([coordinates.astype(np.float32),
                                       np.zeros((len(coordinates), 1), dtype=np.float32)])
            line_indices = np.array(subentities, dtype=np.float32)  # indices trait is float32
            line_indices = np.repeat(line_indices, 2, axis=1)[:,1:]
            line_indices = np.hstack([line_indices, line_indices[:,0:1]]).ravel()
            self.lines = k3d.lines(vertices=line_vertices,
                                   indices=line_indices,
                                   indices_type='segment',
                                   shader='simple',
                                   color=0x444444,
                                   opacity=0.1,)
            self += self.lines

        self.time = 0

        if not warp:
            self.camera_no_pan = True
            self.camera_no_rotate = True
            self.camera_no_zoom = True

        self.grid_visible = False
        self.menu_visibility = False
        self.axes_helper = 0
        center = np.hstack([(bounding_box[1] + bounding_box[0]) / 2, 0.])
        radius = (np.max(bounding_box[1] - bounding_box[0]) / 2) * 1.1

        FOV = 46
        camera_pos = center + np.array([0., 0., radius/np.tan(np.deg2rad(FOV/2))])
        up = np.array([0, 1, 0])
        self.camera = np.hstack([camera_pos, center, up])
        self.camera_fov = FOV

    def set(self, U, vmin, vmax, warp=None):
        if warp is not None:
            self.warp = warp
        for idx, u in enumerate(U):
            u = u.astype(np.float32)
            if self.codim == 2:
                data = u[self.entity_map]
            elif self.reference_element == triangle:
                data = u
            else:
                data = np.tile(u, 2)
            if self.warp:
                if self.codim == 2:
                    self.vertices[str(idx)][:,-1] = u[self.entity_map] * self.warp
                elif self.reference_element == triangle:
                    self.vertices[str(idx)][:,-1] = np.repeat(u, 3) * self.warp
                else:
                    self.vertices[str(idx)][:,-1] = np.tile(np.repeat(u, 3), 2) * self.warp
            self.data[str(idx)] = data

        # setting color_range as a dict does not work.
        # setting color_range as a single pair of values also does not work as long
        # as vertices and attributes are dicts.
        # as a workaround, we set these first to single arrays ...
        self.mesh.vertices = self.vertices[str(int(self.time))]
        if self.codim == 2:
            self.mesh.attribute = self.data[str(int(self.time))]
        else:
            self.mesh.triangles_attribute = self.data[str(int(self.time))]
        self.mesh.color_range = [vmin, vmax]

        if self.warp:
            self.mesh.vertices = {k: v.copy() for k, v in self.vertices.items()}
        if self.codim == 2:
            self.mesh.attribute = self.data
        else:
            self.mesh.triangles_attribute = self.data


@defaults('warp_by_scalar', 'scale_factor', 'background_color')
def visualize_k3d(grid, U, bounding_box=None, codim=2, title=None, legend=None,
                  separate_colorbars=False, rescale_colorbars=False, columns=2,
                  warp_by_scalar=True, scale_factor='auto', show_mesh=True, height=300,
                  color_map=mpl.colormaps['viridis'], background_color=0xffffff, return_widget=True):
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
    warp_by_scalar
        If `True`, visualize as a surface where the z-coordinate is determined by `U`.
    scale_factor
        If `warp_by_scalar` is enabled, the factor by which the values in `U` are scaled
        to obtain the z-coordinate. If set to `'auto'`, the scale factor is determined
        from the minimum/maximum values in `U` and the `bounding_box`.
    show_mesh
        If `True` and `warp_by_scalar` is enabled, additionally visualize the mesh in the
        (z=0)-plane.
    height
        Height of the subplots in pixels.
    color_map
        Matplotlib `Colormap` object or a K3D array((step, r, g, b)).
    background_color
        Background color of the plot.
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

    from pymor.discretizers.builtin.gui.visualizers import _vmins_vmaxs
    vmins, vmaxs = _vmins_vmaxs(U, separate_colorbars, rescale_colorbars)

    if warp_by_scalar:
        if scale_factor == 'auto':
            bb = grid.bounding_box()
            bb_fac = np.max(bb[1] - bb[0]) / 3
            scale_factors = [bb_fac/(max(abs(vmin[0]), abs(vmax[0])) + 1e-15)  # prevent division by zero
                             for vmin, vmax in zip(vmins, vmaxs)]
        else:
            scale_factors = [scale_factor] * len(U)
    else:
        scale_factors = [0] * len(U)

    plots = [VectorArrayPlot(grid, u,
                             codim=codim,
                             grid_auto_fit=False,
                             camera_auto_fit=False,
                             color_range=[vmin[0], vmax[0]],  # rescale_colorbars not supported yet
                             color_map=color_map,
                             colormap_scientific=True,
                             warp=sf,
                             show_mesh=show_mesh,
                             bounding_box=bounding_box,
                             height=height,
                             background_color=background_color)
             for u, vmin, vmax, sf in zip(U, vmins, vmaxs, scale_factors)]

    for p in plots[1:]:
        jslink((plots[0], 'camera'), (p, 'camera'))

    rows = int(np.ceil(len(plots) / columns))
    if legend is None:
        plot_widget = GridspecLayout(rows, columns if len(U) > 1 else 1, width='100%')
        for (i, j), p in zip(np.ndindex(rows, columns), plots):
            plot_widget[i, j] = p
    else:
        plot_widget = GridspecLayout(rows*2, columns if len(U) > 1 else 1, width='100%')
        for (i, j), p, l in zip(np.ndindex(rows, columns), plots, legend):
            p.layout.width = '100%'
            plot_widget[2*i,   j] = Label(l, layout=Layout(display='flex', justify_content='center',
                                                           justify_self='center'))
            plot_widget[2*i+1, j] = p
        plot_widget.layout.grid_template_rows=' '.join(['auto'] * rows * 2)

    main_widget = []
    if title is not None:
        main_widget.append(Label(title))

    main_widget.append(plot_widget)

    if len(U[0]) > 1:
        animation_widget = AnimationWidget(len(U[0]))

        for p in plots:
            jslink((p, 'time'), (animation_widget.frame_slider, 'value'))
        main_widget.append(animation_widget)

    main_widget = VBox(main_widget, layout=Layout(align_items='center'))

    def set(U):
        assert isinstance(U, VectorArray) \
               or (isinstance(U, tuple)
                   and all(isinstance(u, VectorArray) for u in U)
                   and all(len(u) == len(U[0]) for u in U))
        U = (U.to_numpy().astype(np.float64, copy=False),) if isinstance(U, VectorArray) else \
            tuple(u.to_numpy().astype(np.float64, copy=False) for u in U)
        vmins, vmaxs = _vmins_vmaxs(U, separate_colorbars, rescale_colorbars)

        for u, p, vmin, vmax in zip(U, plots, vmins, vmaxs):
            p.set(u, vmin[0], vmax[0])

    main_widget.set = set

    return main_widget
