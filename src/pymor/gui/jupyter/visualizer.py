# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import time
import warnings
import IPython
import numpy as np
from ipywidgets import IntSlider, interact, widgets, Play
from k3d.objects import Mesh
from k3d.plot import Plot as k3dPlot
from k3d.transform import process_transform_arguments
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap

from pymor.grids.constructions import flatten_grid
from pymor.core.config import config
from pymor.gui.matplotlib import MatplotlibPatchAxes
from pymor.vectorarrays.interfaces import VectorArrayInterface


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
            self.vertices[:,:-1] = self.coordinates

        self.idx = 0
        self.mesh = process_transform_arguments(Mesh(vertices=np.array(self.vertices, np.float32),
             indices=np.array(self.subentities, np.uint32),
             color=0x0000FF,
             opacity=1.0,
             attribute=self.data[self.idx],
             color_range=(np.nanmin(self.data), np.nanmax(self.data)),
             color_map=np.array(color_map, np.float32),
             wireframe=False,
             compression_level=0))
        self += self.mesh
        self.camera_no_pan = True
        self.camera_no_rotate = True
        self.camera_no_zoom = True

    def _goto_idx(self, idx):
        if idx > len(self.data) or idx < 0:
            warnings.warn(f'Index {idx} outside data range for VectorArrayPlot', RuntimeWarning)
            return
        self.idx = idx
        self.mesh.attribute = self.data[self.idx]

    def dec(self):
        self._goto_idx(self.idx-1)

    def inc(self):
        self._goto_idx(self.idx+1)


def visualize_k3d(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, columns=2,
         color_map=get_cmap('viridis')):
    """Generate a k3d Plot and associated controls for  scalar data associated to a two-dimensional |Grid|.

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

    plot = VectorArrayPlot(U=U, grid=grid, codim=codim, color_attribute_name='None', grid_auto_fit=False,
                              camera_auto_fit=False, color_map=color_map)
    # display needs to have been called before changing camera/grid_visible
    plot.display()
    # could be replaced with testing if the widget is'ready'
    time.sleep(0.5)
    plot.grid_visible = False
    plot.menu_visibility = False
    # guesstimate
    fov_angle = 30
    absx = np.abs(combined_bounds[0] - combined_bounds[3])
    c_dist = np.sin((90 - fov_angle) * np.pi / 180) * absx / (2 * np.sin(fov_angle * np.pi / 180))
    xhalf = (combined_bounds[0] + combined_bounds[3]) / 2
    yhalf = (combined_bounds[1] + combined_bounds[4]) / 2
    zhalf = (combined_bounds[2] + combined_bounds[5]) / 2
    # camera[posx, posy, posz, targetx, targety, targetz, upx, upy, upz]
    plot.camera = (xhalf, yhalf, zhalf + c_dist,
                      xhalf, yhalf, zhalf,
                      0, 1, 0)
    size = len(U)
    if size > 1:
        play = Play(min=0, max=size - 1, step=1, value=0, description='Timestep:')
        interact(idx=play).widget(plot._goto_idx)
        slider = IntSlider(min=0, max=size-1, step=1, value=0, description='Timestep:')
        interact(idx=slider).widget(plot._goto_idx)
        widgets.jslink((play, 'value'), (slider, 'value'))
        hbox = widgets.HBox([play, slider])
        IPython.display.display(hbox)

    return plot


def visualize_patch(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, columns=2):
    """Visualize scalar data associated to a two-dimensional |Grid| as a patch plot.

    The grid's |ReferenceElement| must be the triangle or square. The data can either
    be attached to the faces or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    U
        |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
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
    """

    assert isinstance(U, VectorArrayInterface) \
        or (isinstance(U, tuple)
            and all(isinstance(u, VectorArrayInterface) for u in U)
            and all(len(u) == len(U[0]) for u in U))
    U = (U.to_numpy().astype(np.float64, copy=False),) if isinstance(U, VectorArrayInterface) else \
        tuple(u.to_numpy().astype(np.float64, copy=False) for u in U)

    if not config.HAVE_MATPLOTLIB:
        raise ImportError('cannot visualize: import of matplotlib failed')
    if not config.HAVE_IPYWIDGETS and len(U[0]) > 1:
        raise ImportError('cannot visualize: import of ipywidgets failed')

    if isinstance(legend, str):
        legend = (legend,)
    assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)
    if len(U) < 2:
        columns = 1

    class Plot:

        def __init__(self):
            if separate_colorbars:
                if rescale_colorbars:
                    self.vmins = tuple(np.min(u[0]) for u in U)
                    self.vmaxs = tuple(np.max(u[0]) for u in U)
                else:
                    self.vmins = tuple(np.min(u) for u in U)
                    self.vmaxs = tuple(np.max(u) for u in U)
            else:
                if rescale_colorbars:
                    self.vmins = (min(np.min(u[0]) for u in U),) * len(U)
                    self.vmaxs = (max(np.max(u[0]) for u in U),) * len(U)
                else:
                    self.vmins = (min(np.min(u) for u in U),) * len(U)
                    self.vmaxs = (max(np.max(u) for u in U),) * len(U)

            import matplotlib.pyplot as plt

            rows = int(np.ceil(len(U) / columns))
            self.figure = figure = plt.figure()

            self.plots = plots = []
            axes = []
            for i, (vmin, vmax) in enumerate(zip(self.vmins, self.vmaxs)):
                ax = figure.add_subplot(rows, columns, i+1)
                axes.append(ax)
                plots.append(MatplotlibPatchAxes(figure, grid, bounding_box=bounding_box, vmin=vmin, vmax=vmax,
                                                 codim=codim, colorbar=separate_colorbars))
                if legend:
                    ax.set_title(legend[i])

            plt.tight_layout()
            if not separate_colorbars:
                figure.colorbar(plots[0].p, ax=axes)

        def set(self, U, ind):
            if rescale_colorbars:
                if separate_colorbars:
                    self.vmins = tuple(np.min(u[ind]) for u in U)
                    self.vmaxs = tuple(np.max(u[ind]) for u in U)
                else:
                    self.vmins = (min(np.min(u[ind]) for u in U),) * len(U)
                    self.vmaxs = (max(np.max(u[ind]) for u in U),) * len(U)

            for u, plot, vmin, vmax in zip(U, self.plots, self.vmins, self.vmaxs):
                plot.set(u[ind], vmin=vmin, vmax=vmax)

    plot = Plot()
    plot.set(U, 0)

    if len(U[0]) > 1:

        from ipywidgets import interact, IntSlider

        def set_time(t):
            plot.set(U, t)

        interact(set_time, t=IntSlider(min=0, max=len(U[0])-1, step=1, value=0))

    return plot
