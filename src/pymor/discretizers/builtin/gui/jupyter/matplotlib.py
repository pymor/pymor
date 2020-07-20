# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
import itertools

import numpy as np
from IPython.core.display import display
from ipywidgets import HTML, HBox, widgets, Layout
import matplotlib.pyplot as plt

from pymor.core.config import config
from pymor.discretizers.builtin.gui.matplotlib import MatplotlibPatchAxes, Matplotlib1DAxes
from pymor.vectorarrays.interface import VectorArray


class MPLPlotBase:

    def __init__(self, U, grid, codim, legend, bounding_box=None, separate_colorbars=False, count=None,
                 separate_plots=False):
        assert isinstance(U, VectorArray) \
               or (isinstance(U, tuple)
                   and all(isinstance(u, VectorArray) for u in U)
                   and all(len(u) == len(U[0]) for u in U))
        self.fig_ids = (U.uid,) if isinstance(U, VectorArray) else tuple(u.uid for u in U)
        U = (U.to_numpy().astype(np.float64, copy=False),) if isinstance(U, VectorArray) else \
            tuple(u.to_numpy().astype(np.float64, copy=False) for u in U)

        if not config.HAVE_MATPLOTLIB:
            raise ImportError('cannot visualize: import of matplotlib failed')
        if not config.HAVE_IPYWIDGETS and len(U[0]) > 1:
            raise ImportError('cannot visualize: import of ipywidgets failed')
        self.legend = (legend,) if isinstance(legend, str) else legend
        assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)
        self._set_limits(U)

        self.plots = []
        # this _supposed_ to let animations run in sync
        sync_timer = None

        do_animation = len(U[0]) > 1

        for i, (vmin, vmax, u) in enumerate(zip(self.vmins, self.vmaxs, U)):
            figure = plt.figure(self.fig_ids[i])
            ax = plt.axes()
            sync_timer = sync_timer or figure.canvas.new_timer()
            if grid.dim == 2:
                self.plots.append(MatplotlibPatchAxes(U=u, figure=figure, sync_timer=sync_timer, grid=grid,
                                                 bounding_box=bounding_box, vmin=vmin, vmax=vmax,
                                                 codim=codim, colorbar=separate_colorbars or i == len(U) - 1))
            else:
                assert count
                self.plots.append(Matplotlib1DAxes(U=u, figure=figure, sync_timer=sync_timer, grid=grid,
                                                   vmin=vmin, vmax=vmax, count=count, codim=codim,
                                                   separate_plots=separate_plots))
            if self.legend:
                ax.set_title(self.legend[i])

            plt.tight_layout()

        if do_animation:
            for fig_id in self.fig_ids:
                # avoids figure double display
                plt.close(fig_id)
            html = [p.html for p in self.plots]
            template = """<div style="float: left; padding: 10px;">{0}</div>"""
            # IPython display system checks for presence and calls this func
            self._repr_html_ = lambda : '\n'.join(template.format(a._repr_html_()) for a in html)
        else:
            self._out = widgets.Output()
            with self._out:
                plt.show()
            # IPython display system checks for presence and calls this func
            self._ipython_display_ = self._out._ipython_display_




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

    class Plot(MPLPlotBase):

        def _set_limits(self, np_U):
            if separate_colorbars:
                # todo rescaling not set up
                if rescale_colorbars:
                    self.vmins = tuple(np.min(u[0]) for u in np_U)
                    self.vmaxs = tuple(np.max(u[0]) for u in np_U)
                else:
                    self.vmins = tuple(np.min(u) for u in np_U)
                    self.vmaxs = tuple(np.max(u) for u in np_U)
            else:
                if rescale_colorbars:
                    self.vmins = (min(np.min(u[0]) for u in np_U),) * len(np_U)
                    self.vmaxs = (max(np.max(u[0]) for u in np_U),) * len(np_U)
                else:
                    self.vmins = (min(np.min(u) for u in np_U),) * len(np_U)
                    self.vmaxs = (max(np.max(u) for u in np_U),) * len(np_U)

        def __init__(self):
            super(Plot, self).__init__(U, grid, codim, legend, bounding_box=bounding_box,
                                       separate_colorbars=separate_colorbars)

        def set(self, ind):
            np_U = self.U
            if self.rescale_colorbars:
                if self.separate_colorbars:
                    self.vmins = tuple(np.min(u[ind]) for u in np_U)
                    self.vmaxs = tuple(np.max(u[ind]) for u in np_U)
                else:
                    self.vmins = (min(np.min(u[ind]) for u in np_U),) * len(np_U)
                    self.vmaxs = (max(np.max(u[ind]) for u in np_U),) * len(np_U)

            for u, plot, vmin, vmax in zip(np_U, self.plots, self.vmins, self.vmaxs):
                plot.set(u[ind], vmin=vmin, vmax=vmax)

    return Plot()



def visualize_matplotlib_1d(grid, U, codim=1, title=None, legend=None, separate_plots=False, separate_axes=False, columns=2):
    """Visualize scalar data associated to a one-dimensional |Grid| as a plot.

    The grid's |ReferenceElement| must be the line. The data can either
    be attached to the subintervals or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    U
        |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
        as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
        provided, in which case several plots are made into the same axes. The
        lengths of all arrays have to agree.
    codim
        The codimension of the entities the data in `U` is attached to (either 0 or 1).
    title
        Title of the plot.
    legend
        Description of the data that is plotted. Most useful if `U` is a tuple in which
        case `legend` has to be a tuple of strings of the same length.
    separate_plots
        If `True`, use subplots to visualize multiple |VectorArrays|.
    separate_axes
        If `True`, use separate axes for each subplot.
    column
        Number of columns the subplots are organized in.
    """

    class Plot(MPLPlotBase):

        def _set_limits(self, np_U):
            if separate_plots:
                if separate_axes:
                    self.vmins = tuple(np.min(u) for u in np_U[0])
                    self.vmaxs = tuple(np.max(u) for u in np_U[0])
                else:
                    self.vmins = (min(np.min(u) for u in np_U),) * len(np_U[0])
                    self.vmaxs = (max(np.max(u) for u in np_U),) * len(np_U[0])
            else:
                self.vmins = (min(np.min(u) for u in np_U[0]),)
                self.vmaxs = (max(np.max(u) for u in np_U[0]),)

        def __init__(self):
            count = 1
            super(Plot, self).__init__(U, grid, codim, legend, separate_plots=separate_plots, count=count)

        def set(self, ind):
            np_U = self.U[ind]
            if separate_plots:
                for u, plot, vmin, vmax in zip(np_U, self.plots, self.vmins, self.vmaxs):
                    plot.set(u[np.newaxis, ...], vmin=vmin, vmax=vmax)
            else:
                self.plots[0].set(np_U, vmin=self.vmins[0], vmax=self.vmaxs[0])

    return Plot()
