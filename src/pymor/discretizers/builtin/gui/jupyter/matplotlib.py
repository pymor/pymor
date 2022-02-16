# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
import numpy as np
from ipywidgets import widgets
import matplotlib.pyplot as plt

from pymor.core.config import config
from pymor.discretizers.builtin.gui.matplotlib_base import MatplotlibPatchAxes, Matplotlib1DAxes
from pymor.vectorarrays.interface import VectorArray


class MPLPlotBase:

    def __init__(self, U, grid, codim, legend, bounding_box=None, separate_colorbars=False, columns=2,
                 separate_plots=False, separate_axes=False):
        assert isinstance(U, VectorArray) \
               or (isinstance(U, tuple)
                   and all(isinstance(u, VectorArray) for u in U)
                   and all(len(u) == len(U[0]) for u in U))
        if separate_plots:
            self.fig_ids = (U.uid,) if isinstance(U, VectorArray) else tuple(u.uid for u in U)
        else:
            # using the same id multiple times lets us automagically re-use the same figure
            self.fig_ids = (U.uid,) if isinstance(U, VectorArray) else [U[0].uid] * len(U)
        self.U = U = (U.to_numpy().astype(np.float64, copy=False),) if isinstance(U, VectorArray) else \
            tuple(u.to_numpy().astype(np.float64, copy=False) for u in U)
        if grid.dim == 1 and len(U[0]) > 1 and not separate_plots:
            raise NotImplementedError('Plotting of VectorArrays with length > 1 is only available with '
                                      '`separate_plots=True`')

        if not config.HAVE_MATPLOTLIB:
            raise ImportError('cannot visualize: import of matplotlib failed')
        if not config.HAVE_IPYWIDGETS and len(U[0]) > 1:
            raise ImportError('cannot visualize: import of ipywidgets failed')
        self.legend = (legend,) if isinstance(legend, str) else legend
        assert self.legend is None or isinstance(self.legend, tuple) and len(self.legend) == len(U)
        self._set_limits(U)

        self.plots = []
        # this _supposed_ to let animations run in sync
        sync_timer = None

        do_animation = not separate_axes and len(U[0]) > 1

        if separate_plots:
            for i, (vmin, vmax, u) in enumerate(zip(self.vmins, self.vmaxs, U)):
                figure = plt.figure(self.fig_ids[i])
                sync_timer = sync_timer or figure.canvas.new_timer()
                if grid.dim == 2:
                    plot = MatplotlibPatchAxes(U=u, figure=figure, sync_timer=sync_timer, grid=grid, vmin=vmin,
                                               vmax=vmax, bounding_box=bounding_box, codim=codim, columns=columns,
                                               colorbar=separate_colorbars or i == len(U) - 1)
                else:
                    plot = Matplotlib1DAxes(U=u, figure=figure, sync_timer=sync_timer, grid=grid, vmin=vmin, vmax=vmax,
                                            columns=columns, codim=codim, separate_axes=separate_axes)
                if self.legend:
                    plot.ax[0].set_title(self.legend[i])
                self.plots.append(plot)
        else:
            figure = plt.figure(self.fig_ids[0])
            sync_timer = sync_timer or figure.canvas.new_timer()
            if grid.dim == 2:
                plot = MatplotlibPatchAxes(U=U, figure=figure, sync_timer=sync_timer, grid=grid, vmin=self.vmins,
                                           vmax=self.vmaxs, bounding_box=bounding_box, codim=codim, columns=columns,
                                           colorbar=True)
            else:
                plot = Matplotlib1DAxes(U=U, figure=figure, sync_timer=sync_timer, grid=grid, vmin=self.vmins,
                                        vmax=self.vmaxs, columns=columns, codim=codim, separate_axes=separate_axes)
            if self.legend:
                plot.ax[0].set_title(self.legend[0])
            self.plots.append(plot)

        if do_animation:
            for fig_id in self.fig_ids:
                # avoids figure double display
                plt.close(fig_id)
            html = [p.html for p in self.plots]
            template = """<div style="float: left; padding: 10px;">{0}</div>"""
            # IPython display system checks for presence and calls this func
            self._repr_html_ = lambda: '\n'.join(template.format(a._repr_html_()) for a in html)
        else:
            self._out = widgets.Output()
            with self._out:
                plt.show()
            # avoids figure double display
            plt.close()
            # IPython display system checks for presence and calls this func
            self._ipython_display_ = self._out._ipython_display_


def visualize_patch(grid, U, bounding_box=None, codim=2, title=None, legend=None,
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
        A bounding box in which the grid is contained (defaults to grid.bounding_box()).
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
            super(Plot, self).__init__(U, grid, codim, legend, bounding_box=bounding_box, columns=columns,
                                       separate_colorbars=separate_colorbars, separate_plots=True,
                                       separate_axes=False)

    return Plot()


def visualize_matplotlib_1d(grid, U, codim=1, title=None, legend=None, separate_plots=True, separate_axes=False,
                            columns=2):
    """Visualize scalar data associated to a one-dimensional |Grid| as a plot.

    The grid's |ReferenceElement| must be the line. The data can either
    be attached to the subintervals or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    U
        |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
        as an animation in a single axes object or a series of axes, depending on the
        `separate_axes` switch. It is also possible to provide a tuple of |VectorArrays|,
        in which case several plots are made into one or multiple figures,
        depending on the `separate_plots` switch. The lengths of all arrays have to agree.
    codim
        The codimension of the entities the data in `U` is attached to (either 0 or 1).
    title
        Title of the plot.
    legend
        Description of the data that is plotted. Most useful if `U` is a tuple in which
        case `legend` has to be a tuple of strings of the same length.
    separate_plots
        If `True`, use multiple figures to visualize multiple |VectorArrays|.
    separate_axes
        If `True`, use separate axes for each figure instead of an Animation.
    column
        Number of columns the subplots are organized in.
    """

    class Plot(MPLPlotBase):

        def _set_limits(self, np_U):
            if separate_plots:
                if separate_axes:
                    self.vmins = tuple(np.min(u) for u in np_U)
                    self.vmaxs = tuple(np.max(u) for u in np_U)
                else:
                    self.vmins = (min(np.min(u) for u in np_U),) * len(np_U)
                    self.vmaxs = (max(np.max(u) for u in np_U),) * len(np_U)
            else:
                self.vmins = min(np.min(u) for u in np_U)
                self.vmaxs = max(np.max(u) for u in np_U)

        def __init__(self):
            super(Plot, self).__init__(U, grid, codim, legend, separate_plots=separate_plots, columns=columns,
                                       separate_axes=separate_axes)

    return Plot()
