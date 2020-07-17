# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
import itertools

import numpy as np
from ipywidgets import HTML, HBox
import matplotlib.pyplot as plt

from pymor.core.config import config
from pymor.discretizers.builtin.gui.matplotlib import MatplotlibPatchAxes, Matplotlib1DAxes
from pymor.vectorarrays.interface import VectorArray


class concat_display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">{0}</div>"""

    def __init__(self, *args):
        self.args = args

    def _repr_html_(self):
        return '\n'.join(self.template.format(a._repr_html_()) for a in self.args)

    def __repr__(self):
        return '\n\n'.join(repr(a) for a in self.args)


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

    assert isinstance(U, VectorArray) \
        or (isinstance(U, tuple)
            and all(isinstance(u, VectorArray) for u in U)
            and all(len(u) == len(U[0]) for u in U))
    U = (U.to_numpy().astype(np.float64, copy=False),) if isinstance(U, VectorArray) else \
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
                # todo rescaling not set up
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


            rows = int(np.ceil(len(U) / columns))
            self.figure = figure = plt.figure()

            self.plots = plots = []
            axes = plt.subplots(nrows=rows, ncols=columns, squeeze=False)
            coord = itertools.product(range(rows), range(columns))
            for i, (vmin, vmax, u, c) in enumerate(zip(self.vmins, self.vmaxs, U, coord)):
                ax = axes[c]

                plots.append(MatplotlibPatchAxes(U=u, ax=ax, figure=figure, grid=grid, bounding_box=bounding_box, vmin=vmin, vmax=vmax,
                                                 codim=codim, colorbar=separate_colorbars or i == len(U)-1))
                if legend:
                    ax.set_title(legend[i])


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

    if len(U[0]) > 1:
        # otherwise the subplot displays twice
        plt.close(plot.figure)
        return concat_display(*[p.html for p in plot.plots])

    return plot

        # otherwise the subplot displays twice
        plt.close(plot.figure)
        return concat_display(*[p.html for p in plot.plots])

    return plot



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
    assert isinstance(U, VectorArray) \
        or (isinstance(U, tuple)
            and all(isinstance(u, VectorArray) for u in U)
            and all(len(u) == len(U[0]) for u in U))
    U = (U.to_numpy(),) if isinstance(U, VectorArray) else tuple(u.to_numpy() for u in U)

    if not config.HAVE_MATPLOTLIB:
        raise ImportError('cannot visualize: import of matplotlib failed')
    if not config.HAVE_IPYWIDGETS and len(U[0]) > 1:
        raise ImportError('cannot visualize: import of ipywidgets failed')

    if isinstance(legend, str):
        legend = (legend,)
    assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)


    class Plot:

        def __init__(self):
            if separate_plots:
                if separate_axes:
                    self.vmins = tuple(np.min(u) for u in U[0])
                    self.vmaxs = tuple(np.max(u) for u in U[0])
                else:
                    self.vmins = (min(np.min(u) for u in U),) * len(U[0])
                    self.vmaxs = (max(np.max(u) for u in U),) * len(U[0])
            else:
                self.vmins = (min(np.min(u) for u in U[0]),)
                self.vmaxs = (max(np.max(u) for u in U[0]),)

            import matplotlib.pyplot as plt

            if separate_axes:
                rows = int(np.ceil(len(U[0]) / columns))
            else:
                rows = int(np.ceil(len(U) / columns))

            self.plots = []

            self.figure, axes = plt.subplots(nrows=rows, ncols=columns, squeeze=False)
            coord = itertools.product(range(rows), range(columns))
            for i, (vmin, vmax, u, c) in enumerate(zip(self.vmins, self.vmaxs, U, coord)):
                count = 1
                if not separate_plots:
                    count = len(U[0])
                ax = axes[c]
                self.plots.append(Matplotlib1DAxes(u, ax, self.figure, grid, count, vmin=vmin, vmax=vmax,
                                                   codim=codim))
                if legend:
                    ax.set_title(legend[i])

            plt.tight_layout()

        def set(self, U):
            if separate_plots:
                for u, plot, vmin, vmax in zip(U, self.plots, self.vmins, self.vmaxs):
                    plot.set(u[np.newaxis, ...], vmin=vmin, vmax=vmax)
            else:
                self.plots[0].set(U, vmin=self.vmins[0], vmax=self.vmaxs[0])

    plot = Plot()

    if len(U[0]) > 1: