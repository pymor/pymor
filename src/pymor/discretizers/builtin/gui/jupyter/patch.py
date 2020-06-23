# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
import numpy as np

from pymor.core.config import config
from pymor.discretizers.builtin.gui.matplotlib import MatplotlibPatchAxes, Matplotlib1DAxes
from pymor.vectorarrays.interface import VectorArray


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
                if grid.dim == 2:
                    plots.append(MatplotlibPatchAxes(figure, grid, bounding_box=bounding_box, vmin=vmin, vmax=vmax,
                                                     codim=codim, colorbar=separate_colorbars))
                else:
                    plots.append(Matplotlib1DAxes(figure, grid, bounding_box=bounding_box, vmin=vmin, vmax=vmax,
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

    return None
