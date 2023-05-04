# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('MATPLOTLIB')


import matplotlib.pyplot as plt
import numpy as np

from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymor.discretizers.builtin.gui.matplotlib_base import Matplotlib1DAxes, MatplotlibPatchAxes
from pymor.vectorarrays.interface import VectorArray


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
    assert isinstance(U, VectorArray) \
           or (isinstance(U, tuple)
               and all(isinstance(u, VectorArray) for u in U)
               and all(len(u) == len(U[0]) for u in U))

    U = (U.to_numpy().astype(np.float64, copy=False),) if isinstance(U, VectorArray) else \
        tuple(u.to_numpy().astype(np.float64, copy=False) for u in U)
    legend = (legend,) if isinstance(legend, str) else legend
    assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)

    from pymor.discretizers.builtin.gui.visualizers import _vmins_vmaxs
    vmins, vmaxs = _vmins_vmaxs(U, separate_colorbars, rescale_colorbars)

    plots = []
    do_animation = len(U[0]) > 1
    if len(U) == 1:
        fig, ax = plt.subplots()
        axs = [ax]
    else:
        rows = int(np.ceil(len(U) / columns))
        figsize = plt.rcParams['figure.figsize']
        fig, axs = plt.subplots(rows, columns, figsize=(figsize[0]*columns, figsize[1]*rows))
        axs = axs.flatten()
        for ax in axs[len(U):]:
            ax.set_axis_off()

    for i, (vmin, vmax, u, ax) in enumerate(zip(vmins, vmaxs, U, axs)):
        plot = MatplotlibPatchAxes(ax, grid, bounding_box=bounding_box, codim=codim)
        plot.set(u[0], vmin=vmin[0], vmax=vmax[0])
        if legend:
            ax.set_title(legend[i])
        plots.append(plot)

    if title is not None:
        fig.suptitle(title)

    if do_animation:
        plt.rcParams['animation.html'] = 'jshtml'
        delay_between_frames = 200  # ms

        fig.patch.set_alpha(0.0)

        def animate(i):
            for p, u, vmin, vmax in zip(plots, U, vmins, vmaxs):
                p.set(u[i], vmin=vmin[i], vmax=vmax[i])

        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, animate, frames=len(U[0]), interval=delay_between_frames, blit=False)
        plt.close(fig)
        return anim
    else:
        plt.show()


def visualize_matplotlib_1d(grid, U, codim=1, title=None, legend=None, separate_plots=False,
                            rescale_axes=False, columns=2):
    """Visualize scalar data associated to a one-dimensional |Grid| as a plot.

    The grid's |ReferenceElement| must be the line. The data can either
    be attached to the subintervals or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    U
        |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
        as an animation. It is also possible to provide a tuple of |VectorArrays|,
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
    rescale_axes
        If `True`, rescale axes to data in each frame.
    columns
        Number of columns the subplots are organized in.
    """
    assert isinstance(grid, OnedGrid)
    assert codim in (0, 1)

    assert isinstance(U, VectorArray) \
           or (isinstance(U, tuple)
               and all(isinstance(u, VectorArray) for u in U)
               and all(len(u) == len(U[0]) for u in U))
    U = (U.to_numpy().astype(np.float64, copy=False),) if isinstance(U, VectorArray) else \
        tuple(u.to_numpy().astype(np.float64, copy=False) for u in U)

    if isinstance(legend, str):
        legend = (legend,)
    assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)

    from pymor.discretizers.builtin.gui.visualizers import _vmins_vmaxs
    vmins, vmaxs = _vmins_vmaxs(U, separate_plots, rescale_axes)

    do_animation = len(U[0]) > 1

    figsize = plt.rcParams['figure.figsize']
    if separate_plots:
        rows = int(np.ceil(len(U) / columns))
        figsize = (figsize[0]*columns, figsize[1]*rows)
    fig = plt.figure(figsize=figsize)
    plot = Matplotlib1DAxes(fig, grid, len(U), legend=legend, codim=codim, separate_plots=separate_plots,
                            columns=columns)

    if title is not None:
        fig.suptitle(title)

    if do_animation:
        plt.rcParams['animation.html'] = 'jshtml'
        delay_between_frames = 200  # ms

        fig.patch.set_alpha(0.0)

        def animate(ind):
            plot.set([u[ind] for u in U],
                     [vmin[ind] for vmin in vmins],
                     [vmax[ind] for vmax in vmaxs])

        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, animate, frames=len(U[0]), interval=delay_between_frames, blit=False)
        plt.close(fig)
        return anim
    else:
        plot.set(U,
                 [vmin[0] for vmin in vmins],
                 [vmax[0] for vmax in vmaxs])
        plt.show()
