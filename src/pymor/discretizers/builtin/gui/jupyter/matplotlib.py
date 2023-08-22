# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('MATPLOTLIB')


import matplotlib.pyplot as plt
import numpy as np

from pymor.core.base import BasicObject
from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymor.discretizers.builtin.gui.matplotlib_base import Matplotlib1DAxes, MatplotlibPatchAxes
from pymor.vectorarrays.interface import VectorArray


class PatchVisualizer(BasicObject):
    """Patch visualizer."""

    def __init__(self, grid, U, bounding_box=None, codim=2, title=None, legend=None,
                 separate_colorbars=False, rescale_colorbars=False, columns=2):
        assert isinstance(U, VectorArray) \
               or (isinstance(U, tuple)
                   and all(isinstance(u, VectorArray) for u in U)
                   and all(len(u) == len(U[0]) for u in U))

        if isinstance(U, VectorArray):
            U = (U,)
        legend = (legend,) if isinstance(legend, str) else legend
        assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)

        self.separate_colorbars, self.rescale_colorbars = separate_colorbars, rescale_colorbars

        if len(U) == 1:
            rows = columns = 1
        else:
            rows = int(np.ceil(len(U) / columns))

        figsize = plt.rcParams['figure.figsize']
        self.fig = fig = plt.figure(figsize=(figsize[0] * columns, figsize[1] * rows))

        axs = []
        plots = []
        for i in range(len(U)):
            ax = fig.add_subplot(rows, columns, i+1)
            if legend:
                ax.set_title(legend[i])
            plot = MatplotlibPatchAxes(ax, grid, bounding_box=bounding_box, codim=codim)
            axs.append(ax)
            plots.append(plot)

        self.plots = plots
        self.fig = fig

        if title is not None:
            fig.suptitle(title)

        self.set(U)

    def set(self, U=None, idx=0):
        if U is None:
            U = self.U
        else:
            assert isinstance(U, VectorArray) \
                   or (isinstance(U, tuple)
                       and all(isinstance(u, VectorArray) for u in U)
                       and all(len(u) == len(U[0]) for u in U))
            self.U = U = (U.to_numpy().astype(np.float64, copy=False),) if isinstance(U, VectorArray) else \
                tuple(u.to_numpy().astype(np.float64, copy=False) for u in U)
            from pymor.discretizers.builtin.gui.visualizers import _vmins_vmaxs
            self.vmins, self.vmaxs = _vmins_vmaxs(U, self.separate_colorbars, self.rescale_colorbars)

        for vmin, vmax, u, plot in zip(self.vmins, self.vmaxs, U, self.plots):
            plot.set(u[idx], vmin=vmin[idx], vmax=vmax[idx])
        self.fig.canvas.draw_idle()


def visualize_patch(grid, U, bounding_box=None, codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, columns=2,
                    return_widget=True):
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

    if isinstance(U, VectorArray):
        U = (U,)

    if return_widget:
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'widget')
        plt.ioff()

    vis = PatchVisualizer(grid, U, bounding_box=bounding_box, codim=codim, title=title, legend=legend,
        separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars, columns=columns)

    do_animation = len(U[0]) > 1

    if return_widget:
        vis.fig.canvas.header_visible = False
        vis.fig.canvas.layout.flex = '0 1 auto'
        vis.fig.tight_layout()
        if do_animation:
            from ipywidgets import VBox

            from pymor.discretizers.builtin.gui.jupyter.animation_widget import AnimationWidget

            animation_widget = AnimationWidget(len(U[0]))
            widget = VBox([vis.fig.canvas, animation_widget])
            widget.layout.align_items = 'stretch'

            def animate(change):
                vis.set(idx=change['new'])
            animation_widget.frame_slider.observe(animate, 'value')

            def set(U):
                vis.set(U, animation_widget.frame_slider.value)
            widget.set = set
        else:
            widget = vis.fig.canvas
            widget.set = vis.set
        return widget
    else:
        if do_animation:
            plt.rcParams['animation.html'] = 'jshtml'
            delay_between_frames = 200  # ms
            vis.fig.patch.set_alpha(0.0)
            from matplotlib.animation import FuncAnimation

            def animate(i):
                vis.set(idx=i)

            anim = FuncAnimation(vis.fig, animate, frames=len(U[0]), interval=delay_between_frames, blit=False)
            plt.close(vis.fig)
            return anim
        else:
            plt.show()


def visualize_matplotlib_1d(grid, U, codim=1, title=None, legend=None, separate_plots=False,
                            rescale_axes=False, columns=2, return_widget=True):
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

    if return_widget:
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'widget')
        plt.ioff()


    figsize = plt.rcParams['figure.figsize']
    if separate_plots:
        rows = int(np.ceil(len(U) / columns))
        figsize = (figsize[0]*columns, figsize[1]*rows)
    fig = plt.figure(figsize=figsize)
    plot = Matplotlib1DAxes(fig, grid, len(U), legend=legend, codim=codim, separate_plots=separate_plots,
                            columns=columns)

    if title is not None:
        fig.suptitle(title)

    data = [U, vmins, vmaxs]
    def set_data(U=None, ind=0):
        if U is not None:
            U = (U.to_numpy().astype(np.float64, copy=False),) if isinstance(U, VectorArray) else \
                tuple(u.to_numpy().astype(np.float64, copy=False) for u in U)
            vmins, vmaxs = _vmins_vmaxs(U, separate_plots, rescale_axes)
            data[0:3] = U, vmins, vmaxs

        U, vmins, vmaxs = data
        plot.set([u[ind] for u in U],
                 [vmin[ind] for vmin in vmins],
                 [vmax[ind] for vmax in vmaxs])
        fig.canvas.draw_idle()

    if return_widget:
        fig.canvas.header_visible = False
        fig.canvas.layout.flex = '0 1 auto'
        fig.tight_layout()
        if do_animation:
            from ipywidgets import VBox

            from pymor.discretizers.builtin.gui.jupyter.animation_widget import AnimationWidget

            animation_widget = AnimationWidget(len(U[0]))
            widget = VBox([fig.canvas, animation_widget])
            widget.layout.align_items = 'stretch'

            def time_changed(change):
                set_data(U=None, ind=change['new'])
            animation_widget.frame_slider.observe(time_changed, 'value')

            def set(U):
                set_data(U, ind=animation_widget.frame_slider.value)
            widget.set = set
        else:
            widget = fig.canvas
            widget.set = set_data
        return widget
    else:
        if do_animation:
            plt.rcParams['animation.html'] = 'jshtml'
            delay_between_frames = 200  # ms

            fig.patch.set_alpha(0.0)

            from matplotlib.animation import FuncAnimation
            anim = FuncAnimation(fig, lambda ind: set_data(ind=ind), frames=len(U[0]),
                                 interval=delay_between_frames, blit=False)
            plt.close(fig)
            return anim
        else:
            plot.set(U,
                     [vmin[0] for vmin in vmins],
                     [vmax[0] for vmax in vmaxs])
            plt.show()

