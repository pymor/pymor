# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Visualization of grid data using matplotlib.

This module provides widgets for displaying plots of
scalar data assigned to one- and two-dimensional grids using
:mod:`matplotlib`. These widgets are not intended to be used directly.
"""
from pymor.core.config import config
config.require('MATPLOTLIB')


import numpy as np
from IPython.core.display import HTML
from matplotlib import animation
from pymor.core.base import abstractmethod

from pymor.discretizers.builtin.grids.constructions import flatten_grid
from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymor.discretizers.builtin.grids.referenceelements import triangle, square


class MatplotlibAxesBase:

    def __init__(self, figure, sync_timer, grid, U=None, vmin=None, vmax=None, codim=2, separate_axes=False, columns=2,
                 aspect_ratio=1):
        # aspect_ratio is height/width
        self.vmin = vmin
        self.vmax = vmax
        self.codim = codim

        self.grid = grid
        if separate_axes:
            if len(U) == 1:
                columns = 1  # otherwise we get a sep axes object with 0 data
            rows = int(np.ceil(len(U) / columns))
            self.ax = figure.subplots(rows, columns, squeeze=False).flatten()
        else:
            self.ax = (figure.gca(),)
        for ax in self.ax:
            ax.set_aspect(aspect_ratio)
        self.figure = figure
        self.codim = codim
        self.grid = grid
        self.separate_axes = separate_axes
        self.count = len(U) if separate_axes or isinstance(U, tuple) else 1
        self.aspect_ratio = aspect_ratio

        self._plot_init()

        # assignment delayed to ensure _plot_init works w/o data
        self.U = U
        # Rest is only needed with animation
        if U is not None and not separate_axes and self.count == 1:
            assert len(self.ax) == 1
            delay_between_frames = 200  # ms
            self.anim = animation.FuncAnimation(figure, self.animate,
                                                frames=U, interval=delay_between_frames,
                                                blit=True, event_source=sync_timer)
            # generating the HTML instance outside this class causes the plot display to fail
            self.html = HTML(self.anim.to_jshtml())
        else:
            self.set(self.U)

    @abstractmethod
    def _plot_init(self):
        """Setup MPL figure display with empty data."""
        pass

    @abstractmethod
    def set(self, U):
        """Load new data into existing plot objects."""
        pass

    @abstractmethod
    def animate(self, u):
        """Load new data into existing plot objects."""
        pass


class MatplotlibPatchAxes(MatplotlibAxesBase):

    def __init__(self, figure, grid, bounding_box=None, U=None, vmin=None, vmax=None, codim=2, columns=2,
                 colorbar=True, sync_timer=None):
        assert grid.reference_element in (triangle, square)
        assert grid.dim == 2
        assert codim in (0, 2)

        subentities, coordinates, entity_map = flatten_grid(grid)
        self.subentities = subentities if grid.reference_element is triangle \
            else np.vstack((subentities[:, 0:3], subentities[:, [2, 3, 0]]))
        self.coordinates = coordinates
        self.entity_map = entity_map
        self.reference_element = grid.reference_element
        self.colorbar = colorbar
        self.animate = self.set

        if bounding_box is None:
            bounding_box = grid.bounding_box()
        assert len(bounding_box) == 2 and all(len(b) == 2 for b in bounding_box)
        aspect_ratio = (bounding_box[1][1] - bounding_box[0][1]) / (bounding_box[1][0] - bounding_box[0][0])

        super().__init__(U=U, figure=figure, grid=grid,  vmin=vmin, vmax=vmax, codim=codim, columns=columns,
                         sync_timer=sync_timer, aspect_ratio=aspect_ratio)

    def _plot_init(self):
        if self.codim == 2:
            self.p = self.ax[0].tripcolor(self.coordinates[:, 0], self.coordinates[:, 1], self.subentities,
                                          np.zeros(len(self.coordinates)),
                                          vmin=self.vmin, vmax=self.vmax, shading='gouraud')
        else:
            self.p = self.ax[0].tripcolor(self.coordinates[:, 0], self.coordinates[:, 1], self.subentities,
                                          facecolors=np.zeros(len(self.subentities)),
                                          vmin=self.vmin, vmax=self.vmax, shading='flat')
        if self.colorbar:
            # thin plots look ugly with a huge colorbar on the right
            if self.aspect_ratio < 0.75:
                orientation = 'horizontal'
            else:
                orientation = 'vertical'
            self.figure.colorbar(self.p, ax=self.ax[0], orientation=orientation)

    def set(self, U, vmin=None, vmax=None):
        self.vmin = self.vmin if vmin is None else vmin
        self.vmax = self.vmax if vmax is None else vmax
        if self.codim == 2:
            self.p.set_array(U)
        elif self.reference_element is triangle:
            self.p.set_array(U)
        else:
            self.p.set_array(np.tile(U, 2))
        self.p.set_clim(self.vmin, self.vmax)
        return (self.p,)


class Matplotlib1DAxes(MatplotlibAxesBase):

    def __init__(self, U, figure, grid, vmin=None, vmax=None, codim=1, separate_axes=False, sync_timer=None,
                 columns=2, bounding_box=None):
        assert isinstance(grid, OnedGrid)
        assert codim in (0, 1)

        if bounding_box is None:
            bounding_box = grid.bounding_box()
        aspect_ratio = (bounding_box[1] - bounding_box[0]) / (vmax - vmin)

        super().__init__(U=U, figure=figure, grid=grid, vmin=vmin, vmax=vmax, codim=codim, columns=columns,
                         sync_timer=sync_timer, separate_axes=separate_axes, aspect_ratio=aspect_ratio)

    def _plot_init(self):
        centers = self.grid.centers(1)
        if self.grid.identify_left_right:
            centers = np.concatenate((centers, [[self.grid.domain[1]]]), axis=0)
            self.periodic = True
        else:
            self.periodic = False
        if self.codim == 1:
            xs = centers
        else:
            xs = np.repeat(centers, 2)[1:-1]
        if self.separate_axes:
            self.lines = [ax.plot(xs, np.zeros_like(xs))[0] for ax in self.ax]
        else:
            self.lines = [self.ax[0].plot(xs, np.zeros_like(xs))[0] for _ in range(self.count)]
        pad = (self.vmax - self.vmin) * 0.1
        for ax in self.ax:
            ax.set_ylim(self.vmin - pad, self.vmax + pad)

    def _set(self, u, i):
        if self.codim == 1:
            if self.periodic:
                self.lines[i].set_ydata(np.concatenate((u, [self.U[0]])))
            else:
                self.lines[i].set_ydata(u)
        else:
            self.lines[i].set_ydata(np.repeat(u, 2))

    def animate(self, u):
        for i in range(len(self.ax)):
            self._set(u, i)
        return self.lines

    def set(self, U, vmin=None, vmax=None):
        self.vmin = self.vmin if vmin is None else vmin
        self.vmax = self.vmax if vmax is None else vmax

        if isinstance(U, tuple):
            for i, u in enumerate(U):
                self._set(u, i)
        else:
            for i, (u, _) in enumerate(zip(U, self.ax)):
                self._set(u, i)
        pad = (self.vmax - self.vmin) * 0.1
        for ax in self.ax:
            ax.set_ylim(self.vmin - pad, self.vmax + pad)
