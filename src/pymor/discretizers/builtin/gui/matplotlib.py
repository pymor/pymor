# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Visualization of grid data using matplotlib.

This module provides widgets for displaying plots of
scalar data assigned to one- and two-dimensional grids using
:mod:`matplotlib`. These widgets are not intended to be used directly.
"""
from pymor.core.config import config, is_jupyter
config.require('MATPLOTLIB')
config.require('QT')
from pymor.vectorarrays.interface import VectorArray
    def __init__(self, figure, sync_timer, grid, U, limits, codim=2, separate_axes=False, columns=2,
        assert isinstance(U, VectorArray)
        self.count = len(U) if separate_axes else 1
        if is_jupyter() and not separate_axes and self.count == 1:
            framecount = len(U)
                                                frames=list(range(framecount)), interval=delay_between_frames,
            self.set(self.U, limits=limits)
    def set(self, U, limits):
    def step(self, ind):
        """Change currently displayed |VectorArray| index (of previously ~set array)"""
        pass

    @abstractmethod
    def __init__(self, U, figure, grid, limits, codim=1, separate_axes=False, sync_timer=None,
        assert isinstance(U, VectorArray)
        self.limits = limits
        vmin, vmax = self.limits[0][0], self.limits[0][1]
        super().__init__(U=U, figure=figure, grid=grid, limits=limits, codim=codim, columns=columns,
        vmin, vmax = self.limits[0][0], self.limits[0][1]
        pad = (vmax - vmin) * 0.1
            ax.set_ylim(vmin - pad, vmax + pad)
    def set(self, U, limits=None):
        self.limits = limits or self.limits
        vmin, vmax = self.limits[0][0], self.limits[0][1]
        pad = (vmax - vmin) * 0.1
            ax.set_ylim(vmin - pad, vmax + pad)


class MatplotlibPatchAxes(MatplotlibAxesBase):

    def __init__(self, figure, grid, U, limits, bounding_box=None, codim=2, columns=2,
                 colorbar=True, sync_timer=None):
        assert grid.reference_element in (triangle, square)
        assert grid.dim == 2
        assert codim in (0, 2)
        assert isinstance(U, VectorArray)

        subentities, coordinates, entity_map = flatten_grid(grid)
        self.subentities = subentities if grid.reference_element is triangle \
            else np.vstack((subentities[:, 0:3], subentities[:, [2, 3, 0]]))
        self.coordinates = coordinates
        self.entity_map = entity_map
        self.reference_element = grid.reference_element
        self.colorbar = colorbar
        self.limits = limits
        self.animate = self.step

        if bounding_box is None:
            bounding_box = grid.bounding_box()
        assert len(bounding_box) == 2 and all(len(b) == 2 for b in bounding_box)
        aspect_ratio = (bounding_box[1][1] - bounding_box[0][1]) / (bounding_box[1][0] - bounding_box[0][0])

        super().__init__(U=U, figure=figure, grid=grid, codim=codim, columns=columns, limits=limits,
                         sync_timer=sync_timer, aspect_ratio=aspect_ratio)

    def _plot_init(self):
        if self.codim == 2:
            self.p = self.ax[0].tripcolor(self.coordinates[:, 0], self.coordinates[:, 1], self.subentities,
                                          np.zeros(len(self.coordinates)),
                                          vmin=0, vmax=1, shading='gouraud')
        else:
            self.p = self.ax[0].tripcolor(self.coordinates[:, 0], self.coordinates[:, 1], self.subentities,
                                          facecolors=np.zeros(len(self.subentities)),
                                          vmin=0, vmax=1, shading='flat')
        if self.colorbar:
            # thin plots look ugly with a huge colorbar on the right
            if self.aspect_ratio < 0.75:
                orientation = 'horizontal'
            else:
                orientation = 'vertical'
            self.figure.colorbar(self.p, ax=self.ax[0], orientation=orientation)

    def set(self, U, limits):
        self.U = U
        self.limits = limits
        return self.step(0)

    def step(self, ind):
        assert ind < len(self.U)
        U = self.U[ind].to_numpy()[0]
        if self.codim == 2:
            self.p.set_array(U)
        elif self.reference_element is triangle:
            self.p.set_array(U)
        else:
            self.p.set_array(np.tile(U, 2))
        # limits are always a tuple
        l, r = self.limits[ind]
        self.p.set_clim(l[0], r[0])
        return (self.p,)

import numpy as np
from qtpy.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymor.discretizers.builtin.grids.referenceelements import triangle, square
from pymor.discretizers.builtin.gui.matplotlib_base import MatplotlibPatchAxes


# noinspection PyShadowingNames
class Matplotlib1DWidget(FigureCanvas):

        def __init__(self, vecarray_tuple, parent, grid, count, limits, legend=None, codim=1,
                 separate_plots=False, dpi=100):
        assert isinstance(grid, OnedGrid)
        assert codim in (0, 1)

        figure = Figure(dpi=dpi)
        if not separate_plots:
            axes = figure.gca()
        self.codim = codim
        lines = ()
        centers = grid.centers(1)
        if grid.identify_left_right:
            centers = np.concatenate((centers, [[grid.domain[1]]]), axis=0)
            self.periodic = True
        else:
            self.periodic = False
        if codim == 1:
            xs = centers
        else:
            xs = np.repeat(centers, 2)[1:-1]
        for i in range(count):
            if separate_plots:
                figure.add_subplot(int(count / 2) + count % 2, 2, i + 1)
                axes = figure.gca()
                    pad = (limits[i][1] - limits[i][0]) * 0.1
                    axes.set_ylim(limits[i][0] - pad, limits[i][1] + pad)
            l, = axes.plot(xs, np.zeros_like(xs))
            lines = lines + (l,)
            if legend and separate_plots:
                axes.legend([legend[i]])
        if not separate_plots:
                min_vmin = min((min(l[0]) for l in limits))
                max_vmax = max((max(l[1]) for l in limits))
                if max_vmax == min_vmin:
                pad = 0.5
            else:
                    pad = (max_vmax - min_vmin) * 0.1
                axes.set_ylim(min_vmin - pad, max_vmax + pad)
            if legend:
                axes.legend(legend)
        self.lines = lines

        super().__init__(figure)
        self.setParent(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
            self.set(vecarray_tuple, limits)

        def set(self, vecarray_tuple, limits):
            self.vecarray_tuple = vecarray_tuple
            self.limits = limits
            self.step(0)

        def step(self, ind):
            for l, u in zip(self.lines, self.vecarray_tuple):
                u = u.to_numpy()
            if self.codim == 1:
                if self.periodic:
                    l.set_ydata(np.concatenate((u[ind], [u[ind][0]])))
                else:
                    l.set_ydata(u[ind])
            else:
                l.set_ydata(np.repeat(u[ind], 2))
        self.draw()


class MatplotlibPatchWidget(FigureCanvas):

        def __init__(self, U, limits, parent, grid, bounding_box=None, codim=2, dpi=100):
        assert grid.reference_element in (triangle, square)
        assert grid.dim == 2
        assert codim in (0, 2)
            assert isinstance(U, VectorArray)

        self.figure = Figure(dpi=dpi)
        super().__init__(self.figure)

        self.setParent(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        self.patch_axes = MatplotlibPatchAxes(figure=self.figure, grid=grid, bounding_box=bounding_box,
                                                  U=U, limits=limits, codim=codim)

        def set(self, U, limits):
            self.U = U
            self.limits = limits
            self.step(0)

        def step(self, ind):
            self.patch_axes.step(ind)
        self.draw()
