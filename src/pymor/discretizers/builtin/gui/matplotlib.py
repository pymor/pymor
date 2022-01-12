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
config.require('QT')

import numpy as np
from qtpy.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymor.discretizers.builtin.grids.referenceelements import triangle, square
from pymor.discretizers.builtin.gui.matplotlib_base import MatplotlibPatchAxes


# noinspection PyShadowingNames
class Matplotlib1DWidget(FigureCanvas):

    def __init__(self, U, parent, grid, count, vmin=None, vmax=None, legend=None, codim=1,
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
                pad = (vmax[i] - vmin[i]) * 0.1
                axes.set_ylim(vmin[i] - pad, vmax[i] + pad)
            l, = axes.plot(xs, np.zeros_like(xs))
            lines = lines + (l,)
            if legend and separate_plots:
                axes.legend([legend[i]])
        if not separate_plots:
            if max(vmax) == min(vmin):
                pad = 0.5
            else:
                pad = (max(vmax) - min(vmin)) * 0.1
            axes.set_ylim(min(vmin) - pad, max(vmax) + pad)
            if legend:
                axes.legend(legend)
        self.lines = lines

        super().__init__(figure)
        self.setParent(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

    def set(self, U, ind):
        for l, u in zip(self.lines, U):
            if self.codim == 1:
                if self.periodic:
                    l.set_ydata(np.concatenate((u[ind], [u[ind][0]])))
                else:
                    l.set_ydata(u[ind])
            else:
                l.set_ydata(np.repeat(u[ind], 2))
        self.draw()


class MatplotlibPatchWidget(FigureCanvas):

    def __init__(self, parent, grid, bounding_box=None, vmin=None, vmax=None, codim=2, dpi=100):
        assert grid.reference_element in (triangle, square)
        assert grid.dim == 2
        assert codim in (0, 2)

        self.figure = Figure(dpi=dpi)
        super().__init__(self.figure)

        self.setParent(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        self.patch_axes = MatplotlibPatchAxes(figure=self.figure, grid=grid, bounding_box=bounding_box,
                                              vmin=vmin, vmax=vmax, codim=codim)

    def set(self, U, vmin=None, vmax=None):
        self.patch_axes.set(U, vmin, vmax)
        self.draw()
