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

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ModuleNotFoundError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import QSizePolicy

from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymor.discretizers.builtin.grids.referenceelements import square, triangle
from pymor.discretizers.builtin.gui.matplotlib_base import Matplotlib1DAxes, MatplotlibPatchAxes


class Matplotlib1DWidget(FigureCanvas):

    def __init__(self, U, parent, grid, count, legend=None, codim=1,
                 separate_plots=False, columns=2, dpi=100):
        assert isinstance(grid, OnedGrid)
        assert codim in (0, 1)

        figure = Figure(dpi=dpi)
        super().__init__(figure)

        self.setParent(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        self.axes = Matplotlib1DAxes(figure, grid, count, legend=legend, codim=codim,
                                     separate_plots=separate_plots, columns=columns)

    def set(self, U, vmin=None, vmax=None):
        self.axes.set(U, vmin=vmin, vmax=vmax)
        self.draw()


class MatplotlibPatchWidget(FigureCanvas):

    def __init__(self, parent, grid, bounding_box=None, codim=2, dpi=100):
        assert grid.reference_element in (triangle, square)
        assert grid.dim == 2
        assert codim in (0, 2)

        self.figure = Figure(dpi=dpi)
        super().__init__(self.figure)

        self.setParent(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        self.patch_axes = MatplotlibPatchAxes(self.figure.gca(), grid, bounding_box=bounding_box, codim=codim)

    def set(self, U, vmin=None, vmax=None):
        self.patch_axes.set(U, vmin, vmax)
        self.draw()
