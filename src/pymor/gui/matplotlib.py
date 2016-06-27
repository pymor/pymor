# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

""" This module provides a widgets for displaying plots of
scalar data assigned to one- and two-dimensional grids using
:mod:`matplotlib`. This widget is not intended to be used directly.
Instead, use :meth:`~pymor.gui.qt.visualize_matplotlib_1d` or
:class:`~pymor.gui.qt.Matplotlib1DVisualizer`.
"""

import numpy as np

try:
    from PySide.QtGui import QSizePolicy
    HAVE_PYSIDE = True
except ImportError:
    HAVE_PYSIDE = False

# matplotlib's default is to use PyQt for Qt4 bindings. However, we use PySide ..
try:
    import matplotlib
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

HAVE_ALL = HAVE_PYSIDE and HAVE_MATPLOTLIB

if HAVE_ALL:
    matplotlib.rcParams['backend.qt4'] = 'PySide'

    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

    from matplotlib.figure import Figure

    from pymor.grids.constructions import flatten_grid
    from pymor.grids.oned import OnedGrid
    from pymor.grids.referenceelements import triangle, square

    # noinspection PyShadowingNames
    class Matplotlib1DWidget(FigureCanvas):

        def __init__(self, parent, grid, count, vmin=None, vmax=None, legend=None, codim=1,
                     separate_plots=False, dpi=100):
            assert isinstance(grid, OnedGrid)
            assert codim in (0, 1)

            figure = Figure(dpi=dpi)
            if not separate_plots:
                axes = figure.gca()
                axes.hold(True)
            self.codim = codim
            lines = ()
            centers = grid.centers(1)
            if grid._identify_left_right:
                centers = np.concatenate((centers, [[grid._domain[1]]]), axis=0)
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

            subentities, coordinates, entity_map = flatten_grid(grid)
            self.subentities = subentities if grid.reference_element is triangle \
                else np.vstack((subentities[:, 0:3], subentities[:, [2, 3, 0]]))
            self.coordinates = coordinates
            self.entity_map = entity_map
            self.reference_element = grid.reference_element
            self.vmin = vmin
            self.vmax = vmax
            self.codim = codim
            self.setParent(parent)
            self.setMinimumSize(300, 300)
            self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        def set(self, U, vmin=None, vmax=None):
            self.vmin = self.vmin if vmin is None else vmin
            self.vmax = self.vmax if vmax is None else vmax
            U = np.array(U)
            f = self.figure
            f.clear()
            a = f.gca()
            if self.codim == 2:
                p = a.tripcolor(self.coordinates[:, 0], self.coordinates[:, 1], self.subentities, U,
                                vmin=self.vmin, vmax=self.vmax, shading='flat')
            elif self.reference_element is triangle:
                p = a.tripcolor(self.coordinates[:, 0], self.coordinates[:, 1], self.subentities, facecolors=U,
                                vmin=self.vmin, vmax=self.vmax, shading='flat')
            else:
                p = a.tripcolor(self.coordinates[:, 0], self.coordinates[:, 1], self.subentities,
                                facecolors=np.tile(U, 2), vmin=self.vmin, vmax=self.vmax, shading='flat')

            self.figure.colorbar(p)
            self.draw()

else:

    class Matplotlib1DWidget(object):
        pass

    class MatplotlibPatchWidget(object):
        pass
