# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

''' This module provides a widget for displaying plots of
scalar data assigned to a 1D-Grid using :mod:`matplotlib`. This widget is not
intended to be used directly. Instead, use
:meth:`~pymor.gui.qt.visualize_matplotlib_1d` or
:class:`~pymor.gui.qt.Matplotlib1DVisualizer`.
'''

from __future__ import absolute_import, division, print_function

from itertools import izip

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
    from pymor.grids.referenceelements import line, triangle, square

    class Matplotlib1DWidget(FigureCanvas):

        def __init__(self, parent, grid, count, vmin=None, vmax=None, legend=None, codim=1, dpi=100):
            assert grid.reference_element is line
            assert codim in (0, 1)

            self.figure = Figure(dpi=dpi)
            self.axes = self.figure.gca()
            self.axes.hold(True)
            lines = tuple()
            for _ in xrange(count):
                l, = self.axes.plot(grid.centers(codim), np.zeros_like(grid.centers(codim)))
                lines = lines + (l,)
            self.axes.set_ylim(vmin, vmax)
            if legend:
                self.axes.legend(legend)
            self.lines = lines

            super(Matplotlib1DWidget, self).__init__(self.figure)
            self.setParent(parent)
            self.setMinimumSize(300, 300)
            self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        def set(self, U, ind):
            for line, u in izip(self.lines, U):
                line.set_ydata(u[ind])
            self.draw()

    class MatplotlibPatchWidget(FigureCanvas):

        def __init__(self, parent, grid, bounding_box=None, vmin=None, vmax=None, codim=2, dpi=100):
            assert grid.reference_element in (triangle, square)
            assert grid.dim == 2
            assert codim in (0, 2)

            self.figure = Figure(dpi=dpi)
            super(MatplotlibPatchWidget, self).__init__(self.figure)

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

        def set(self, U):
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
