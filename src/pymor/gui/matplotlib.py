# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import math as m

import numpy as np
from PySide.QtOpenGL import QGLWidget
from PySide.QtGui import QSizePolicy, QPainter, QFontMetrics

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pymor.la.numpyvectorarray import NumpyVectorArray
from pymor.grids.referenceelements import line


class Matplotlib1DWidget(FigureCanvas):

    def __init__(self, parent, grid, vmin=None, vmax=None, codim=1, dpi=100):
        assert grid.reference_element is line
        assert codim in (0, 1)

        self.figure = Figure(dpi=dpi)
        self.axes = self.figure.gca()
        self.axes.hold(False)
        self.line, = self.axes.plot(grid.centers(codim), np.zeros_like(grid.centers(codim)),  'b')
        self.axes.set_ylim(vmin, vmax)

        super(Matplotlib1DWidget, self).__init__(self.figure)
        self.setParent(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

    def set(self, U):
        self.line.set_ydata(U)
        self.draw()
