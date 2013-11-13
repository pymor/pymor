# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip
import math as m

import numpy as np
from PySide.QtOpenGL import QGLWidget
from PySide.QtGui import QSizePolicy, QPainter, QFontMetrics

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pymor.grids.referenceelements import line
from pymor.la.numpyvectorarray import NumpyVectorArray


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
