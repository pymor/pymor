from __future__ import absolute_import, division, print_function

import numpy as np

from PySide import QtGui, QtCore
from PySide.QtCore import Qt

from pymor.la.interfaces import Communicable
from pymor.gui.glumpy import GlumpyPatchWidget


def visualize_glumpy_patch(grid, U):

    class MainWindow(QtGui.QWidget):
        def __init__(self, grid, U):
            assert isinstance(U, Communicable)
            U = U.data
            super(MainWindow, self).__init__()
            layout = QtGui.QVBoxLayout(self)
            plot = GlumpyPatchWidget(self, grid, vmin=np.min(U), vmax=np.max(U))
            layout.addWidget(plot)
            if len(U) == 1:
                plot.set(U.ravel())
            else:
                plot.set(U[0])
                slider = QtGui.QSlider(Qt.Horizontal, self)
                slider.setMinimum(0)
                slider.setMaximum(len(U) - 1)
                slider.setTickPosition(QtGui.QSlider.TicksBelow)
                slider.valueChanged.connect(self.slider_changed)
                layout.addWidget(slider)
            self.setLayout(layout)
            self.plot = plot
            self.U = U

        def slider_changed(self, ind):
            self.plot.set(self.U[ind])
            print(ind)

    try:
        app = QtGui.QApplication([])
    except RuntimeError:
        app = QtCore.QCoreApplication.instance()

    win = MainWindow(grid, U)
    win.show()
    app.exec_()
