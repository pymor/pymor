# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import math as m

import numpy as np

from PySide.QtGui import (QWidget, QVBoxLayout, QHBoxLayout, QSlider, QApplication, QLCDNumber,
                          QSizePolicy, QAction, QStyle, QToolBar, QLabel)
from PySide.QtCore import Qt, QCoreApplication, QTimer

from pymor.la.interfaces import Communicable
from pymor.gui.glumpy import GlumpyPatchWidget, ColorBarWidget


def visualize_glumpy_patch(grid, U, bounding_box=[[0, 0], [1, 1]], codim=2):

    class MainWindow(QWidget):
        def __init__(self, grid, U):
            assert isinstance(U, Communicable)
            super(MainWindow, self).__init__()

            U = U.data

            layout = QVBoxLayout()

            plotBox = QHBoxLayout()
            plot = GlumpyPatchWidget(self, grid, vmin=np.min(U), vmax=np.max(U), bounding_box=bounding_box, codim=codim)
            bar = ColorBarWidget(self, vmin=np.min(U), vmax=np.max(U))
            plotBox.addWidget(plot)
            plotBox.addWidget(bar)
            layout.addLayout(plotBox)

            if len(U) == 1:
                plot.set(U.ravel())
            else:
                plot.set(U[0])

                hlayout = QHBoxLayout()

                self.slider = QSlider(Qt.Horizontal)
                self.slider.setMinimum(0)
                self.slider.setMaximum(len(U) - 1)
                self.slider.setTickPosition(QSlider.TicksBelow)
                hlayout.addWidget(self.slider)

                lcd = QLCDNumber(m.ceil(m.log10(len(U))))
                lcd.setDecMode()
                lcd.setSegmentStyle(QLCDNumber.Flat)
                hlayout.addWidget(lcd)

                layout.addLayout(hlayout)

                hlayout = QHBoxLayout()

                toolbar = QToolBar()
                self.a_play = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), 'Play', self)
                self.a_play.setCheckable(True)
                self.a_rewind = QAction(self.style().standardIcon(QStyle.SP_MediaSeekBackward), 'Rewind', self)
                self.a_toend = QAction(self.style().standardIcon(QStyle.SP_MediaSeekForward), 'End', self)
                self.a_step_backward = QAction(self.style().standardIcon(QStyle.SP_MediaSkipBackward), 'Step Back', self)
                self.a_step_forward = QAction(self.style().standardIcon(QStyle.SP_MediaSkipForward), 'Step', self)
                self.a_loop = QAction(self.style().standardIcon(QStyle.SP_BrowserReload), 'Loop', self)
                self.a_loop.setCheckable(True)
                toolbar.addAction(self.a_play)
                toolbar.addAction(self.a_rewind)
                toolbar.addAction(self.a_toend)
                toolbar.addAction(self.a_step_backward)
                toolbar.addAction(self.a_step_forward)
                toolbar.addAction(self.a_loop)
                hlayout.addWidget(toolbar)

                self.speed = QSlider(Qt.Horizontal)
                self.speed.setMinimum(0)
                self.speed.setMaximum(100)
                hlayout.addWidget(QLabel('Speed:'))
                hlayout.addWidget(self.speed)

                layout.addLayout(hlayout)

                self.timer = QTimer()
                self.timer.timeout.connect(self.update_solution)

                self.slider.valueChanged.connect(self.slider_changed)
                self.slider.valueChanged.connect(lcd.display)
                self.speed.valueChanged.connect(self.speed_changed)
                self.a_play.toggled.connect(self.toggle_play)
                self.a_rewind.triggered.connect(self.rewind)
                self.a_toend.triggered.connect(self.to_end)
                self.a_step_forward.triggered.connect(self.step_forward)
                self.a_step_backward.triggered.connect(self.step_backward)

                self.speed.setValue(50)

            self.setLayout(layout)
            self.plot = plot
            self.U = U

        def slider_changed(self, ind):
            self.plot.set(self.U[ind])

        def speed_changed(self, val):
            self.timer.setInterval(val * 20)

        def update_solution(self):
            ind = self.slider.value() + 1
            if ind >= len(self.U):
                if self.a_loop.isChecked():
                    ind = 0
                else:
                    self.a_play.setChecked(False)
                    return
            self.slider.setValue(ind)

        def toggle_play(self, checked):
            if checked:
                if self.slider.value() + 1 == len(self.U):
                    self.slider.setValue(0)
                self.timer.start()
            else:
                self.timer.stop()

        def rewind(self):
            self.slider.setValue(0)

        def to_end(self):
            self.a_play.setChecked(False)
            self.slider.setValue(len(self.U) - 1)

        def step_forward(self):
            self.a_play.setChecked(False)
            ind = self.slider.value() + 1
            if ind == len(self.U) and self.a_loop.isChecked():
                ind = 0
            if ind < len(self.U):
                self.slider.setValue(ind)

        def step_backward(self):
            self.a_play.setChecked(False)
            ind = self.slider.value() - 1
            if ind == -1 and self.a_loop.isChecked():
                ind = len(self.U) - 1
            if ind >= 0:
                self.slider.setValue(ind)


    try:
        app = QApplication([])
    except RuntimeError:
        app = QCoreApplication.instance()

    win = MainWindow(grid, U)
    win.show()
    app.exec_()
