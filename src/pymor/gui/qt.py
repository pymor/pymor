# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip
import math as m

import numpy as np

from PySide.QtGui import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QApplication, QLCDNumber,
                          QSizePolicy, QAction, QStyle, QToolBar, QLabel, QFileDialog)
from PySide.QtCore import Qt, QCoreApplication, QTimer
from pymor.core import BasicInterface
from pymor.la.interfaces import Communicable
from pymor.la import NumpyVectorArray
from pymor.grids import RectGrid, TriaGrid, OnedGrid
from pymor.gui.glumpy import GlumpyPatchWidget, ColorBarWidget
from pymor.gui.matplotlib import Matplotlib1DWidget
from pymor.tools.vtkio import write_vtk


class PlotMainWindow(QWidget):
    def __init__(self, U, plot, length=1, title=None):
        super(PlotMainWindow, self).__init__()

        layout = QVBoxLayout()

        if title:
            title = QLabel('<b>' + title + '</b>')
            title.setAlignment(Qt.AlignHCenter)
            layout.addWidget(title)
        layout.addWidget(plot)

        plot.set(U, 0)

        if length > 1:
            hlayout = QHBoxLayout()

            self.slider = QSlider(Qt.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(length - 1)
            self.slider.setTickPosition(QSlider.TicksBelow)
            hlayout.addWidget(self.slider)

            lcd = QLCDNumber(m.ceil(m.log10(length)))
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
            if hasattr(self, 'save'):
                self.a_save = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), 'Save', self)
                toolbar.addAction(self.a_save)
                self.a_save.triggered.connect(self.save)
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

        elif hasattr(self, 'save'):
            hlayout = QHBoxLayout()
            toolbar = QToolBar()
            self.a_save = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), 'Save', self)
            toolbar.addAction(self.a_save)
            hlayout.addWidget(toolbar)
            layout.addLayout(hlayout)
            self.a_save.triggered.connect(self.save)

        self.setLayout(layout)
        self.plot = plot
        self.U = U
        self.length = length

    def slider_changed(self, ind):
        self.plot.set(self.U, ind)

    def speed_changed(self, val):
        self.timer.setInterval(val * 20)

    def update_solution(self):
        ind = self.slider.value() + 1
        if ind >= self.length:
            if self.a_loop.isChecked():
                ind = 0
            else:
                self.a_play.setChecked(False)
                return
        self.slider.setValue(ind)

    def toggle_play(self, checked):
        if checked:
            if self.slider.value() + 1 == self.length:
                self.slider.setValue(0)
            self.timer.start()
        else:
            self.timer.stop()

    def rewind(self):
        self.slider.setValue(0)

    def to_end(self):
        self.a_play.setChecked(False)
        self.slider.setValue(self.length - 1)

    def step_forward(self):
        self.a_play.setChecked(False)
        ind = self.slider.value() + 1
        if ind == self.length and self.a_loop.isChecked():
            ind = 0
        if ind < self.length:
            self.slider.setValue(ind)

    def step_backward(self):
        self.a_play.setChecked(False)
        ind = self.slider.value() - 1
        if ind == -1 and self.a_loop.isChecked():
            ind = self.length - 1
        if ind >= 0:
            self.slider.setValue(ind)


def launch_qt_app(main_window_factory, block):

    def doit():
        try:
            app = QApplication([])
        except RuntimeError:
            app = QCoreApplication.instance()
        main_window = main_window_factory()
        main_window.show()
        app.exec_()

    from multiprocessing import Process
    p = Process(target=doit)
    p.start()
    if block:
        p.join()


def visualize_glumpy_patch(grid, U, bounding_box=[[0, 0], [1, 1]], codim=2, title=None, legend=None,
                           separate_colorbars=False, block=False):

    class MainWindow(PlotMainWindow):
        def __init__(self, grid, U, bounding_box, codim, title, legend, separate_colorbars):
            assert isinstance(U, Communicable) or isinstance(U, tuple) and all(isinstance(u, Communicable) for u in U) \
                and all(len(u) == len(U[0]) for u in U)
            U = (U.data,) if isinstance(U, Communicable) else tuple(u.data for u in U)
            if isinstance(legend, str):
                legend = (legend,)
            assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)

            class PlotWidget(QWidget):
                def __init__(self):
                    super(PlotWidget, self).__init__()
                    if separate_colorbars:
                        vmins = tuple(np.min(u) for u in U)
                        vmaxs = tuple(np.max(u) for u in U)
                    else:
                        vmins = (min(np.min(u) for u in U),) * len(U)
                        vmaxs = (max(np.max(u) for u in U),) * len(U)
                    layout = QHBoxLayout()
                    plot_layout = QGridLayout()
                    plots = [GlumpyPatchWidget(self, grid, vmin=vmin, vmax=vmax, bounding_box=bounding_box, codim=codim)
                             for vmin, vmax in izip(vmins, vmaxs)]
                    if legend:
                        for i, plot, l in izip(xrange(len(plots)), plots, legend):
                            subplot_layout = QVBoxLayout()
                            caption = QLabel(l)
                            caption.setAlignment(Qt.AlignHCenter)
                            subplot_layout.addWidget(caption)
                            if not separate_colorbars:
                                subplot_layout.addWidget(plot)
                            else:
                                hlayout = QHBoxLayout()
                                hlayout.addWidget(plot)
                                hlayout.addWidget(ColorBarWidget(self, vmin=vmins[i], vmax=vmaxs[i]))
                                subplot_layout.addLayout(hlayout)
                            plot_layout.addLayout(subplot_layout, int(i/2), (i % 2), 1, 1)
                    else:
                        for i, plot in enumerate(plots):
                            if not separate_colorbars:
                                plot_layout.addWidget(plot, int(i/2), (i % 2), 1, 1)
                            else:
                                hlayout = QHBoxLayout()
                                hlayout.addWidget(plot)
                                hlayout.addWidget(ColorBarWidget(self, vmin=vmins[i], vmax=vmaxs[i]))
                                plot_layout.addLayout(plot, int(i/2), (i % 2), 1, 1)
                    layout.addLayout(plot_layout)
                    if not separate_colorbars:
                        layout.addWidget(ColorBarWidget(self, vmin=vmin, vmax=vmax))
                    self.setLayout(layout)
                    self.plots = plots

                def set(self, U, ind):
                    for u, plot in izip(U, self.plots):
                        plot.set(u[ind])

            super(MainWindow, self).__init__(U, PlotWidget(), title=title, length=len(U[0]))
            self.grid = grid
            self.codim = codim

        def save(self):
            filename = QFileDialog.getSaveFileName(self, 'Save as vtk file')[0]
            base_name = filename.split('.vtu')[0].split('.vtk')[0].split('.pvd')[0]
            if base_name:
                if len(self.U) == 1:
                    write_vtk(self.grid, NumpyVectorArray(self.U[0], copy=False), base_name, codim=self.codim)
                else:
                    for i, u in enumerate(self.U):
                        write_vtk(self.grid, NumpyVectorArray(u, copy=False), '{}-{}'.format(base_name, i),
                                  codim=self.codim)

    launch_qt_app(lambda: MainWindow(grid, U, bounding_box, codim, title=title, legend=legend,
                                     separate_colorbars=separate_colorbars), block)



def visualize_matplotlib_1d(grid, U, codim=1, title=None, legend=None, block=False):

    class MainWindow(PlotMainWindow):
        def __init__(self, grid, U, codim, title, legend):
            assert isinstance(U, Communicable) or isinstance(U, tuple) and all(isinstance(u, Communicable) for u in U) \
                and all((len(u) == len(U[0]) and u.dim == U[0].dim) for u in U)
            U = (U.data,) if isinstance(U, Communicable) else tuple(u.data for u in U)
            if isinstance(legend, str):
                legend = (legend,)
            assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)

            plot_widget = Matplotlib1DWidget(None, grid, count=len(U), vmin=np.min(U), vmax=np.max(U), legend=legend, codim=codim)
            super(MainWindow, self).__init__(U, plot_widget, title=title, length=len(U[0]))
            self.grid = grid

    launch_qt_app(lambda: MainWindow(grid, U, codim, title=title, legend=legend), block)


class GlumpyPatchVisualizer(BasicInterface):

    def __init__(self, grid, bounding_box=[[0, 0], [1, 1]], codim=2, block=False):
        assert isinstance(grid, (RectGrid, TriaGrid))
        assert codim in (0, 2)
        self.grid = grid
        self.bounding_box = bounding_box
        self.codim = codim
        self.block = block

    def visualize(self, U, discretization, title=None, legend=None, separate_colorbars=False, block=None, filename=None):
        assert isinstance(U, (Communicable, tuple))
        if filename:
            if isinstance(U, Communicable):
                write_vtk(self.grid, U, filename, codim=self.codim)
            else:
                for i, u in enumerate(self.U):
                    write_vtk(self.grid, u, '{}-{}'.format(filename, i), codim=self.codim)
        else:
            block = self.block if block is None else block
            visualize_glumpy_patch(self.grid, U, bounding_box=self.bounding_box, codim=self.codim, title=title,
                                   legend=legend, separate_colorbars=separate_colorbars, block=block)


class Matplotlib1DVisualizer(BasicInterface):

    def __init__(self, grid, codim=1, block=False):
        assert isinstance(grid, OnedGrid)
        assert codim in (0, 1)
        self.grid = grid
        self.codim = codim
        self.block = block

    def visualize(self, U, discretization, title=None, legend=None, block=None):
        block = self.block if block is None else block
        visualize_matplotlib_1d(self.grid, U, codim=self.codim, title=title, legend=legend, block=block)
