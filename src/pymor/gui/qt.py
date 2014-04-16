# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

''' This module provides a few methods and classes for visualizing data
associated to grids. We use the `PySide <http://www.pyside.org>`_ bindings
for the `Qt <http://www.qt-project.org>`_ widget toolkit for the GUI.
'''

from __future__ import absolute_import, division, print_function

from itertools import izip
import math as m

import numpy as np

try:
    from PySide.QtGui import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QApplication, QLCDNumber,
                              QAction, QStyle, QToolBar, QLabel, QFileDialog, QMessageBox)
    from PySide.QtCore import Qt, QCoreApplication, QTimer
    HAVE_PYSIDE = True
except ImportError:
    HAVE_PYSIDE = False

from pymor import defaults
from pymor.core import BasicInterface, getLogger
from pymor.grids import RectGrid, TriaGrid, OnedGrid
from pymor.gui.glumpy import GlumpyPatchWidget, ColorBarWidget, HAVE_GLUMPY, HAVE_GL
from pymor.gui.matplotlib import Matplotlib1DWidget, MatplotlibPatchWidget, HAVE_MATPLOTLIB
from pymor.la import VectorArrayInterface, NumpyVectorArray
from pymor.tools.vtkio import HAVE_PYVTK, write_vtk


if HAVE_PYSIDE:

    class PlotMainWindow(QWidget):
        '''Base class for plot main windows.'''

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
                self.a_step_backward = QAction(self.style().standardIcon(QStyle.SP_MediaSkipBackward),
                                               'Step Back', self)
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
    '''Wrapper to display plot in a separate process.'''

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


def visualize_patch(grid, U, bounding_box=[[0, 0], [1, 1]], codim=2, title=None, legend=None,
                    separate_colorbars=False, backend=None, block=False):
    '''Visualize scalar data associated to a two-dimensional |Grid| as a patch plot.

    The grid's |ReferenceElement| must be the triangle or square. The data can either
    be attached to the faces or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    U
        |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
        as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
        provided, in which case a subplot is created for each entry of the tuple. The
        lengths of all arrays have to agree.
    bounding_box
        A bounding box in which the grid is contained.
    codim
        The codimension of the entities the data in `U` is attached to (either 0 or 2).
    title
        Title of the plot.
    legend
        Description of the data that is plotted. Most useful if `U` is a tuple in which
        case `legend` has to be a tuple of strings of the same length.
    separate_colorbars
        If `True` use separate colorbars for each subplot.
    backend
        Plot backend to use ('gl' or 'matplotlib'). If `None`, the
        `qt_visualize_patch_backend` |default| is used.
    block
        If `True` block execution until the plot window is closed.
    '''
    if not HAVE_PYSIDE:
        raise ImportError('cannot visualize: import of PySide failed')

    if backend is None:
        backend = defaults.qt_visualize_patch_backend
    assert backend in {'gl', 'matplotlib'}

    if backend == 'gl':
        if not HAVE_GL:
            raise ImportError('cannot visualize: import of PyOpenGL failed')
        if not HAVE_GLUMPY:
            raise ImportError('cannot visualize: import of glumpy failed')
    else:
        if not HAVE_MATPLOTLIB:
            raise ImportError('cannot visualize: import of matplotlib failed')

    class MainWindow(PlotMainWindow):
        def __init__(self, grid, U, bounding_box, codim, title, legend, separate_colorbars, backend):
            assert isinstance(U, VectorArrayInterface) and hasattr(U, 'data') \
                or (isinstance(U, tuple) and all(isinstance(u, VectorArrayInterface) and hasattr(u, 'data') for u in U)
                    and all(len(u) == len(U[0]) for u in U))
            U = (U.data,) if hasattr(U, 'data') else tuple(u.data for u in U)
            if isinstance(legend, str):
                legend = (legend,)
            assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)
            if backend == 'gl':
                widget = GlumpyPatchWidget
            else:
                widget = MatplotlibPatchWidget
                if not separate_colorbars and len(U) > 1:
                    l = getLogger('pymor.gui.qt.visualize_patch')
                    l.warn('separate_colorbars=False not supported for matplotlib backend')
                separate_colorbars = True

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
                    plots = [widget(self, grid, vmin=vmin, vmax=vmax, bounding_box=bounding_box, codim=codim)
                             for vmin, vmax in izip(vmins, vmaxs)]
                    if legend:
                        for i, plot, l in izip(xrange(len(plots)), plots, legend):
                            subplot_layout = QVBoxLayout()
                            caption = QLabel(l)
                            caption.setAlignment(Qt.AlignHCenter)
                            subplot_layout.addWidget(caption)
                            if not separate_colorbars or backend == 'matplotlib':
                                subplot_layout.addWidget(plot)
                            else:
                                hlayout = QHBoxLayout()
                                hlayout.addWidget(plot)
                                hlayout.addWidget(ColorBarWidget(self, vmin=vmins[i], vmax=vmaxs[i]))
                                subplot_layout.addLayout(hlayout)
                            plot_layout.addLayout(subplot_layout, int(i/2), (i % 2), 1, 1)
                    else:
                        for i, plot in enumerate(plots):
                            if not separate_colorbars or backend == 'matplotlib':
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
            if not HAVE_PYVTK:
                msg = QMessageBox(QMessageBox.Critical, 'Error', 'VTK output disabled. Pleas install pyvtk.')
                msg.exec_()
                return
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
                                     separate_colorbars=separate_colorbars, backend=backend), block)


def visualize_matplotlib_1d(grid, U, codim=1, title=None, legend=None, block=False):
    '''Visualize scalar data associated to a one-dimensional |Grid| as a plot.

    The grid's |ReferenceElement| must be the line. The data can either
    be attached to the subintervals or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    U
        |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
        as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
        provided, in which case several plots are made into the same axes. The
        lengths of all arrays have to agree.
    codim
        The codimension of the entities the data in `U` is attached to (either 0 or 1).
    title
        Title of the plot.
    legend
        Description of the data that is plotted. Most useful if `U` is a tuple in which
        case `legend` has to be a tuple of strings of the same length.
    block
        If `True` block execution until the plot window is closed.
    '''
    if not HAVE_PYSIDE:
        raise ImportError('cannot visualize: import of PySide failed')
    if not HAVE_MATPLOTLIB:
        raise ImportError('cannot visualize: import of matplotlib failed')

    class MainWindow(PlotMainWindow):
        def __init__(self, grid, U, codim, title, legend):
            assert isinstance(U, VectorArrayInterface) and hasattr(U, 'data') \
                or (isinstance(U, tuple) and all(isinstance(u, VectorArrayInterface) and hasattr(u, 'data') for u in U)
                    and all(len(u) == len(U[0]) for u in U))
            U = (U.data,) if hasattr(U, 'data') else tuple(u.data for u in U)
            if isinstance(legend, str):
                legend = (legend,)
            assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)

            plot_widget = Matplotlib1DWidget(None, grid, count=len(U), vmin=np.min(U), vmax=np.max(U),
                                             legend=legend, codim=codim)
            super(MainWindow, self).__init__(U, plot_widget, title=title, length=len(U[0]))
            self.grid = grid

    launch_qt_app(lambda: MainWindow(grid, U, codim, title=title, legend=legend), block)


class PatchVisualizer(BasicInterface):
    '''Visualize scalar data associated to a two-dimensional |Grid| as a patch plot.

    The grid's |ReferenceElement| must be the triangle or square. The data can either
    be attached to the faces or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    bounding_box
        A bounding box in which the grid is contained.
    codim
        The codimension of the entities the data in `U` is attached to (either 0 or 2).
    backend
        Plot backend to use ('gl' or 'matplotlib'). If `None`, the
        `qt_visualize_patch_backend` |default| is used.
    block
        If `True` block execution until the plot window is closed.
    '''

    def __init__(self, grid, bounding_box=[[0, 0], [1, 1]], codim=2, backend=None, block=False):
        assert isinstance(grid, (RectGrid, TriaGrid))
        assert codim in (0, 2)
        self.grid = grid
        self.bounding_box = bounding_box
        self.codim = codim
        self.backend = backend
        self.block = block

    def visualize(self, U, discretization, title=None, legend=None, separate_colorbars=False,
                  block=None, filename=None):
        '''Visualize the provided data.

        Parameters
        ----------
        U
            |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
            as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
            provided, in which case a subplot is created for each entry of the tuple. The
            lengths of all arrays have to agree.
        discretization
            Filled in :meth:`pymor.discretizations.DiscretizationBase.visualize` (ignored).
        title
            Title of the plot.
        legend
            Description of the data that is plotted. Most useful if `U` is a tuple in which
            case `legend` has to be a tuple of strings of the same length.
        separate_colorbars
            If `True` use separate colorbars for each subplot.
        block
            If `True` block execution until the plot window is closed. If `None`, use the
            default provided during instantiation.
        filename
            If specified, write the data to a VTK-file using
            :func:`pymor.tools.vtkio.write_vtk` instead of displaying it.
        '''
        assert isinstance(U, VectorArrayInterface) and hasattr(U, 'data') \
            or (isinstance(U, tuple) and all(isinstance(u, VectorArrayInterface) and hasattr(u, 'data') for u in U)
                and all(len(u) == len(U[0]) for u in U))
        if filename:
            if not isinstance(U, tuple):
                write_vtk(self.grid, U, filename, codim=self.codim)
            else:
                for i, u in enumerate(self.U):
                    write_vtk(self.grid, u, '{}-{}'.format(filename, i), codim=self.codim)
        else:
            block = self.block if block is None else block
            visualize_patch(self.grid, U, bounding_box=self.bounding_box, codim=self.codim, title=title,
                            legend=legend, separate_colorbars=separate_colorbars, backend=self.backend,
                            block=block)


class Matplotlib1DVisualizer(BasicInterface):
    '''Visualize scalar data associated to a one-dimensional |Grid| as a plot.

    The grid's |ReferenceElement| must be the line. The data can either
    be attached to the subintervals or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    codim
        The codimension of the entities the data in `U` is attached to (either 0 or 1).
    block
        If `True` block execution until the plot window is closed.
    '''

    def __init__(self, grid, codim=1, block=False):
        assert isinstance(grid, OnedGrid)
        assert codim in (0, 1)
        self.grid = grid
        self.codim = codim
        self.block = block

    def visualize(self, U, discretization, title=None, legend=None, block=None):
        '''Visualize the provided data.

        Parameters
        ----------
        U
            |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
            as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
            provided, in which case several plots are made into the same axes. The
            lengths of all arrays have to agree.
        discretization
            Filled in :meth:`pymor.discretizations.DiscretizationBase.visualize` (ignored).
        title
            Title of the plot.
        legend
            Description of the data that is plotted. Most useful if `U` is a tuple in which
            case `legend` has to be a tuple of strings of the same length.
        block
            If `True` block execution until the plot window is closed. If `None`, use the
            default provided during instantiation.
        '''
        block = self.block if block is None else block
        visualize_matplotlib_1d(self.grid, U, codim=self.codim, title=title, legend=legend, block=block)
