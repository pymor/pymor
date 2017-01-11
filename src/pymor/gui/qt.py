# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

""" This module provides a few methods and classes for visualizing data
associated to grids. We use the `PySide <http://www.pyside.org>`_ bindings
for the `Qt <http://www.qt-project.org>`_ widget toolkit for the GUI.
"""

import math as m

import numpy as np

import multiprocessing
import os
import signal
import time

import sys

from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.core.interfaces import BasicInterface
from pymor.core.logger import getLogger
from pymor.core.exceptions import PySideMissing
from pymor.grids.oned import OnedGrid
from pymor.grids.referenceelements import triangle, square
from pymor.gui.gl import GLPatchWidget, ColorBarWidget
from pymor.gui.matplotlib import Matplotlib1DWidget, MatplotlibPatchWidget
from pymor.tools.vtkio import write_vtk
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace

if config.HAVE_PYSIDE:
    from PySide.QtGui import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QApplication, QLCDNumber,
                              QAction, QStyle, QToolBar, QLabel, QFileDialog, QMessageBox)
    from PySide.QtCore import Qt, QCoreApplication, QTimer

    class PlotMainWindow(QWidget):
        """Base class for plot main windows."""

        def __init__(self, U, plot, length=1, title=None):
            super().__init__()

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


_launch_qt_app_pids = set()



def _doit():
    try:
        app = QApplication([])
    except RuntimeError:
        app = QCoreApplication.instance()
    main_window = main_window_factory()
    main_window.show()
    app.exec_()

def _launch_qt_app(main_window_factory, block):
    """Wrapper to display plot in a separate process."""

    import sys
    if block and not getattr(sys, '_called_from_test', False):
        _doit()
    else:
        p = multiprocessing.Process(target=_doit)
        p.start()
        _launch_qt_app_pids.add(p.pid)


def stop_gui_processes():
    for p in multiprocessing.active_children():
        if p.pid in _launch_qt_app_pids:
            try:
                # kill is needed on linux, term on win32
                sig = getattr(signal, 'SIGKILL', signal.SIGTERM)
                os.kill(p.pid, sig)
            except OSError:
                pass


def _default_backend():
    return 'gl' if 'win' not in sys.platform else 'matplotlib'


@defaults('backend', sid_ignore=('backend',))
def visualize_patch(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, backend=_default_backend(), block=False, columns=2):
    """Visualize scalar data associated to a two-dimensional |Grid| as a patch plot.

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
        If `True`, use separate colorbars for each subplot.
    rescale_colorbars
        If `True`, rescale colorbars to data in each frame.
    backend
        Plot backend to use ('gl' or 'matplotlib').
    block
        If `True`, block execution until the plot window is closed.
    columns
        The number of columns in the visualizer GUI in case multiple plots are displayed
        at the same time.
    """
    if not config.HAVE_PYSIDE:
        raise PySideMissing()

    assert backend in {'gl', 'matplotlib'}

    if backend == 'gl':
        if not config.HAVE_GL:
            logger = getLogger('pymor.gui.qt.visualize_patch')
            logger.warn('import of PyOpenGL failed, falling back to matplotlib; rendering will be slow')
            backend = 'matplotlib'
        elif not config.HAVE_QTOPENGL:
            logger = getLogger('pymor.gui.qt.visualize_patch')
            logger.warn('import of PySide.QtOpenGL failed, falling back to matplotlib; rendering will be slow')
            backend = 'matplotlib'
        if backend == 'matplotlib' and not config.HAVE_MATPLOTLIB:
            raise ImportError('cannot visualize: import of matplotlib failed')
    else:
        if not config.HAVE_MATPLOTLIB:
            raise ImportError('cannot visualize: import of matplotlib failed')

    # TODO extract class
    class MainWindow(PlotMainWindow):
        def __init__(self, grid, U, bounding_box, codim, title, legend, separate_colorbars, rescale_colorbars, backend):

            assert isinstance(U, VectorArrayInterface) and hasattr(U, 'data') \
                or (isinstance(U, tuple) and all(isinstance(u, VectorArrayInterface) and hasattr(u, 'data') for u in U)
                    and all(len(u) == len(U[0]) for u in U))
            U = (U.data.astype(np.float64, copy=False),) if hasattr(U, 'data') else \
                tuple(u.data.astype(np.float64, copy=False) for u in U)
            if isinstance(legend, str):
                legend = (legend,)
            assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)
            if backend == 'gl':
                widget = GLPatchWidget
                cbar_widget = ColorBarWidget
            else:
                widget = MatplotlibPatchWidget
                cbar_widget = None
                if not separate_colorbars and len(U) > 1:
                    l = getLogger('pymor.gui.qt.visualize_patch')
                    l.warn('separate_colorbars=False not supported for matplotlib backend')
                separate_colorbars = True

            class PlotWidget(QWidget):
                def __init__(self):
                    super().__init__()
                    if separate_colorbars:
                        if rescale_colorbars:
                            self.vmins = tuple(np.min(u[0]) for u in U)
                            self.vmaxs = tuple(np.max(u[0]) for u in U)
                        else:
                            self.vmins = tuple(np.min(u) for u in U)
                            self.vmaxs = tuple(np.max(u) for u in U)
                    else:
                        if rescale_colorbars:
                            self.vmins = (min(np.min(u[0]) for u in U),) * len(U)
                            self.vmaxs = (max(np.max(u[0]) for u in U),) * len(U)
                        else:
                            self.vmins = (min(np.min(u) for u in U),) * len(U)
                            self.vmaxs = (max(np.max(u) for u in U),) * len(U)

                    layout = QHBoxLayout()
                    plot_layout = QGridLayout()
                    self.colorbarwidgets = [cbar_widget(self, vmin=vmin, vmax=vmax) if cbar_widget else None
                                            for vmin, vmax in zip(self.vmins, self.vmaxs)]
                    plots = [widget(self, grid, vmin=vmin, vmax=vmax, bounding_box=bounding_box, codim=codim)
                             for vmin, vmax in zip(self.vmins, self.vmaxs)]
                    if legend:
                        for i, plot, colorbar, l in zip(range(len(plots)), plots, self.colorbarwidgets, legend):
                            subplot_layout = QVBoxLayout()
                            caption = QLabel(l)
                            caption.setAlignment(Qt.AlignHCenter)
                            subplot_layout.addWidget(caption)
                            if not separate_colorbars or backend == 'matplotlib':
                                subplot_layout.addWidget(plot)
                            else:
                                hlayout = QHBoxLayout()
                                hlayout.addWidget(plot)
                                if colorbar:
                                    hlayout.addWidget(colorbar)
                                subplot_layout.addLayout(hlayout)
                            plot_layout.addLayout(subplot_layout, int(i/columns), (i % columns), 1, 1)
                    else:
                        for i, plot, colorbar in zip(range(len(plots)), plots, self.colorbarwidgets):
                            if not separate_colorbars or backend == 'matplotlib':
                                plot_layout.addWidget(plot, int(i/columns), (i % columns), 1, 1)
                            else:
                                hlayout = QHBoxLayout()
                                hlayout.addWidget(plot)
                                if colorbar:
                                    hlayout.addWidget(colorbar)
                                plot_layout.addLayout(hlayout, int(i/columns), (i % columns), 1, 1)
                    layout.addLayout(plot_layout)
                    if not separate_colorbars:
                        layout.addWidget(self.colorbarwidgets[0])
                        for w in self.colorbarwidgets[1:]:
                            w.setVisible(False)
                    self.setLayout(layout)
                    self.plots = plots

                def set(self, U, ind):
                    if rescale_colorbars:
                        if separate_colorbars:
                            self.vmins = tuple(np.min(u[ind]) for u in U)
                            self.vmaxs = tuple(np.max(u[ind]) for u in U)
                        else:
                            self.vmins = (min(np.min(u[ind]) for u in U),) * len(U)
                            self.vmaxs = (max(np.max(u[ind]) for u in U),) * len(U)

                    for u, plot, colorbar, vmin, vmax in zip(U, self.plots, self.colorbarwidgets, self.vmins,
                                                              self.vmaxs):
                        plot.set(u[ind], vmin=vmin, vmax=vmax)
                        if colorbar:
                            colorbar.set(vmin=vmin, vmax=vmax)

            super().__init__(U, PlotWidget(), title=title, length=len(U[0]))
            self.grid = grid
            self.codim = codim

        def save(self):
            if not config.HAVE_PYVTK:
                msg = QMessageBox(QMessageBox.Critical, 'Error', 'VTK output disabled. Pleas install pyvtk.')
                msg.exec_()
                return
            filename = QFileDialog.getSaveFileName(self, 'Save as vtk file')[0]
            base_name = filename.split('.vtu')[0].split('.vtk')[0].split('.pvd')[0]
            if base_name:
                if len(self.U) == 1:
                    write_vtk(self.grid, NumpyVectorSpace.make_array(self.U[0]), base_name, codim=self.codim)
                else:
                    for i, u in enumerate(self.U):
                        write_vtk(self.grid, NumpyVectorSpace.make_array(u), '{}-{}'.format(base_name, i),
                                  codim=self.codim)

    _launch_qt_app(lambda: MainWindow(grid, U, bounding_box, codim, title=title, legend=legend,
                                      separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                                      backend=backend),
                   block)


def visualize_matplotlib_1d(grid, U, codim=1, title=None, legend=None, separate_plots=False, block=False):
    """Visualize scalar data associated to a one-dimensional |Grid| as a plot.

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
    separate_plots
        If `True`, use subplots to visualize multiple |VectorArrays|.
    block
        If `True`, block execution until the plot window is closed.
    """
    if not config.HAVE_PYSIDE:
        raise PySideMissing()
    if not config.HAVE_MATPLOTLIB:
        raise ImportError('cannot visualize: import of matplotlib failed')

    class MainWindow(PlotMainWindow):
        def __init__(self, grid, U, codim, title, legend, separate_plots):
            assert isinstance(U, VectorArrayInterface) and hasattr(U, 'data') \
                or (isinstance(U, tuple) and all(isinstance(u, VectorArrayInterface) and hasattr(u, 'data') for u in U)
                    and all(len(u) == len(U[0]) for u in U))
            U = (U.data,) if hasattr(U, 'data') else tuple(u.data for u in U)
            if isinstance(legend, str):
                legend = (legend,)
            assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)

            plot_widget = Matplotlib1DWidget(None, grid, count=len(U), vmin=[np.min(u) for u in U],
                                             vmax=[np.max(u) for u in U], legend=legend, codim=codim,
                                             separate_plots=separate_plots)
            super().__init__(U, plot_widget, title=title, length=len(U[0]))
            self.grid = grid

    _launch_qt_app(lambda: MainWindow(grid, U, codim, title=title, legend=legend, separate_plots=separate_plots), block)


class PatchVisualizer(BasicInterface):
    """Visualize scalar data associated to a two-dimensional |Grid| as a patch plot.

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
        Plot backend to use ('gl' or 'matplotlib').
    block
        If `True`, block execution until the plot window is closed.
    """

    def __init__(self, grid, bounding_box=([0, 0], [1, 1]), codim=2, backend=None, block=False):
        assert grid.reference_element in (triangle, square)
        assert grid.dim == 2
        assert codim in (0, 2)
        self.grid = grid
        self.bounding_box = bounding_box
        self.codim = codim
        self.backend = backend
        self.block = block

    def visualize(self, U, discretization, title=None, legend=None, separate_colorbars=False,
                  rescale_colorbars=False, block=None, filename=None, columns=2):
        """Visualize the provided data.

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
            If `True`, use separate colorbars for each subplot.
        rescale_colorbars
            If `True`, rescale colorbars to data in each frame.
        block
            If `True`, block execution until the plot window is closed. If `None`, use the
            default provided during instantiation.
        filename
            If specified, write the data to a VTK-file using
            :func:`pymor.tools.vtkio.write_vtk` instead of displaying it.
        columns
            The number of columns in the visualizer GUI in case multiple plots are displayed
            at the same time.
        """
        assert isinstance(U, VectorArrayInterface) and hasattr(U, 'data') \
            or (isinstance(U, tuple) and all(isinstance(u, VectorArrayInterface) and hasattr(u, 'data') for u in U)
                and all(len(u) == len(U[0]) for u in U))
        if filename:
            if not isinstance(U, tuple):
                write_vtk(self.grid, U, filename, codim=self.codim)
            else:
                for i, u in enumerate(U):
                    write_vtk(self.grid, u, '{}-{}'.format(filename, i), codim=self.codim)
        else:
            block = self.block if block is None else block
            visualize_patch(self.grid, U, bounding_box=self.bounding_box, codim=self.codim, title=title,
                            legend=legend, separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                            backend=self.backend, block=block, columns=columns)


class Matplotlib1DVisualizer(BasicInterface):
    """Visualize scalar data associated to a one-dimensional |Grid| as a plot.

    The grid's |ReferenceElement| must be the line. The data can either
    be attached to the subintervals or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    codim
        The codimension of the entities the data in `U` is attached to (either 0 or 1).
    block
        If `True`, block execution until the plot window is closed.
    """

    def __init__(self, grid, codim=1, block=False):
        assert isinstance(grid, OnedGrid)
        assert codim in (0, 1)
        self.grid = grid
        self.codim = codim
        self.block = block

    def visualize(self, U, discretization, title=None, legend=None, block=None):
        """Visualize the provided data.

        Parameters
        ----------
        U
            |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
            as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
            provided, in which case several plots are made into the same axes. The
            lengths of all arrays have to agree.
        discretization
            Filled in by :meth:`pymor.discretizations.DiscretizationBase.visualize` (ignored).
        title
            Title of the plot.
        legend
            Description of the data that is plotted. Most useful if `U` is a tuple in which
            case `legend` has to be a tuple of strings of the same length.
        block
            If `True`, block execution until the plot window is closed. If `None`, use the
            default provided during instantiation.
        """
        block = self.block if block is None else block
        visualize_matplotlib_1d(self.grid, U, codim=self.codim, title=title, legend=legend, block=block)
