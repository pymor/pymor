# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Visualization of grid data using Qt.

This module provides a few methods and classes for visualizing data
associated to grids. We use the `Qt <https://www.qt-project.org>`_ widget
toolkit for the GUI.
"""
from pymor.core.config import config
from pymor.discretizers.builtin.grids.oned import OnedGrid

config.require('QT')

import math as m
import subprocess
import sys
from tempfile import NamedTemporaryFile

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QAction,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLCDNumber,
    QMessageBox,
    QSlider,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.core.pickle import dump
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


@defaults('method')
def background_visualization_method(method='ipython_if_possible'):
    assert method in ('ipython', 'ipython_if_possible', 'pymor-vis')

    if getattr(sys, '_called_from_test', False):
        return 'ipython'
    elif method == 'ipython_if_possible':
        try:
            from IPython import get_ipython
            ip = get_ipython()
        except ImportError:
            ip = None
        if ip is None:
            return 'pymor-vis'
        else:
            return 'ipython'
    else:
        return method


def _launch_qt_app(main_window_factory, block):
    """Wrapper to display plot in a separate process."""
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    if getattr(sys, '_called_from_test', False):
        QTimer.singleShot(1200, app.quit)
        block = True

    if not block:
        try:
            from IPython import get_ipython
            ip = get_ipython()
        except ImportError:
            ip = None
        if ip is None:
            logger = getLogger('pymor.discretizers.builtin.gui.qt')
            logger.warning('Not running within IPython. Falling back to blocking visualization.')
            block = True
        else:
            ip.run_line_magic('gui', 'qt')

    main_window = main_window_factory()
    main_window.show()

    if block:
        app.exec()
    else:
        global _qt_app
        _qt_app = app                 # deleting the app ref somehow closes the window
        _qt_windows.add(main_window)  # need to keep ref to keep window alive


class PlotMainWindow(QWidget):
    """Base class for plot main windows."""

    def __init__(self, U, vmins, vmaxs, plot, length=1, title=None, save_action=None):
        super().__init__()

        self.U, self.vmins, self.vmaxs, self.plot, self.length, self.save_action \
            = U, vmins, vmaxs, plot, length, save_action

        layout = QVBoxLayout()

        if title:
            title = QLabel('<b>' + title + '</b>')
            title.setAlignment(Qt.AlignHCenter)
            layout.addWidget(title)
        layout.addWidget(plot)

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
            if save_action is not None:
                self.a_save = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), 'Save', self)
                toolbar.addAction(self.a_save)
                self.a_save.triggered.connect(lambda: self.save_action(self))
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

        elif save_action is not None:
            hlayout = QHBoxLayout()
            toolbar = QToolBar()
            self.a_save = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), 'Save', self)
            toolbar.addAction(self.a_save)
            hlayout.addWidget(toolbar)
            layout.addLayout(hlayout)
            self.a_save.triggered.connect(lambda: self.save_action(self))

        self.setLayout(layout)
        self.set(0)

    def set(self, ind):
        self.plot.set([u[ind] for u in self.U],
                      [vmin[ind] for vmin in self.vmins],
                      [vmax[ind] for vmax in self.vmaxs])

    def slider_changed(self, ind):
        self.set(ind)

    def speed_changed(self, val):
        self.timer.setInterval((100 - val) * 20)

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

    def closeEvent(self, event=None):
        """This is directly called from CI.

        Xvfb (sometimes) raises errors on interpreter shutdown
        when there are still 'live' MPL plot objects. This
        happens even if the referencing MainWindow was already destroyed
        """
        try:
            self.plot.p.close()
        except Exception:
            pass
        try:
            del self.plot.p
        except Exception:
            pass

        try:
            self.deleteLater()
            _qt_windows.remove(self)
        except KeyError:
            pass  # we should be in blocking mode ...
        if event is not None:
            event.accept()


_qt_app = None
_qt_windows = set()


def visualize_patch(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, backend='gl', block=False, columns=2):
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
    assert backend in {'gl', 'matplotlib'}
    assert isinstance(U, VectorArray) \
        or (isinstance(U, tuple) and all(isinstance(u, VectorArray) for u in U)
            and all(len(u) == len(U[0]) for u in U))
    if isinstance(legend, str):
        legend = (legend,)
    assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)

    if not block:
        if background_visualization_method() == 'pymor-vis':
            data = dict(dim=2,
                        grid=grid, U=U, bounding_box=bounding_box, codim=codim, title=title, legend=legend,
                        separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                        backend=backend, columns=columns)
            with NamedTemporaryFile(mode='wb', delete=False) as f:
                dump(data, f)
                filename = f.name
            subprocess.Popen(['python3', '-m', 'pymor.scripts.pymor_vis', '--delete', filename])
            return

    U = (U.to_numpy().astype(np.float64, copy=False),) if isinstance(U, VectorArray) else \
        tuple(u.to_numpy().astype(np.float64, copy=False) for u in U)

    if backend == 'gl':
        if not config.HAVE_GL:
            logger = getLogger('pymor.discretizers.builtin.gui.qt.visualize_patch')
            logger.warning('import of PyOpenGL failed, falling back to matplotlib; rendering will be slow')
            backend = 'matplotlib'
        elif not config.HAVE_QTOPENGL:
            logger = getLogger('pymor.discretizers.builtin.gui.qt.visualize_patch')
            logger.warning('import of Qt.QtOpenGL failed, falling back to matplotlib; rendering will be slow')
            backend = 'matplotlib'
        if backend == 'matplotlib' and not config.HAVE_MATPLOTLIB:
            raise ImportError('cannot visualize: import of matplotlib failed')
    else:
        if not config.HAVE_MATPLOTLIB:
            raise ImportError('cannot visualize: import of matplotlib failed')

    if backend == 'gl':
        from pymor.discretizers.builtin.gui.qt.gl import ColorBarWidget, GLPatchWidget
        widget = GLPatchWidget
        cbar_widget = ColorBarWidget
    else:
        from pymor.discretizers.builtin.gui.qt.matplotlib import MatplotlibPatchWidget
        widget = MatplotlibPatchWidget
        cbar_widget = None

    class PlotWidget(QWidget):
        def __init__(self):
            super().__init__()
            layout = QHBoxLayout()
            plot_layout = QGridLayout()
            self.colorbarwidgets = [cbar_widget(self, vmin=0., vmax=1.) if cbar_widget else None
                                    for _ in range(len(U))]
            plots = [widget(self, grid, bounding_box=bounding_box, codim=codim) for _ in range(len(U))]
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
            if not separate_colorbars and backend != 'matplotlib':
                layout.addWidget(self.colorbarwidgets[0])
                for w in self.colorbarwidgets[1:]:
                    w.setVisible(False)
            self.setLayout(layout)
            self.plots = plots

        def set(self, U, vmins, vmaxs):
            for u, plot, colorbar, vmin, vmax in zip(U, self.plots, self.colorbarwidgets, vmins, vmaxs):
                plot.set(u, vmin, vmax)
                if colorbar:
                    colorbar.set(vmin, vmax)

    def save_action(parent):
        if not config.HAVE_VTKIO:
            msg = QMessageBox(QMessageBox.Critical, 'Error', 'VTK output disabled. Please install pyvtk.')
            msg.exec_()
            return
        from pymor.discretizers.builtin.grids.vtkio import write_vtk
        filename = QFileDialog.getSaveFileName(parent, 'Save as vtk file')[0]
        base_name = filename.split('.vtu')[0].split('.vtk')[0].split('.pvd')[0]
        if base_name:
            if len(U) == 1:
                write_vtk(grid, NumpyVectorSpace.make_array(U[0]), base_name, codim=codim)
            else:
                for i, u in enumerate(U):
                    write_vtk(grid, NumpyVectorSpace.make_array(u), f'{base_name}-{i}',
                              codim=codim)

    from pymor.discretizers.builtin.gui.visualizers import _vmins_vmaxs
    vmins, vmaxs = _vmins_vmaxs(U, separate_colorbars, rescale_colorbars)

    _launch_qt_app(lambda: PlotMainWindow(U, vmins, vmaxs, PlotWidget(), length=len(U[0]), title=title,
                                          save_action=save_action),
                   block)


def visualize_matplotlib_1d(grid, U, codim=1, title=None, legend=None, separate_plots=False,
                            rescale_axes=False, columns=2, block=False):
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
    rescale_axes
        If `True`, rescale axes to data in each frame.
    columns
        Number of columns the subplots are organized in.
    block
        If `True`, block execution until the plot window is closed.
    """
    if not config.HAVE_MATPLOTLIB:
        raise ImportError('cannot visualize: import of matplotlib failed')

    from pymor.discretizers.builtin.gui.visualizers import _vmins_vmaxs

    assert isinstance(grid, OnedGrid)
    assert codim in (0, 1)

    assert isinstance(U, VectorArray) \
        or (isinstance(U, tuple)
            and all(isinstance(u, VectorArray) for u in U)
            and all(len(u) == len(U[0]) for u in U))
    if isinstance(legend, str):
        legend = (legend,)
    assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)

    if not block:
        if background_visualization_method() == 'pymor-vis':
            data = dict(dim=1,
                        grid=grid, U=U, codim=codim, title=title, legend=legend,
                        separate_plots=separate_plots)
            with NamedTemporaryFile(mode='wb', delete=False) as f:
                dump(data, f)
                filename = f.name
            subprocess.Popen(['python', '-m', 'pymor.scripts.pymor_vis', '--delete', filename])
            return

    U = (U.to_numpy(),) if isinstance(U, VectorArray) else tuple(u.to_numpy() for u in U)
    vmins, vmaxs = _vmins_vmaxs(U, separate_plots, rescale_axes)

    from pymor.discretizers.builtin.gui.qt.matplotlib import Matplotlib1DWidget
    plot_widget = Matplotlib1DWidget(U, None, grid, len(U), legend=legend, codim=codim,
                                     separate_plots=separate_plots, columns=columns)

    _launch_qt_app(lambda: PlotMainWindow(U, vmins, vmaxs, plot_widget, title=title, length=len(U[0])),
                   block)
