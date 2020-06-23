# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

""" This module provides a widgets for displaying plots of
scalar data assigned to one- and two-dimensional grids using
:mod:`matplotlib`. This widget is not intended to be used directly.
"""

import numpy as np

from pymor.core.config import config
from pymor.discretizers.builtin.grids.constructions import flatten_grid
from pymor.discretizers.builtin.grids.referenceelements import triangle, square


class MatplotlibPatchAxes:

    def __init__(self, figure, grid, bounding_box=None, vmin=None, vmax=None, codim=2,
                 colorbar=True):
        assert grid.reference_element in (triangle, square)
        assert grid.dim == 2
        assert codim in (0, 2)

        subentities, coordinates, entity_map = flatten_grid(grid)
        self.subentities = subentities if grid.reference_element is triangle \
            else np.vstack((subentities[:, 0:3], subentities[:, [2, 3, 0]]))
        self.coordinates = coordinates
        self.entity_map = entity_map
        self.reference_element = grid.reference_element
        self.vmin = vmin
        self.vmax = vmax
        self.codim = codim
        a = figure.gca()
        if self.codim == 2:
            self.p = a.tripcolor(self.coordinates[:, 0], self.coordinates[:, 1], self.subentities,
                                 np.zeros(len(self.coordinates)),
                                 vmin=self.vmin, vmax=self.vmax, shading='gouraud')
        else:
            self.p = a.tripcolor(self.coordinates[:, 0], self.coordinates[:, 1], self.subentities,
                                 facecolors=np.zeros(len(self.subentities)),
                                 vmin=self.vmin, vmax=self.vmax, shading='flat')
        if colorbar:
            figure.colorbar(self.p, ax=a)

    def set(self, U, vmin=None, vmax=None):
        self.vmin = self.vmin if vmin is None else vmin
        self.vmax = self.vmax if vmax is None else vmax
        U = np.array(U)
        p = self.p
        if self.codim == 2:
            p.set_array(U)
        elif self.reference_element is triangle:
            p.set_array(U)
        else:
            p.set_array(np.tile(U, 2))
        p.set_clim(self.vmin, self.vmax)


class Matplotlib1DAxes:

    def __init__(self, figure, grid, bounding_box=None, vmin=None, vmax=None, codim=1,
                 colorbar=True):
        assert isinstance(grid, OnedGrid)
        assert codim in (0, 1)

        self.codim = codim
        self.grid = grid

        centers = grid.centers(1)
        if grid.identify_left_right:
            centers = np.concatenate((centers, [[grid.domain[1]]]), axis=0)
            self.periodic = True
        else:
            self.periodic = False
        if codim == 1:
            xs = centers
        else:
            xs = np.repeat(centers, 2)[1:-1]

        a = figure.gca()
        self.lines, = a.plot(xs, np.zeros_like(xs))

        # TODO
        import matplotlib.pyplot as plt
        self.p = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"), norm=plt.Normalize(vmin=vmin or 0, vmax=vmax or 1))

        if colorbar:
            figure.colorbar(self.p, ax=a)


    def set(self, U, vmin=None, vmax=None):
        self.vmin = self.vmin if vmin is None else vmin
        self.vmax = self.vmax if vmax is None else vmax
        u = np.array(U)
        if self.codim == 1:
            if self.periodic:
                self.lines.set_ydata(np.concatenate((u, [u[0]])))
            else:
                self.lines.set_ydata(u)
        else:
            self.lines.set_ydata(np.repeat(u, 2))
        self.p.set_clim(self.vmin, self.vmax)


if config.HAVE_QT and config.HAVE_MATPLOTLIB:
    from Qt.QtWidgets import QSizePolicy

    import Qt
    if Qt.__qt_version__[0] == '4':
        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    elif Qt.__qt_version__[0] == '5':
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    else:
        raise NotImplementedError

    from matplotlib.figure import Figure

    from pymor.discretizers.builtin.grids.oned import OnedGrid

    # noinspection PyShadowingNames
    class Matplotlib1DWidget(FigureCanvas):

        def __init__(self, parent, grid, count, vmin=None, vmax=None, legend=None, codim=1,
                     separate_plots=False, dpi=100):
            assert isinstance(grid, OnedGrid)
            assert codim in (0, 1)

            figure = Figure(dpi=dpi)
            if not separate_plots:
                axes = figure.gca()
            self.codim = codim
            lines = ()
            centers = grid.centers(1)
            if grid.identify_left_right:
                centers = np.concatenate((centers, [[grid.domain[1]]]), axis=0)
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

            self.setParent(parent)
            self.setMinimumSize(300, 300)
            self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

            self.patch_axes = MatplotlibPatchAxes(self.figure, grid, bounding_box, vmin, vmax, codim)

        def set(self, U, vmin=None, vmax=None):
            self.patch_axes.set(U, vmin, vmax)
            self.draw()

else:

    class Matplotlib1DWidget:
        pass

    class MatplotlibPatchWidget:
        pass
