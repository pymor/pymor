try:
    from kivy.app import App
    from kivy.uix.widget import Widget
    from kivy.uix.button import Button
    HAVE_KIVY = True
except ImportError:
    HAVE_KIVY = False

try:
    import matplotlib
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

HAVE_ALL = HAVE_KIVY and HAVE_MATPLOTLIB

from pymor.grids.oned import OnedGrid
from pymor.grids.referenceelements import triangle, square
from pymor.grids.constructions import flatten_grid
import numpy as np
from matplotlib.figure import Figure

if HAVE_ALL:

    # as soon as a FigureCanvas is imported a kivy window is instanciated by the matplotlib bindings.
    # In order to be able to show multiple windows, this class has to be parsed in a new process.
    # This is done by wrapping it in a function

    def getMatplotlibOnedWidget(parent, grid, count, vmin=None, vmax=None, legend=None, codim=1,
                     separate_plots=False, dpi=100):
        from pymor.gui.kivy_matplotlib import FigureCanvasKivyAgg

        class MatplotlibOnedWidget(FigureCanvasKivyAgg):

            def __init__(self, parent, grid, count, vmin=None, vmax=None, legend=None, codim=1,
                     separate_plots=False, dpi=100):
                assert isinstance(grid, OnedGrid)
                assert codim in (0, 1)

                figure = Figure(dpi=dpi)
                if not separate_plots:
                    axes = figure.gca()
                    axes.hold(True)
                self.codim = codim
                lines = tuple()
                centers = grid.centers(1)
                if grid._identify_left_right:
                    centers = np.concatenate((centers, [[grid._domain[1]]]), axis=0)
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

                super(MatplotlibOnedWidget, self).__init__(figure)

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

        return MatplotlibOnedWidget(parent=parent, grid=grid, count=count, vmin=vmin, vmax=vmax, legend=legend,
                                    codim=codim, separate_plots=separate_plots, dpi=dpi)

    def getMatplotlibPatchWidget(parent, grid, bounding_box=None, vmin=None, vmax=None, codim=2, dpi=100):
        from pymor.gui.kivy_matplotlib import FigureCanvasKivyAgg

        class MatplotlibPatchWidget(FigureCanvasKivyAgg):

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

            def set(self, U, vmin=None, vmax=None):
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

        return MatplotlibPatchWidget(parent=parent, grid=grid, bounding_box=bounding_box, vmin=vmin, vmax=vmax,
                                     codim=codim, dpi=dpi)
