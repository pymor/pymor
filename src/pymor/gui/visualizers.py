from pymor.core.interfaces import BasicInterface
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.tools.vtkio import write_vtk

from pymor.grids.oned import OnedGrid
from pymor.grids.referenceelements import triangle, square

from pymor.gui.qt.qt import visualize_matplotlib_1d as visualize_matplotlib_1d_qt, visualize_patch as visualize_patch_qt
from pymor.gui.kivy_frontend.kivy_frontend import visualize_oned as visualize_oned_kivy, \
    visualize_patch as visualize_patch_kivy


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

    def __init__(self, grid, codim=1, frontend='kivy', block=False):
        assert isinstance(grid, OnedGrid)
        assert codim in (0, 1)
        assert frontend in ("qt", "kivy")
        self.grid = grid
        self.codim = codim
        self.block = block
        self.frontend = frontend

    def visualize(self, U, discretization, title=None, legend=None, separate_plots=False, block=None):
        """Visualize the provided data.

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
        """
        assert isinstance(U, VectorArrayInterface) and hasattr(U, 'data') \
            or (isinstance(U, tuple) and all(isinstance(u, VectorArrayInterface) and hasattr(u, 'data') for u in U)
                and all(len(u) == len(U[0]) for u in U))
        block = self.block if block is None else block
        if self.frontend == "qt":
            visualize = visualize_matplotlib_1d_qt
        else:
            visualize = visualize_oned_kivy

        visualize(self.grid, U, codim=self.codim, title=title, legend=legend, separate_plots=separate_plots,
                  block=block)


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
        If `True` block execution until the plot window is closed.
    """

    def __init__(self, grid, bounding_box=([0, 0], [1, 1]), codim=2, frontend='kivy', backend=None, block=False):
        assert grid.reference_element in (triangle, square)
        assert grid.dim_outer == 2
        assert codim in (0, 2)
        assert frontend in ("qt", "kivy")
        self.grid = grid
        self.bounding_box = bounding_box
        self.codim = codim
        self.frontend = frontend
        self.backend = backend
        self.block = block

    def visualize(self, U, discretization, title=None, legend=None, separate_colorbars=False,
                  rescale_colorbars=False, block=None, filename=None, columns=2, backend=None):
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
        assert backend is None or backend in ['matplotlib', 'gl']
        if filename:
            if not isinstance(U, tuple):
                write_vtk(self.grid, U, filename, codim=self.codim)
            else:
                for i, u in enumerate(U):
                    write_vtk(self.grid, u, '{}-{}'.format(filename, i), codim=self.codim)
        else:
            block = self.block if block is None else block

            if self.frontend == "qt":
                visualize = visualize_patch_qt
            else:
                visualize = visualize_patch_kivy
            backend = backend if backend is not None else self.backend
            visualize(self.grid, U, bounding_box=self.bounding_box, codim=self.codim, title=title,
                            legend=legend, separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                            backend=backend, block=block, columns=columns)