# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


from pymor.core.base import ImmutableObject
from pymor.core.defaults import defaults
from pymor.core.config import is_jupyter
from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymor.discretizers.builtin.grids.referenceelements import triangle, square
from pymor.vectorarrays.interface import VectorArray


class PatchVisualizer(ImmutableObject):
    """Visualize scalar data associated to a two-dimensional |Grid| as a patch plot.

    The grid's |ReferenceElement| must be the triangle or square. The data can either
    be attached to the faces or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    codim
        The codimension of the entities the data in `U` is attached to (either 0 or 2).
    bounding_box
        A bounding box in which the grid is contained.
    backend
        Plot backend to use ('jupyter_or_gl', 'jupyter', 'gl', 'matplotlib').
    block
        If `True`, block execution until the plot window is closed.
    """

    @defaults('backend')
    def __init__(self, grid, codim=2, bounding_box=None, backend='jupyter_or_gl', block=False):
        assert grid.reference_element in (triangle, square)
        assert grid.dim == 2
        assert codim in (0, 2)
        assert backend in {'jupyter_or_gl', 'jupyter', 'gl', 'matplotlib'}
        if backend == 'jupyter_or_gl':
            backend = 'jupyter' if is_jupyter() else 'gl'
        if bounding_box is None:
            bounding_box = grid.bounding_box()
        self.__auto_init(locals())

    def visualize(self, U, title=None, legend=None, separate_colorbars=False,
                  rescale_colorbars=False, block=None, filename=None, columns=2):
        """Visualize the provided data.

        Parameters
        ----------
        U
            |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
            as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
            provided, in which case a subplot is created for each entry of the tuple. The
            lengths of all arrays have to agree.
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
            :func:`~pymor.discretizers.builtin.grids.vtkio.write_vtk` instead of displaying it.
        columns
            The number of columns in the visualizer GUI in case multiple plots are displayed
            at the same time.
        """
        assert isinstance(U, VectorArray) \
            or (isinstance(U, tuple)
                and all(isinstance(u, VectorArray) for u in U)
                and all(len(u) == len(U[0]) for u in U))
        if filename:
            from pymor.discretizers.builtin.grids.vtkio import write_vtk
            if not isinstance(U, tuple):
                write_vtk(self.grid, U, filename, codim=self.codim)
            else:
                for i, u in enumerate(U):
                    write_vtk(self.grid, u, f'{filename}-{i}', codim=self.codim)
        else:
            if self.backend == 'jupyter':
                from pymor.discretizers.builtin.gui.jupyter import get_visualizer
                return get_visualizer()(self.grid, U, bounding_box=self.bounding_box, codim=self.codim, title=title,
                                        legend=legend, separate_colorbars=separate_colorbars,
                                        rescale_colorbars=rescale_colorbars, columns=columns)
            else:
                block = self.block if block is None else block
                from pymor.discretizers.builtin.gui.qt import visualize_patch
                return visualize_patch(self.grid, U, bounding_box=self.bounding_box, codim=self.codim, title=title,
                                       legend=legend, separate_colorbars=separate_colorbars,
                                       rescale_colorbars=rescale_colorbars, backend=self.backend, block=block,
                                       columns=columns)


class OnedVisualizer(ImmutableObject):
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
    backend
        Plot backend to use ('jupyter_or_matplotlib', 'jupyter', 'matplotlib').
    """

    @defaults('backend')
    def __init__(self, grid, codim=1, block=False, backend='jupyter_or_matplotlib'):
        assert isinstance(grid, OnedGrid)
        assert codim in (0, 1)
        assert backend in {'jupyter_or_matplotlib', 'jupyter', 'matplotlib'}
        if backend == 'jupyter_or_matplotlib':
            backend = 'jupyter' if is_jupyter() else 'matplotlib'
        self.__auto_init(locals())

    def visualize(self, U, title=None, legend=None, separate_plots=True,
                  separate_axes=False, block=None, filename=None, columns=2):
        """Visualize the provided data.

        Parameters
        ----------
        U
            |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
            as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
            provided, in which case several plots are made into the same axes. The
            lengths of all arrays have to agree.
        title
            Title of the plot.
        legend
            Description of the data that is plotted. Most useful if `U` is a tuple in which
            case `legend` has to be a tuple of strings of the same length.
        block
            If `True`, block execution until the plot window is closed. If `None`, use the
            default provided during instantiation.
        """
        if filename is not None:
            raise NotImplementedError

        if self.backend == 'jupyter':
            from pymor.discretizers.builtin.gui.jupyter.matplotlib import visualize_matplotlib_1d
            return visualize_matplotlib_1d(self.grid, U, codim=self.codim, title=title, legend=legend,
                                           separate_plots=separate_plots, separate_axes=separate_axes, columns=columns)
        else:
            block = self.block if block is None else block
            from pymor.discretizers.builtin.gui.qt import visualize_matplotlib_1d
            return visualize_matplotlib_1d(self.grid, U, codim=self.codim, title=title, legend=legend, block=block)
