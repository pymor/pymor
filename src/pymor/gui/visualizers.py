# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)


from pymor.core.interfaces import BasicInterface
from pymor.grids.oned import OnedGrid
from pymor.grids.referenceelements import triangle, square
from pymor.tools.vtkio import write_vtk
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.core.config import is_jupyter


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
        Plot backend to use ('gl', 'matplotlib', 'jupyter').
    block
        If `True`, block execution until the plot window is closed.
    """

    def __init__(self, grid, bounding_box=([0, 0], [1, 1]), codim=2, backend=None, block=False):
        assert grid.reference_element in (triangle, square)
        assert grid.dim == 2
        assert codim in (0, 2)
        backend = backend or ('jupyter' if is_jupyter() else None)
        self.__auto_init(locals())
            # TODO this check is not currently working as expected
            from IPython import get_ipython
            if type(get_ipython()).__module__.startswith('ipykernel.'):
                    backend = 'jupyter'

    def visualize(self, U, m, title=None, legend=None, separate_colorbars=False,
                  rescale_colorbars=False, block=None, filename=None, columns=2):
        """Visualize the provided data.

        Parameters
        ----------
        U
            |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
            as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
            provided, in which case a subplot is created for each entry of the tuple. The
            lengths of all arrays have to agree.
        m
            Filled in by :meth:`pymor.models.ModelBase.visualize` (ignored).
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
        assert isinstance(U, VectorArrayInterface) \
            or (isinstance(U, tuple)
                and all(isinstance(u, VectorArrayInterface) for u in U)
                and all(len(u) == len(U[0]) for u in U))
        if filename:
            if not isinstance(U, tuple):
                write_vtk(self.grid, U, filename, codim=self.codim)
            else:
                for i, u in enumerate(U):
                    write_vtk(self.grid, u, f'{filename}-{i}', codim=self.codim)
        else:
            if self.backend == 'jupyter':
                from pymor.gui.jupyter import visualize_k3d_vtk
                return visualize_k3d_vtk(self.grid, U, bounding_box=self.bounding_box, codim=self.codim, title=title,
                                  legend=legend, separate_colorbars=separate_colorbars,
                                  rescale_colorbars=rescale_colorbars, columns=columns)
            else:
                block = self.block if block is None else block
                from pymor.gui.qt import visualize_patch
                return visualize_patch(self.grid, U, bounding_box=self.bounding_box, codim=self.codim, title=title,
                                legend=legend, separate_colorbars=separate_colorbars,
                                rescale_colorbars=rescale_colorbars, backend=self.backend, block=block,
                                columns=columns)


class OnedVisualizer(BasicInterface):
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
        self.__auto_init(locals())

    def visualize(self, U, m, title=None, legend=None, block=None):
        """Visualize the provided data.

        Parameters
        ----------
        U
            |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
            as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
            provided, in which case several plots are made into the same axes. The
            lengths of all arrays have to agree.
        m
            Filled in by :meth:`pymor.models.ModelBase.visualize` (ignored).
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
        from pymor.gui.qt import visualize_matplotlib_1d
        visualize_matplotlib_1d(self.grid, U, codim=self.codim, title=title, legend=legend, block=block)
