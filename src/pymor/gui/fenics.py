# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

try:
    import dolfin as df
    HAVE_FENICS = True
except ImportError:
    HAVE_FENICS = False

if HAVE_FENICS:
    from pymor.core.interfaces import BasicInterface
    from pymor.vectorarrays.fenics import FenicsVector
    from pymor.vectorarrays.list import ListVectorArray

    class FenicsVisualizer(BasicInterface):
        """Visualize a FEniCS grid function.

        Parameters
        ----------
        function_space
            The FEniCS FunctionSpace for which we want to visualize dof vectors.
        """

        def __init__(self, function_space):
            self.function_space = function_space
            self.space = ListVectorArray([FenicsVector(df.Function(function_space).vector())]).space

        def visualize(self, U, discretization, title='', legend=None, filename=None):
            """Visualize the provided data.

            Parameters
            ----------
            U
                |VectorArray| of the data to visualize (length must be 1). Alternatively,
                a tuple of |VectorArrays| which will be visualized in separate windows.
                If `filename` is specified, only one |VectorArray| may be provided which,
                however, is allowed to contain multipled vectors which will be interpreted
                as a time series.
            discretization
                Filled in :meth:`pymor.discretizations.DiscretizationBase.visualize` (ignored).
            title
                Title of the plot.
            legend
                Description of the data that is plotted. If `U` is a tuple of |VectorArrays|,
                `legend` has to be a tuple of the same length.
            filename
                If specified, write the data to that file. `filename` needs to have an extension
                supported by FEniCS (e.g. `.pvd`).
            """
            if filename:
                assert not isinstance(U, tuple)
                assert U in self.space
                f = df.File(filename)
                function = df.Function(self.function_space)
                if legend:
                    function.rename(legend, legend)
                for u in U._list:
                    function.vector()[:] = u.impl
                    f << function
            else:
                assert U in self.space and len(U) == 1 \
                    or (isinstance(U, tuple) and all(u in self.space for u in U) and all(len(u) == 1 for u in U))
                if not isinstance(U, tuple):
                    U = (U,)
                if isinstance(legend, str):
                    legend = (legend,)
                assert legend is None or len(legend) == len(U)

                for i, u in enumerate(U):
                    function = df.Function(self.function_space)
                    function.vector()[:] = u._list[0].impl
                    if legend:
                        tit = title + ' -- ' if title else ''
                        tit += legend[i]
                    else:
                        tit = title
                    df.plot(function, interactive=False, title=tit)
                df.interactive()
