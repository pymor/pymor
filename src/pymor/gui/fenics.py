# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

try:
    import dolfin as df
    HAVE_FENICS = True
except ImportError:
    HAVE_FENICS = False

if HAVE_FENICS:
    import numpy as np

    from pymor.core.interfaces import BasicInterface
    from pymor.vectorarrays.fenics import FenicsVectorSpace

    class FenicsVisualizer(BasicInterface):
        """Visualize a FEniCS grid function.

        Parameters
        ----------
        function_space
            The FEniCS FunctionSpace for which we want to visualize dof vectors.
        """

        def __init__(self, function_space):
            self.function_space = function_space
            self.space = FenicsVectorSpace(function_space)

        def visualize(self, U, discretization, title='', legend=None, filename=None, block=True,
                      separate_colorbars=True):
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
            separate_colorbars
                If `True`, use separate colorbars for each subplot.
            block
                If `True`, block execution until the plot window is closed
                (non-blocking execution is currently unsupported).
            """
            if not block:
                raise NotImplementedError
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

                if not separate_colorbars:
                    vmin = np.inf
                    vmax = -np.inf
                    for u in U:
                        vec = u._list[0].impl
                        vmin = min(vmin, vec.min())
                        vmax = max(vmax, vec.max())

                for i, u in enumerate(U):
                    function = df.Function(self.function_space)
                    function.vector()[:] = u._list[0].impl
                    if legend:
                        tit = title + ' -- ' if title else ''
                        tit += legend[i]
                    else:
                        tit = title
                    if separate_colorbars:
                        df.plot(function, interactive=False, title=tit)
                    else:
                        df.plot(function, interactive=False, title=tit,
                                range_min=vmin, range_max=vmax)
                df.interactive()
