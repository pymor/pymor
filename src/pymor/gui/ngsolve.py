# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

if config.HAVE_NGSOLVE:
    import ngsolve as ngs
    from pymor.core.interfaces import ImmutableInterface
    from pymor.vectorarrays.interfaces import VectorArrayInterface
    from pymor.vectorarrays.ngsolve import NGSolveVectorSpace

    class NGSolveVisualizer(ImmutableInterface):
        """Visualize an NGSolve grid function."""

        def __init__(self, mesh, fespace):
            self.mesh = mesh
            self.fespace = fespace
            self.space = NGSolveVectorSpace(fespace.ndof)

        def visualize(self, U, discretization, legend=None, separate_colorbars=True, block=True):
            """Visualize the provided data."""
            if isinstance(U, VectorArrayInterface):
                U = (U,)
            assert all(u in self.space for u in U)
            if any(len(u) != 1 for u in U):
                raise NotImplementedError

            if legend is None:
                legend = ['VectorArray{}'.format(i) for i in range(len(U))]
            if isinstance(legend, str):
                legend = [legend]
            assert len(legend) == len(U)
            legend = [l.replace(' ', '_') for l in legend]  # NGSolve GUI will fail otherwise

            if not separate_colorbars:
                raise NotImplementedError

            grid_functions = []
            for u in U:
                gf = ngs.GridFunction(self.fespace)
                gf.vec.data = u._list[0].impl
                grid_functions.append(gf)

            for gf, name in zip(grid_functions, legend):
                ngs.Draw(gf, self.mesh, name=name)
