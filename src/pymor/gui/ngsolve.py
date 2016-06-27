# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

try:
    import ngsolve
    HAVE_NGSOLVE = True
except ImportError:
    HAVE_NGSOLVE = False

if HAVE_NGSOLVE:
    import os
    from tempfile import NamedTemporaryFile
    from threading import Thread

    from pymor.core.interfaces import ImmutableInterface
    from pymor.core.pickle import dump
    from pymor.vectorarrays.interfaces import VectorArrayInterface
    from pymor.vectorarrays.ngsolve import NGSolveVectorSpace

    NGSOLVE_VISUALIZE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ngsolve_visualize.py')

    class NGSolveVisualizer(ImmutableInterface):
        """Visualize an NGSolve grid function."""

        def __init__(self, fespace):
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

            thread = NGSolveVisualizerThread((self.fespace, U, legend))
            thread.start()
            if block:
                thread.join()


    class NGSolveVisualizerThread(Thread):

        def __init__(self, data):
            super().__init__()
            self.data = data

        def run(self):
            with NamedTemporaryFile() as f:
                dump(self.data, f)
                del(self.data)
                f.flush()
                os.system('NGSOLVE_VISUALIZE_FILE={} netgen {}'.format(f.name, NGSOLVE_VISUALIZE_PATH))
