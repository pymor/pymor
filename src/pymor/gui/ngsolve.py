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

    from pymor.core.interfaces import ImmutableInterface
    from pymor.core.pickle import dump

    NGSOLVE_VISUALIZE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ngsolve_visualize.py')

    class NGSolveVisualizer(ImmutableInterface):
        """Visualize a FEniCS grid function."""

        def __init__(self, fespace):
            self.fespace = fespace

        def visualize(self, U, discretization, block=True):
            """Visualize the provided data."""
            if not block:
                raise NotImplementedError
            # if filename:
            #     assert not isinstance(U, tuple)
            #     assert U in self.space
            #     f = df.File(filename)
            #     function = df.Function(self.function_space)
            #     if legend:
            #         function.rename(legend, legend)
            #     for u in U._list:
            #         function.vector()[:] = u.impl
            #         f << function
            # else:
            assert len(U) == 1
            with NamedTemporaryFile() as f:
                dump(self.fespace, f)
                dump(U._list[0].impl, f)
                os.system('NGSOLVE_VISUALIZE_FILE={} netgen {}'.format(f.name, NGSOLVE_VISUALIZE_PATH))
