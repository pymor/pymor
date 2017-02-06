# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_DUNEGDT:

    from pymor.core.interfaces import ImmutableInterface

    class DuneGDTVisualizer(ImmutableInterface):
        """Visualize a dune-gdt grid function.

        Parameters
        ----------
        space
            The dune-gdt space for which we want to visualize DOF vectors.
        """

        def __init__(self, space):
            self.space = space

        def visualize(self, U, discretization, filename=None, legend=None):
            assert not isinstance(U, tuple)  # only single VectorArray supported
            assert len(U) == 1
            from dune.gdt import make_discrete_function
            make_discrete_function(self.space, U._list[0].impl, legend or 'out').visualize(filename)
