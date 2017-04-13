# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_DUNEGDT:

    from collections import OrderedDict
    import os
    import subprocess
    from tempfile import mkstemp

    from dune.gdt import make_const_discrete_function

    from pymor.core.interfaces import ImmutableInterface
    from pymor.bindings.dunext import DuneXTVector
    from pymor.grids.rect import RectGrid
    from pymor.gui.qt import PatchVisualizer
    from pymor.vectorarrays.list import NumpyVector, NumpyListVectorSpace, ListVectorArray


    class DuneGDTVisualizer(ImmutableInterface):
        """Visualize a dune-gdt discrete function using paraview.

        Parameters
        ----------
        space
            The dune-gdt space for which we want to visualize DOF vectors.
        """

        def __init__(self, space):
            self.space = space

        def visualize(self, U, discretization, title=None, legend=None, separate_colorbars=False,
                      rescale_colorbars=False, block=None, filename=None, columns=2):

            def visualize_single(vec, vec_name, f_name):
                discrete_function = make_const_discrete_function(self.space, vec, vec_name)
                discrete_function.visualize(f_name[:-4])

            suffix = '.vtp' if self.space.dimDomain == 1 else '.vtu'
            interactive = filename is None
            to_clean_up = []

            if isinstance(U, tuple):
                # we presume to have several vectors which should be visualized side by side
                assert all([isinstance(u, ListVectorArray) for u in U])
                assert all([len(u) == 1 for u in U])
                assert all([all([isinstance(v, DuneXTVector) for v in u._list]) for u in U])
                if legend is not None and len(legend) == len(U):
                    names = legend
                elif title is not None and len(title) == len(U):
                    names = title
                else:
                    names = list()
                    for ii in range(len(U)):
                        names.append('STATE_{}'.format(ii))
                data = OrderedDict()
                if interactive:
                    for ii in range(len(U)):
                        _, filename = mkstemp(suffix='_{}{}'.format(ii, suffix))
                        data[filename] = U[ii]._list[0]
                else:
                    if (filename.endswith('.vtp') or filename.endswith('.vtu')):
                        filename = filename[:-4]
                    assert len(filename) > 0
                    for ii in range(len(U)):
                        data['{}_{}{}'.format(filename, ii, suffix)] = U[ii]._list[0]
                for name, f_name in zip(names, data):
                    visualize_single(data[f_name].impl, name, f_name)
                if interactive:
                    _, pvd_filename = mkstemp(suffix='.pvd')
                    with open(pvd_filename, 'w') as pvd_file:
                        pvd_file.write('<?xml version="1.0"?>\n')
                        pvd_file.write('<VTKFile type="Collection" version="0.1">\n')
                        pvd_file.write('  <Collection>\n')
                        part = 0
                        for f_name in data:
                            pvd_file.write('    <DataSet timestep="0" group="" part="{}" file="{}"/>\n'.format(part,
                                                                                                               f_name))
                            part += 1
                        pvd_file.write('  </Collection>\n')
                        pvd_file.write('</VTKFile>\n')
                    subprocess.call(['paraview', pvd_filename])
                    for f_name in data:
                        to_clean_up.append(f_name)
                    to_clean_up.append(pvd_filename)
            else:
                assert isinstance(U, ListVectorArray)
                name = legend if legend else (title if title else 'STATE')
                if len(U) == 1:
                    # we presume we have a single vector to be visualized
                    U = U._list[0]
                    assert isinstance(U, DuneXTVector)
                    if interactive:
                        _, filename = mkstemp(suffix=suffix)
                    else:
                        if not (filename.endswith('.vtp') or filename.endswith('.vtu')):
                            filename = filename + suffix
                    visualize_single(U.impl, name, filename)
                    if interactive:
                        subprocess.call(['paraview', filename])
                        to_clean_up.append(filename)
                else:
                    # we presume we have a single trajectory to be visualized
                    assert all([isinstance(u, DuneXTVector) for u in U._list])
                    data = OrderedDict()
                    if interactive:
                        for ii in range(len(U)):
                            _, filename = mkstemp(suffix='_{}{}'.format(ii, suffix))
                            data[filename] = U._list[ii]
                    else:
                        if (filename.endswith('.vtp') or filename.endswith('.vtu')):
                            filename = filename[:-4]
                        assert len(filename) > 0
                        for ii in range(len(U)):
                            data['{}_{}{}'.format(filename, ii, suffix)] = U._list[ii]
                    for f_name, vector in data.items():
                        visualize_single(vector.impl, name, f_name)
                    if interactive:
                        _, pvd_filename = mkstemp(suffix='.pvd')
                        with open(pvd_filename, 'w') as pvd_file:
                            pvd_file.write('<?xml version="1.0"?>\n')
                            pvd_file.write('<VTKFile type="Collection" version="0.1">\n')
                            pvd_file.write('  <Collection>\n')
                            timestep = 0
                            for f_name in data:
                                pvd_file.write('    <DataSet timestep="{}" group="" part="0" file="{}"/>\n'.format(
                                    timestep, f_name))
                                timestep += 1
                            pvd_file.write('  </Collection>\n')
                            pvd_file.write('</VTKFile>\n')
                        subprocess.call(['paraview', pvd_filename])
                        for f_name in data:
                            to_clean_up.append(f_name)
                        to_clean_up.append(pvd_filename)

            for f_name in to_clean_up:
                os.remove(f_name)


    class DuneGDTpyMORVisualizerWrapper(ImmutableInterface):
        """Visualize a dune-gdt discrete function using the visualizer of pyMOR's discretizations.

        Currently, only visualization of Q1 CG on a 2d YaspGrid is supported.

        Parameters
        ----------
        grid
            The pyMOR |Grid| on which to interpret the DoF vectors.
        """

        def __init__(self, grid):
            assert isinstance(grid, RectGrid)
            assert grid.dim == 2
            self.grid = grid
            self.visualizer = PatchVisualizer(grid=grid, bounding_box=grid.bounding_box(), codim=2)

        def visualize(self, U, discretization, title=None, legend=None, separate_colorbars=False,
                      rescale_colorbars=False, block=None, filename=None, columns=2):

            def wrap(V):
                return ListVectorArray([NumpyVector(u.data) for u in V._list], space=NumpyListVectorSpace(V.dim))

            if isinstance(U, tuple):
                U = tuple([wrap(u) for u in U])
            else:
                U = wrap(U)

            self.visualizer.visualize(U, discretization, title=title, legend=legend,
                                      separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                                      block=block, filename=filename, columns=columns)

