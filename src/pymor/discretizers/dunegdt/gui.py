# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_DUNEGDT:

    import numpy as np
    from collections import OrderedDict
    import os
    import subprocess
    import tempfile
    from tempfile import mkstemp
    from matplotlib import pyplot as plt

    from dune.xt.common.vtk.plot import plot as k3d_plot
    from dune.gdt import DiscreteFunction

    from pymor.bindings.dunegdt import ComplexifiedDuneXTVector, DuneXTVector
    from pymor.core.base import ImmutableObject
    from pymor.discretizers.builtin.grids.oned import OnedGrid
    from pymor.discretizers.builtin.gui.visualizers import OnedVisualizer
    from pymor.vectorarrays.list import ListVectorArray
    from pymor.vectorarrays.numpy import NumpyVectorSpace


    class DuneGDT1dAsNumpyVisualizer(ImmutableObject):
        """Visualize a dune-gdt discrete function using OnedVisualizer.

        Parameters
        ----------
        space
            The dune-gdt space for which we want to visualize DOF vectors.
        grid
            The dune grid associated with space (assumed to be equidistant!).
        """

        def __init__(self, space, grid):
            assert grid.dimension == 1
            assert space.dimDomain == 1
            # build pyMOR grid
            centers = np.array(grid.centers(1), copy=False).ravel()
            domain = (centers[0], centers[-1])
            num_intervals = grid.size(0)
            pymor_grid = OnedGrid(domain, num_intervals)
            if space.max_polorder == 0:
                assert space.num_DoFs == grid.size(0)
                self.visualizer = OnedVisualizer(pymor_grid, codim=0)
            elif space.min_polorder == space.max_polorder == 1:
                assert space.num_DoFs == grid.size(1)
                self.visualizer = OnedVisualizer(pymor_grid, codim=1)
            else:
                # TODO: add P1 interpolation?
                raise NotImplementedError('Not available for higher polynomial orders!')
            self.__auto_init(locals())

        def visualize(self, U, *args, **kwargs):
            # convert to NumpyVectorArray
            U_np = NumpyVectorSpace(U.dim).zeros(len(U))
            for ii, u_dune in enumerate(U._list):
                if isinstance(u_dune, ComplexifiedDuneXTVector):
                    assert u_dune.imag_part is None
                    u_dune = u_dune.real_part
                assert isinstance(u_dune, DuneXTVector)
                u_np = np.array(u_dune.impl, copy=False)
                U_np._array[ii, :] = u_np[:]
            return self.visualizer.visualize(U_np, *args, **kwargs)


    class DuneGDT1dMatplotlibVisualizer(ImmutableObject):
        """Visualize a dune-gdt discrete function using matplotlib.

        Parameters
        ----------
        space
            The dune-gdt space for which we want to visualize DOF vectors.
        """

        def __init__(self, space):
            assert space.dimDomain == 1
            assert space.max_polorder == 1
            self.interpolation_points = space.interpolation_points()
            self.__auto_init(locals())

        def visualize(self, U, m, title=None, legend=None, separate_colorbars=False,
                      rescale_colorbars=False, block=None, filename=None, columns=2):
            assert isinstance(U, ListVectorArray)
            assert len(U) == 1
            U = U._list[0]
            if isinstance(U, ComplexifiedDuneXTVector):
                assert U.imag_part is None
                U = U.real_part
            assert isinstance(U, DuneXTVector)
            X = np.array(self.interpolation_points, copy=False)
            Y = np.array(U.impl, copy=False)
            name = legend if legend else 'STATE'
            plt.plot(X, Y, label=name)
            if title:
                plt.title(title)
            plt.legend()
            plt.plot()


    class DuneGDTParaviewVisualizer(ImmutableObject):
        """Visualize a dune-gdt discrete function using paraview.

        Parameters
        ----------
        space
            The dune-gdt space for which we want to visualize DOF vectors.
        """

        def __init__(self, space):
            self.__auto_init(locals())

        def visualize(self, U, m, title=None, legend=None, separate_colorbars=False,
                      rescale_colorbars=False, block=None, filename=None, columns=2):

            def visualize_single(vec, vec_name, f_name):
                discrete_function = DiscreteFunction(self.space, vec, vec_name)
                discrete_function.visualize(f_name[:-4])

            suffix = '.vtp' if self.space.dimDomain == 1 else '.vtu'
            interactive = filename is None
            to_clean_up = []

            if isinstance(U, tuple):
                # we presume to have several vectors which should be visualized side by side
                assert all([isinstance(u, ListVectorArray) for u in U])
                assert all([len(u) == 1 for u in U])
                assert all([all([isinstance(v, ComplexifiedDuneXTVector) for v in u._list]) for u in U])
                assert all([all([v.imag_part is None for v in u._list]) for u in U])
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
                        data[filename] = U[ii]._list[0].real_part
                else:
                    if (filename.endswith('.vtp') or filename.endswith('.vtu')):
                        filename = filename[:-4]
                    assert len(filename) > 0
                    for ii in range(len(U)):
                        data['{}_{}{}'.format(filename, ii, suffix)] = U[ii]._list[0].real_part
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
                            pvd_file.write('    <DataSet timestep="0" group="" part="{}" file="{}"/>\n'.format(
                                part, f_name))
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
                    if isinstance(U, ComplexifiedDuneXTVector):
                        assert U.imag_part is None
                        U = U.real_part
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
                    assert all([isinstance(u, ComplexifiedDuneXTVector) for u in U._list])
                    assert all([u.imag_part is None for u in U._list])
                    data = OrderedDict()
                    if interactive:
                        for ii in range(len(U)):
                            _, filename = mkstemp(suffix='_{}{}'.format(ii, suffix))
                            data[filename] = U._list[ii].real_part
                    else:
                        if (filename.endswith('.vtp') or filename.endswith('.vtu')):
                            filename = filename[:-4]
                        assert len(filename) > 0
                        for ii in range(len(U)):
                            data['{}_{}{}'.format(filename, ii, suffix)] = U._list[ii].real_part
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


    class DuneGDTK3dVisualizer(ImmutableObject):
        """Visualize a dune-gdt discrete function within a jupyter notebook using K3d.

        Parameters
        ----------
        space
            The dune-gdt space for which we want to visualize DOF vectors.
        """

        def __init__(self, space, name='STATE'):
            self.__auto_init(locals())

        def visualize(self, U, *args, **kwargs):

            def visualize_single(u, filename):
                if isinstance(u, ComplexifiedDuneXTVector):
                    assert u.imag_part is None
                    u = u.real_part
                assert isinstance(u, DuneXTVector)
                df = DiscreteFunction(self.space, u.impl, self.name)
                df.visualize(filename)

            prefix = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()))
            suffix = 'vt{}'.format('p' if self.space.dimDomain == 1 else 'u')
            if isinstance(U, ListVectorArray):
                if len(U) == 0:
                    return
                for i in range(len(U)):
                    visualize_single(U._list[i], f'{prefix}_{i}')
                if len(U) == 1:
                    filename = f'{prefix}_{i}.{suffix}'
                else:
                    with open(f'{prefix}.pvd', 'w') as pvd_file:
                        pvd_file.write('<?xml version=\'1.0\'?>\n')
                        pvd_file.write('<VTKFile type=\'Collection\' version=\'0.1\'>\n')
                        pvd_file.write('<Collection>\n')
                        for i in range(len(U)):
                            pvd_file.write(
                    f'<DataSet timestep=\'{i}\' part=\'1\' name=\'{self.name}\' file=\'{prefix}_{i}.{suffix}\'/>\n')
                        pvd_file.write('</Collection>\n')
                        pvd_file.write('</VTKFile>\n')
                    filename = f'{prefix}.pvd'
            else:
                raise NotImplementedError

            return k3d_plot(filename, color_attribute_name=self.name)
