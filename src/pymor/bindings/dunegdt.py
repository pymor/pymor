# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_DUNEGDT:

    import numpy as np
    from functools import partial
    from collections import OrderedDict
    import os
    import subprocess
    import tempfile
    from tempfile import mkstemp

    from dune.xt.common.vtk.plot import plot as k3d_plot
    from dune.gdt import DiscreteFunction

    from pymor.core.base import ImmutableObject
    from pymor.operators.list import ListVectorArrayOperatorBase
    from pymor.operators.constructions import ZeroOperator
    from pymor.vectorarrays.list import ListVectorArray, Vector, ComplexifiedVector, ComplexifiedListVectorSpace
    from pymor.vectorarrays.interface import _create_random_values


    class DuneXTVector(Vector):
        """Wraps a vector from dune-xt to make it usable with ListVectorArray."""

        def __init__(self, impl):
            self.impl = impl

        # @property
        # def data(self):
            # return np.frombuffer(self.impl)

        def copy(self, deep=False):
            return DuneXTVector(self.impl.copy(deep))

        def scal(self, alpha):
            self.impl.scal(alpha)

        def axpy(self, alpha, x):
            self.impl.axpy(alpha, x.impl)

        def dot(self, other):
            return self.impl.dot(other.impl)

        def l1_norm(self):
            return self.impl.l1_norm()

        def l2_norm(self):
            return self.impl.l2_norm()

        def l2_norm2(self):
            return self.impl.l2_norm() ** 2

        def sup_norm(self):
            return self.impl.sup_norm()

        def dofs(self, dof_indices):
            impl = self.impl
            return np.array([impl[i] for i in dof_indices])

        def amax(self):
            _amax = self.impl.amax()
            return _amax[0], _amax[1]

        def __add__(self, other):
            return DuneXTVector(self.impl + other.impl)

        def __iadd__(self, other):
            self.impl += other.impl
            return self

        __radd__ = __add__

        def __sub__(self, other):
            return DuneXTVector(self.impl - other.impl)

        def __isub__(self, other):
            self.impl -= other.impl
            return self

        def __mul__(self, other):
            return DuneXTVector(self.impl * other)

        def __imul__(self, other):
            self.impl *= other
            return self

        def __neg__(self):
            return self * (-1)


    class DuneXTVectorSpace(ComplexifiedListVectorSpace):

        def __init__(self, vector_type, dim, id='STATE'):
            self.__auto_init(locals())

        def __eq__(self, other):
            return type(other) is DuneXTVectorSpace and self.vector_type == other.vector_type and self.dim == other.dim

        # since we implement __eq__, we also need to implement __hash__
        def __hash__(self):
            return id(self.vector_type) + hash(self.dim)

        def real_zero_vector(self):
            return DuneXTVector(self.vector_type(self.dim, 0.))

        def real_full_vector(self, value):
            return DuneXTVector(self.vector_type(self.dim, value))

        def real_random_vector(self, distribution, random_state, **kwargs):
            values = _create_random_values(self.dim, distribution, random_state, **kwargs)
            return self.vector_from_numpy(values)

        def real_make_vector(self, obj):
            return DuneXTVector(obj)

        def real_vector_from_numpy(self, data, ensure_copy=False):
            v = self.zero_vector()
            np_view = np.array(v, copy=False)
            np_view[:] = data
            return v


    class DuneXTMatrixOperator(ListVectorArrayOperatorBase):
        """Wraps a dune-xt matrix as an |Operator|."""

        linear = True

        def __init__(self, matrix, source_id='STATE', range_id='STATE', solver_options=None, name=None):
            self.source = DuneXTVectorSpace(matrix.vector_type(), matrix.cols, source_id)
            self.range = DuneXTVectorSpace(matrix.vector_type(), matrix.rows, range_id)
            self.__auto_init(locals())

        def _apply_one_vector(self, u, mu=None, prepare_data=None):
            r = self.range.real_zero_vector()
            self.matrix.mv(u.impl, r.impl)
            return r

        def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
            r = self.source.real_zero_vector()
            self.matrix.mtv(v.impl, r.impl)
            return r

        def _apply_inverse_one_vector(self, v, mu=None, initial_guess=None,
                                      least_squares=False, prepare_data=None):
            if least_squares:
                raise NotImplementedError
            r = (self.source.real_zero_vector() if initial_guess is None else
                 initial_guess.copy(deep=True))
            options = self.solver_options.get('inverse') if self.solver_options else None

            from dune.xt.la import make_solver
            solver = make_solver(self.matrix)
            if options:
                solver.apply(v.impl, r.impl, options)
            else:
                solver.apply(v.impl, r.impl)
            return r

        def _assemble_lincomb(self, operators, coefficients, identity_shift=0., solver_options=None, name=None):
            if not all(isinstance(op, DuneXTMatrixOperator) for op in operators):
                return None
            if identity_shift != 0:
                return None
            if np.iscomplexobj(coefficients):
                return None

            if coefficients[0] == 1:
                matrix = operators[0].matrix.copy()
            else:
                matrix = operators[0].matrix * coefficients[0]
            for op, c in zip(operators[1:], coefficients[1:]):
                matrix.axpy(c, op.matrix) # does not work for all backends for different sparsity patterns
                # one would have to extract the patterns from the pruned matrices, merge them and create a new matrix

            return DuneXTMatrixOperator(matrix, self.source.id, self.range.id, solver_options=solver_options, name=name)


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


    class DuneGDTK3dVisualizer(ImmutableObject):
        """Visualize a dune-gdt discrete function within a jupyter notebook using K3d.

        Parameters
        ----------
        space
            The dune-gdt space for which we want to visualize DOF vectors.
        """

        def __init__(self, space, name='STATE'):
            # self.dune_vec = CommonDenseVectorDouble(space.num_DoFs)
            # self.np_view = np.array(self.dune_vec, copy=False)
            # self.discrete_function = make_discrete_function(space, self.dune_vec, name)
            self.__auto_init(locals())

        def visualize(self, U, m):

            def visualize_single(u, filename):
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
                            pvd_file.write(f'<DataSet timestep=\'{i}\' part=\'1\' name=\'{self.name}\' file=\'{prefix}_{i}.{suffix}\'/>\n')
                        pvd_file.write('</Collection>\n')
                        pvd_file.write('</VTKFile>\n') 
                    filename = f'{prefix}.pvd'
            else:
                raise NotImplementedError
            _ = k3d_plot(filename, color_attribute_name=self.name)

