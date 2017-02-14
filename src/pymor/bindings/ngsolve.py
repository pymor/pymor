# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

if config.HAVE_NGSOLVE:
    import ngsolve as ngs
    import numpy as np

    from pymor.core.interfaces import ImmutableInterface
    from pymor.operators.basic import OperatorBase
    from pymor.operators.constructions import ZeroOperator
    from pymor.vectorarrays.interfaces import VectorArrayInterface
    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace


    class NGSolveVector(CopyOnWriteVector):
        """Wraps a NGSolve BaseVector to make it usable with ListVectorArray."""

        def __init__(self, impl):
            self.impl = impl
            self._array = impl.FV().NumPy()

        @classmethod
        def from_instance(cls, instance):
            return cls(instance.impl)

        def _copy_data(self):
            new_impl = ngs.BaseVector(self.impl.size)
            new_impl.data = self.impl
            self.impl = new_impl
            self._array = new_impl.FV().NumPy()

        @property
        def data(self):
            return self._array

        def _scal(self, alpha):
            self.impl.data = float(alpha) * self.impl

        def _axpy(self, alpha, x):
            self.impl.data = self.impl + float(alpha) * x.impl

        def dot(self, other):
            return self.impl.InnerProduct(other.impl)

        def l1_norm(self):
            return np.linalg.norm(self._array, ord=1)

        def l2_norm(self):
            return self.impl.Norm()

        def l2_norm2(self):
            return self.impl.Norm() ** 2

        def components(self, component_indices):
            return self._array[component_indices]

        def amax(self):
            A = np.abs(self._array)
            max_ind = np.argmax(A)
            max_val = A[max_ind]
            return max_ind, max_val


    class NGSolveVectorSpace(ListVectorSpace):

        def __init__(self, dim, id_='STATE'):
            self.dim = dim
            self.id = id_

        def __eq__(self, other):
            return type(other) is NGSolveVectorSpace and self.dim == other.dim and self.id == other.id

        def __hash__(self):
            return hash(self.dim) + hash(self.id)

        def zero_vector(self):
            impl = ngs.BaseVector(self.dim)
            impl.FV().NumPy()[:] = 0
            return NGSolveVector(impl)

        def make_vector(self, obj):
            return NGSolveVector(obj)


    class NGSolveMatrixOperator(OperatorBase):
        """Wraps a NGSolve matrix as an |Operator|."""

        linear = True

        def __init__(self, matrix, free_dofs=None, name=None):
            self.source = NGSolveVectorSpace(matrix.width)
            self.range = NGSolveVectorSpace(matrix.height)
            self.matrix = matrix
            self.free_dofs = free_dofs
            self.name = name

        def apply(self, U, mu=None):
            assert U in self.source
            R = self.range.zeros(len(U))
            for u, r in zip(U._list, R._list):
                self.matrix.Mult(u.impl, r.impl, 1.)
            return R

        def apply_transpose(self, V, mu=None):
            assert V in self.range
            U = self.source.zeros(len(V))
            mat = self.matrix.Transpose()
            for v, u in zip(V._list, U._list):
                mat.Mult(v.impl, u.impl, 1.)
            return U

        def apply_inverse(self, V, mu=None, least_squares=False):
            assert V in self.range
            if least_squares:
                raise NotImplementedError
            R = self.source.zeros(len(V))
            with ngs.TaskManager():
                inv = self.matrix.Inverse(self.free_dofs)
                for r, v in zip(R._list, V._list):
                    r.impl.data = inv * v.impl
            return R

        def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
            if not all(isinstance(op, (NGSolveMatrixOperator, ZeroOperator)) for op in operators):
                return None
            assert not solver_options

            matrix = operators[0].matrix.CreateMatrix()
            matrix.AsVector().data = float(coefficients[0]) * matrix.AsVector()
            for op, c in zip(operators[1:], coefficients[1:]):
                if isinstance(op, ZeroOperator):
                    continue
                matrix.AsVector().data += float(c) * op.matrix.AsVector()
            return NGSolveMatrixOperator(matrix, self.free_dofs, name=name)


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
