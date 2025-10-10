# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
from pathlib import Path

from pymor.core.config import config

config.require('NGSOLVE')


import ngsolve as ngs
import numpy as np

from pymor.core.base import ImmutableObject
from pymor.core.defaults import defaults
from pymor.operators.list import LinearComplexifiedListVectorArrayOperatorBase
from pymor.solvers.list import ComplexifiedListVectorArrayBasedSolver
from pymor.tools.io import change_to_directory
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.list import ComplexifiedListVectorSpace, ComplexifiedVector, CopyOnWriteVector
from pymor.vectorarrays.numpy import NumpyVectorSpace


class NGSolveVectorCommon:
    def amax(self):
        A = np.abs(self.to_numpy())
        max_ind = np.argmax(A)
        max_val = A[max_ind]
        return max_ind, max_val

    def dofs(self, dof_indices):
        return self.to_numpy()[dof_indices]


class NGSolveVector(NGSolveVectorCommon, CopyOnWriteVector):
    """Wraps a NGSolve BaseVector to make it usable with ListVectorArray."""

    def __init__(self, impl):
        self.impl = impl

    @classmethod
    def from_instance(cls, instance):
        return cls(instance.impl)

    def _copy_data(self):
        new_impl = ngs.GridFunction(self.impl.space)
        new_impl.vec.data = self.impl.vec
        self.impl = new_impl

    def to_numpy(self, ensure_copy=False):
        if ensure_copy:
            return self.impl.vec.FV().NumPy().copy()
        self._copy_data_if_needed()
        return self.impl.vec.FV().NumPy()

    def _scal(self, alpha):
        self.impl.vec.data = float(alpha) * self.impl.vec

    def _axpy(self, alpha, x):
        self.impl.vec.data = self.impl.vec + float(alpha) * x.impl.vec

    def inner(self, other):
        return self.impl.vec.InnerProduct(other.impl.vec)

    def norm(self):
        return self.impl.vec.Norm()

    def norm2(self):
        return self.impl.vec.Norm() ** 2


class ComplexifiedNGSolveVector(NGSolveVectorCommon, ComplexifiedVector):
    pass


class NGSolveVectorSpace(ComplexifiedListVectorSpace):

    real_vector_type = NGSolveVector
    vector_type = ComplexifiedNGSolveVector

    def __init__(self, V):
        self.__auto_init(locals())

    def __eq__(self, other):
        return type(other) is NGSolveVectorSpace and self.V == other.V

    def __hash__(self):
        return hash(self.V)

    @property
    def value_dim(self):
        u = self.V.TrialFunction()
        if isinstance(u, list):
            return u[0].dim
        else:
            return u.dim

    @property
    def dim(self):
        return self.V.ndofglobal * self.value_dim

    @classmethod
    def space_from_vector_obj(cls, vec):
        return cls(vec.space)

    def real_zero_vector(self):
        impl = ngs.GridFunction(self.V)
        return NGSolveVector(impl)

    def real_make_vector(self, obj):
        return NGSolveVector(obj)

    def real_vector_from_numpy(self, data, ensure_copy=False):
        v = self.real_zero_vector()
        v.to_numpy()[:] = data
        return v


class NGSolveLinearSolver(ComplexifiedListVectorArrayBasedSolver):

    @defaults('method')
    def __init__(self, method=''):
        self.__auto_init(locals())

    def _prepare(self, operator, U, mu, adjoint):
        operator = operator.assemble(mu)
        if adjoint:
            raise NotImplementedError
        return operator.matrix.Inverse(operator.source.V.FreeDofs(), inverse=self.method)

    def _real_solve_one_vector(self, operator, v, mu, initial_guess, prepare_data):
        inv = prepare_data
        r = operator.source.real_zero_vector()
        r.impl.vec.data = inv * v.impl.vec
        return r

    def _real_solve_adjoint_one_vector(self, operator, u, mu, initial_guess, prepare_data):
        raise NotImplementedError


class NGSolveMatrixOperator(LinearComplexifiedListVectorArrayOperatorBase):
    """Wraps a NGSolve matrix as an |Operator|."""

    def __init__(self, matrix, range, source, solver=None, name=None):
        solver = solver or NGSolveLinearSolver()
        self.__auto_init(locals())

    def _real_apply_one_vector(self, u, mu=None, prepare_data=None):
        r = self.range.real_zero_vector()
        self.matrix.Mult(u.impl.vec, r.impl.vec)
        return r

    def _real_apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        u = self.source.real_zero_vector()
        try:
            mat = self.matrix.Transpose()
        except AttributeError:
            mat = self.matrix.T
        mat.Mult(v.impl.vec, u.impl.vec)
        return u

    def _assemble_lincomb(self, operators, coefficients, identity_shift=0., name=None):
        if not all(isinstance(op, NGSolveMatrixOperator) for op in operators):
            return None
        if identity_shift != 0:
            return None

        matrix = operators[0].matrix.CreateMatrix()
        matrix.AsVector().data = float(coefficients[0]) * matrix.AsVector()
        for op, c in zip(operators[1:], coefficients[1:], strict=True):
            matrix.AsVector().data += float(c) * op.matrix.AsVector()
        return NGSolveMatrixOperator(matrix, self.range, self.source, name=name)

    def as_vector(self, copy=True):
        vec = self.matrix.AsVector().FV().NumPy()
        return NumpyVectorSpace.make_array(vec.copy() if copy else vec)


class NGSolveVisualizer(ImmutableObject):
    """Visualize an NGSolve grid function."""

    def __init__(self, mesh, fespace):
        self.__auto_init(locals())
        self.space = NGSolveVectorSpace(fespace)

    def visualize(self, U, legend=None, separate_colorbars=True, filename=None, block=True):
        """Visualize the provided data."""
        if isinstance(U, VectorArray):
            U = (U,)
        assert all(u in self.space for u in U)
        if any(len(u) != 1 for u in U):
            raise NotImplementedError
        if any(u.vectors[0].imag_part is not None for u in U):
            raise NotImplementedError
        if legend is None:
            legend = [f'VectorArray{i}' for i in range(len(U))]
        if isinstance(legend, str):
            legend = [legend]
        assert len(legend) == len(U)
        legend = [l.replace(' ', '_') for l in legend]  # NGSolve GUI will fail otherwise

        if filename:
            # ngsolve unconditionally appends ".vtk"
            filename = Path(filename).resolve()
            if filename.suffix == '.vtk':
                filename = filename.parent / filename.stem
            else:
                self.logger.warning(f'NGSolve set VTKOutput filename to {filename}.vtk')
            coeffs = [u.vectors[0].real_part.impl for u in U]
            # ngsolve cannot handle full paths for filenames
            with change_to_directory(filename.parent):
                vtk = ngs.VTKOutput(ma=self.mesh, coefs=coeffs, names=legend, filename=str(filename), subdivision=0)
                vtk.Do()
        else:
            if not separate_colorbars:
                raise NotImplementedError

            for u, name in zip(U, legend, strict=True):
                ngs.Draw(u.vectors[0].real_part.impl, self.mesh, name=name)
