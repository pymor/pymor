# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
from pathlib import Path

from pymor.core.config import config
config.require('NGSOLVE')


from pymor.core.defaults import defaults
from pymor.tools.io import change_to_directory


import ngsolve as ngs
import numpy as np

from pymor.core.base import ImmutableObject
from pymor.operators.list import LinearComplexifiedListVectorArrayOperatorBase
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.vectorarrays.list import CopyOnWriteVector, ComplexifiedVector, ComplexifiedListVectorSpace


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

    def __init__(self, V, id='STATE'):
        self.__auto_init(locals())

    def __eq__(self, other):
        return type(other) is NGSolveVectorSpace and self.V == other.V and self.id == other.id

    def __hash__(self):
        return hash(self.V) + hash(self.id)

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
    def space_from_vector_obj(cls, vec, id):
        return cls(vec.space, id)

    def real_zero_vector(self):
        impl = ngs.GridFunction(self.V)
        return NGSolveVector(impl)

    def real_make_vector(self, obj):
        return NGSolveVector(obj)

    def real_vector_from_numpy(self, data, ensure_copy=False):
        v = self.real_zero_vector()
        v.to_numpy()[:] = data
        return v


class NGSolveMatrixOperator(LinearComplexifiedListVectorArrayOperatorBase):
    """Wraps a NGSolve matrix as an |Operator|."""

    def __init__(self, matrix, range, source, solver_options=None, name=None):
        self.__auto_init(locals())

    @defaults('default_solver')
    def _prepare_apply(self, U, mu, kind, least_squares=False, default_solver=''):
        if kind == 'apply_inverse':
            if least_squares:
                raise NotImplementedError
            solver = self.solver_options.get('inverse', default_solver) if self.solver_options else default_solver
            inv = self.matrix.Inverse(self.source.V.FreeDofs(), inverse=solver)
            return inv

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

    def _real_apply_inverse_one_vector(self, v, mu=None, initial_guess=None,
                                       least_squares=False, prepare_data=None):
        inv = prepare_data
        r = self.source.real_zero_vector()
        r.impl.vec.data = inv * v.impl.vec
        return r

    def _assemble_lincomb(self, operators, coefficients, identity_shift=0., solver_options=None, name=None):
        if not all(isinstance(op, NGSolveMatrixOperator) for op in operators):
            return None
        if identity_shift != 0:
            return None

        matrix = operators[0].matrix.CreateMatrix()
        matrix.AsVector().data = float(coefficients[0]) * matrix.AsVector()
        for op, c in zip(operators[1:], coefficients[1:]):
            matrix.AsVector().data += float(c) * op.matrix.AsVector()
        return NGSolveMatrixOperator(matrix, self.range, self.source, solver_options=solver_options, name=name)

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

            for u, name in zip(U, legend):
                ngs.Draw(u.vectors[0].real_part.impl, self.mesh, name=name)
