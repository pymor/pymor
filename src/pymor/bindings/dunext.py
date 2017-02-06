# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_DUNEXT:
    import numpy as np

    from pymor.operators.basic import OperatorBase
    from pymor.operators.constructions import ZeroOperator
    from pymor.vectorarrays.list import VectorInterface, ListVectorSpace


    class DuneXTVector(VectorInterface):
        """Wraps a vector from dune-xt-la to make it usable with ListVectorArray."""

        def __init__(self, impl):
            self.impl = impl

        @property
        def data(self):
            return np.frombuffer(self.impl)

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

        def components(self, component_indices):
            impl = self.impl
            return np.array([impl[i] for i in component_indices])

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



    class DuneXTVectorSpace(ListVectorSpace):

        def __init__(self, vector_type, dim, id_='STATE'):
            self.vector_type = vector_type
            self.dim = dim
            self.id = id_

        def __eq__(self, other):
            return type(other) is DuneXTVectorSpace and self.vector_type == other.vector_type and self.dim == other.dim

        # since we implement __eq__, we also need to implement __hash__
        def __hash__(self):
            return id(self.vector_type) + hash(self.dim)

        def zero_vector(self):
            return DuneXTVector(self.vector_type(self.dim, 0.))

        def make_vector(self, obj):
            return DuneXTVector(obj)

        def vector_from_data(self, data):
            v = self.zero_vector()
            v.data[:] = data
            return v

        @classmethod
        def space_from_vector_obj(cls, vec, id_):
            return cls(type(vec), len(vec), id_)


    class DuneXTMatrixOperator(OperatorBase):
        """Wraps a dune-xt-la matrix as an |Operator|."""

        linear = True

        def __init__(self, matrix, source_id='STATE', range_id='STATE', solver_options=None, name=None):
            self.source = DuneXTVectorSpace(matrix.vector_type(), matrix.cols(), source_id)
            self.range = DuneXTVectorSpace(matrix.vector_type(), matrix.rows(), range_id)
            self.matrix = matrix
            self.solver_options = solver_options
            self.name = name
            self.source_id = source_id  # for with_ support
            self.range_id = range_id

        def apply(self, U, mu=None):
            assert U in self.source
            R = self.range.zeros(len(U))
            for u, r in zip(U._list, R._list):
                self.matrix.mv(u.impl, r.impl)
            return R

        def apply_transpose(self, V, mu=None):
            raise NotImplementedError

        def apply_inverse(self, V, mu=None, least_squares=False):
            assert V in self.range
            if least_squares:
                raise NotImplementedError

            from dune.xt.la import make_solver
            solver = make_solver(self.matrix)
            R = self.source.zeros(len(V))
            options = self.solver_options.get('inverse') if self.solver_options else None
            for v, r in zip(V._list, R._list):
                if options:
                    solver.apply(v.impl, r.impl, options)
                else:
                    solver.apply(v.impl, r.impl)
            return R

        def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
            if not all(isinstance(op, (DuneXTMatrixOperator, ZeroOperator)) for op in operators):
                return None

            if isinstance(operators[0], ZeroOperator):
                return operators[1].assemble_lincomb(operators[1:], coefficients[1:],
                                                     solver_options=solver_options, name=name)

            matrix = operators[0].matrix.copy()
            matrix.scal(coefficients[0])
            for op, c in zip(operators[1:], coefficients[1:]):
                if isinstance(op, ZeroOperator):
                    continue
                matrix.axpy(c, op.matrix)

            return DuneXTMatrixOperator(matrix, self.source.id, self.range.id, solver_options=solver_options, name=name)
