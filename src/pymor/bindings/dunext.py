# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_DUNEXT:
    import numpy as np
    from functools import partial

    from pymor.operators.list import ListVectorArrayOperatorBase
    from pymor.operators.constructions import ZeroOperator
    from pymor.vectorarrays.list import Vector, ComplexifiedVector, ComplexifiedListVectorSpace
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

