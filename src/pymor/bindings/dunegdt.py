# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
config.require('DUNEGDT')


import numpy as np

from dune.xt.la import IstlVector

from pymor.operators.list import LinearComplexifiedListVectorArrayOperatorBase
from pymor.vectorarrays.interface import _create_random_values
from pymor.vectorarrays.list import ComplexifiedListVectorSpace, CopyOnWriteVector


class DuneXTVector(CopyOnWriteVector):
    """Wraps a vector from dune-xt to make it usable with ListVectorArray.

    Parameters
    ----------
    impl
        The actual vector from dune.xt.la, usually IstlVector.
    """

    def __init__(self, impl):
        self.impl = impl

    @classmethod
    def from_instance(cls, instance):
        return cls(instance.impl)

    def _copy_data(self):
        self.impl = self.impl.copy(True)

    def _scal(self, alpha):
        self.impl.scal(alpha)

    def _axpy(self, alpha, x):
        self.impl.axpy(alpha, x.impl)

    def inner(self, other):
        return self.impl.dot(other.impl)

    def norm(self):
        return self.impl.l2_norm()

    def norm2(self):
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

    def to_numpy(self, ensure_copy=False):
        return np.array(self.impl, copy=ensure_copy)


class DuneXTVectorSpace(ComplexifiedListVectorSpace):
    """A |VectorSpace| yielding DuneXTVector

    Parameters
    ----------
    dim
        Dimension of the |VectorSpace|, i.e., length of the resulting vectors.
    vector_type
        Type of the actual vector from dune.xt.la, usually IstlVector.
    id
        Identifier of the |VectorSpace|.
    """

    real_vector_type = DuneXTVector

    def __init__(self, dim, dune_vector_type=IstlVector, id='STATE'):
        self.__auto_init(locals())

    def __eq__(self, other):
        return type(other) is DuneXTVectorSpace \
            and self.dune_vector_type == other.dune_vector_type \
            and self.dim == other.dim \
            and self.id == other.id

    # since we implement __eq__, we also need to implement __hash__
    def __hash__(self):
        return id(self.dune_vector_type) + hash(self.dim)

    def real_zero_vector(self):
        return DuneXTVector(self.dune_vector_type(self.dim, 0.))

    def real_full_vector(self, value):
        return DuneXTVector(self.dune_vector_type(self.dim, value))

    def real_random_vector(self, distribution, random_state, **kwargs):
        values = _create_random_values(self.dim, distribution, random_state, **kwargs)
        return self.real_vector_from_numpy(values)

    def real_vector_from_numpy(self, data, ensure_copy=False):
        v = self.real_zero_vector()
        np_view = np.array(v.impl, copy=False)
        np_view[:] = data
        return v

    def real_make_vector(self, obj):
        return DuneXTVector(obj)


class DuneXTMatrixOperator(LinearComplexifiedListVectorArrayOperatorBase):
    """Wraps a dune-xt matrix as an |Operator|.

    Parameters
    ----------
    matrix
        The actual matrix from dune.xt.la, usually IstlMatrix.
    source_id
        Identifier of the source |VectorSpace|.
    range_id
        Identifier of the source |VectorSpace|.
    solver_options
        If specified, either a string or a dict specifying the solver used in apply_inverse. See
        https://zivgitlab.uni-muenster.de/ag-ohlberger/dune-community/dune-xt/-/tree/master/dune/xt/la/solver
        for available options, depending on the type of `matrix`. E.g., for
        dune.xt.la.IstlSparseMatrix, (as can be queried from dune.xt.la.IstlSparseMatrixSolver
        via `types()` and `options(type)`):
        - 'bicgstab.ssor'
        - 'bicgstab.amg.ssor'
        - 'bicgstab.amg.ilu0'
        - 'bicgstab.ilut'
        - 'bicgstab'
        - 'cg'
    name
        Optional name of the resulting |Operator|.
    """

    linear = True

    def __init__(self, matrix, source_id='STATE', range_id='STATE', solver_options=None, name=None):
        self.source = DuneXTVectorSpace(matrix.cols, matrix.vector_type(), source_id)
        self.range = DuneXTVectorSpace(matrix.rows, matrix.vector_type(), range_id)
        self.__auto_init(locals())

    def _real_apply_one_vector(self, u, mu=None, prepare_data=None):
        r = self.range.real_zero_vector()
        self.matrix.mv(u.impl, r.impl)
        return r

    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        r = self.source.real_zero_vector()
        self.matrix.mtv(v.impl, r.impl)
        return r

    def _real_apply_inverse_one_vector(self, v, mu=None, initial_guess=None,
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
            matrix.axpy(c, op.matrix)  # TODO: Not guaranteed to work for all backends! For different
            # sparsity patterns one would have to extract the patterns from the pruned
            # matrices, merge them and create a new matrix.

        return DuneXTMatrixOperator(matrix, self.source.id, self.range.id, solver_options=solver_options, name=name)
