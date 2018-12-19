# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.exceptions import InversionError
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.list import NumpyListVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace


class NumpyListVectorArrayMatrixOperator(NumpyMatrixOperator):
    """Variant of |NumpyMatrixOperator| using |ListVectorArray| instead of |NumpyVectorArray|."""

    def __init__(self, matrix, source_id=None, range_id=None, solver_options=None, name=None):
        super().__init__(matrix, source_id=source_id, range_id=range_id, solver_options=solver_options, name=name)
        functional = self.range_id is None
        vector = self.source_id is None
        if functional and vector:
            raise NotImplementedError
        if vector:
            self.source = NumpyVectorSpace(1, source_id)
        else:
            self.source = NumpyListVectorSpace(matrix.shape[1], source_id)
        if functional:
            self.range = NumpyVectorSpace(1, range_id)
        else:
            self.range = NumpyListVectorSpace(matrix.shape[0], range_id)
        self.functional = functional
        self.vector = vector

    def apply(self, U, mu=None):
        assert U in self.source

        if self.vector:
            V = super().apply(U, mu=mu)
            return self.range.from_numpy(V.to_numpy())

        V = [self.matrix.dot(v._array) for v in U._list]

        if self.functional:
            return self.range.make_array(np.array(V)) if len(V) > 0 else self.range.empty()
        else:
            return self.range.make_array(V)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range

        if self.functional:
            U = super().apply_adjoint(V, mu=mu)
            return self.source.from_numpy(U.to_numpy())

        adj_op = NumpyMatrixOperator(self.matrix).H

        U = [adj_op.apply(adj_op.source.make_array(v._array)).to_numpy().ravel() for v in V._list]

        if self.vector:
            return self.source.make_array(np.array(U)) if len(U) > 0 else self.source.empty()
        else:
            return self.source.from_numpy(U)

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range
        assert not self.functional and not self.vector

        if V.dim == 0:
            if self.source.dim == 0 and least_squares:
                return self.source.make_array([np.zeros(0) for _ in range(len(V))])
            else:
                raise InversionError

        op = NumpyMatrixOperator(self.matrix, solver_options=self.solver_options)

        return self.source.make_array([op.apply_inverse(NumpyVectorSpace.make_array(v._array),
                                                        least_squares=least_squares).to_numpy().ravel()
                                       for v in V._list])

    def as_range_array(self, mu=None):
        assert not self.sparse
        return self.range.make_array(list(self.matrix.T.copy()))

    def as_source_array(self, mu=None):
        assert not self.sparse
        return self.source.make_array(list(self.matrix.copy()))

    def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
        lincomb = super().assemble_lincomb(operators, coefficients)
        if lincomb is None:
            return None
        else:
            return NumpyListVectorArrayMatrixOperator(lincomb.matrix, source_id=self.source.id, range_id=self.range.id,
                                                      solver_options=solver_options, name=name)
