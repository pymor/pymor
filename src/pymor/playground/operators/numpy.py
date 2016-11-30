# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.exceptions import InversionError
from pymor.operators.numpy import NumpyMatrixOperator, _apply_inverse
from pymor.vectorarrays.list import NumpyListVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace


class NumpyListVectorArrayMatrixOperator(NumpyMatrixOperator):
    """Variant of |NumpyMatrixOperator| using |ListVectorArray| instead of |NumpyVectorArray|."""

    def __init__(self, matrix, functional=False, vector=False, source_id=None, range_id=None,
                 solver_options=None, name=None):
        assert not (functional and vector)
        super().__init__(matrix, source_id=source_id, range_id=range_id, solver_options=solver_options, name=name)
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
            return self.range.from_data(V.data)

        V = [self._matrix.dot(v._array) for v in U._list]

        if self.functional:
            return self.range.make_array(np.array(V)) if len(V) > 0 else self.range.empty()
        else:
            return self.range.make_array(V)

    def apply_adjoint(self, U, mu=None, source_product=None, range_product=None):
        raise NotImplementedError

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range
        assert not self.functional and not self.vector

        if V.dim == 0:
            if self.source.dim == 0 and least_squares:
                return self.source.make_array([np.zeros(0) for _ in range(len(V))])
            else:
                raise InversionError

        options = (self.solver_options.get('inverse') if self.solver_options else
                   'least_squares' if least_squares else
                   None)

        if options and not least_squares:
            solver_type = options if isinstance(options, str) else options['type']
            if solver_type.startswith('least_squares'):
                self.logger.warn('Least squares solver selected but "least_squares == False"')

        try:
            return self.source.make_array([_apply_inverse(self._matrix, v._array.reshape((1, -1)),
                                                          options=options).ravel()
                                           for v in V._list])
        except InversionError as e:
            if least_squares and options:
                solver_type = options if isinstance(options, str) else options['type']
                if not solver_type.startswith('least_squares'):
                    msg = str(e) \
                        + '\nNote: linear solver was selected for solving least squares problem (maybe not invertible?)'
                    raise InversionError(msg)
            raise e

    def as_vector(self, mu=None):
        if self.range == NumpyVectorSpace(1):
            return self.source.make_array([self._matrix.ravel()])
        elif self.source == NumpyVectorSpace(1):
            return self.range.make_array([self._matrix.ravel()])
        else:
            raise TypeError('This operator does not represent a vector or linear functional.')

    def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
        lincomb = super().assemble_lincomb(operators, coefficients)
        if lincomb is None:
            return None
        else:
            return NumpyListVectorArrayMatrixOperator(lincomb._matrix, source_id=self.source.id, range_id=self.range.id,
                                                      solver_options=solver_options, name=name)
