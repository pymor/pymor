# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.core.exceptions import InversionError
from pymor.operators.numpy import NumpyMatrixOperator, _apply_inverse
from pymor.vectorarrays.interfaces import VectorSpace
from pymor.vectorarrays.list import ListVectorArray, NumpyVector
from pymor.vectorarrays.numpy import NumpyVectorArray


class NumpyListVectorArrayMatrixOperator(NumpyMatrixOperator):
    """Variant of |NumpyMatrixOperator| using |ListVectorArray| instead of |NumpyVectorArray|."""

    def __init__(self, matrix, functional=False, vector=False, name=None):
        assert not (functional and vector)
        super(NumpyListVectorArrayMatrixOperator, self).__init__(matrix, name)
        if not vector:
            self.source = VectorSpace(ListVectorArray, (NumpyVector, matrix.shape[1]))
        if not functional:
            self.range = VectorSpace(ListVectorArray, (NumpyVector, matrix.shape[0]))
        self.functional = functional
        self.vector = vector

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        assert U.check_ind(ind)

        if self.vector:
            V = super(NumpyListVectorArrayMatrixOperator, self).apply(U, ind=ind, mu=mu)
            return ListVectorArray([NumpyVector(v, copy=False) for v in V.data],
                                   subtype=self.range.subtype)

        if ind is None:
            vectors = U._list
        elif isinstance(ind, Number):
            vectors = [U._list[ind]]
        else:
            vectors = (U._list[i] for i in ind)
        V = [self._matrix.dot(v._array) for v in vectors]

        if self.functional:
            return NumpyVectorArray(V) if len(V) > 0 else self.range.empty()
        else:
            return ListVectorArray([NumpyVector(v, copy=False) for v in V], subtype=self.range.subtype)

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        raise NotImplementedError

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        assert U in self.range
        assert U.check_ind(ind)
        assert not self.functional and not self.vector

        if U.dim == 0:
            if (self.source.dim == 0
                    or isinstance(options, str) and options.startswith('least_squares')
                    or isinstance(options, dict) and options['type'].startswith('least_squares')):
                return ListVectorArray([NumpyVector(np.zeros(0), copy=False) for _ in range(U.len_ind(ind))],
                                       subtype=self.source.subtype)
            else:
                raise InversionError

        if ind is None:
            vectors = U._list
        elif isinstance(ind, Number):
            vectors = [U._list[ind]]
        else:
            vectors = (U._list[i] for i in ind)

        return ListVectorArray([NumpyVector(_apply_inverse(self._matrix, v._array, options=options), copy=False)
                                for v in vectors],
                               subtype=self.source.subtype)

    def as_vector(self, mu=None):
        if self.source.dim != 1 and self.range.dim != 1:
            raise TypeError('This operator does not represent a vector or linear functional.')
        return ListVectorArray([NumpyVector(self._matrix.ravel(), copy=True)])

    def assemble_lincomb(self, operators, coefficients, name=None):
        lincomb = super(NumpyListVectorArrayMatrixOperator, self).assemble_lincomb(operators, coefficients)
        if lincomb is None:
            return None
        else:
            return NumpyListVectorArrayMatrixOperator(lincomb._matrix, name=name)
