# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
from scipy.sparse import issparse

from pymor.la import NumpyVectorArray
from pymor.operators.interfaces import OperatorInterface, LinearOperatorInterface


class NumpyGenericOperator(OperatorInterface):
    '''Wraps an apply function as a proper discrete operator.

    Parameters
    ----------
    mapping
        The function to wrap. If parameter_type is None, mapping is called with
        the DOF-vector U as only argument. If parameter_type is not None, mapping
        is called with the arguments U and mu.
    dim_source
        Dimension of the operator's source.
    dim_range
        Dimension of the operator's range.
    parameter_type
        Type of the parameter that mapping expects or None.
    name
        Name of the operator.
    '''

    type_source = type_range = NumpyVectorArray

    def __init__(self, mapping, dim_source=1, dim_range=1, parameter_type=None, name=None):
        super(NumpyGenericOperator, self).__init__()
        self.dim_source = dim_source
        self.dim_range = dim_range
        self.name = name
        self._mapping = mapping
        if parameter_type is not None:
            self.build_parameter_type(parameter_type)
            self._with_mu = True
        else:
            self._with_mu = False
        self.lock()

    def apply(self, U, ind=None, mu=None):
        assert isinstance(U, NumpyVectorArray)
        assert U.dim == self.dim_source
        U_array = U._array[:U._len] if ind is None else U._array[ind]
        if self._with_mu:
            mu = self.map_parameter(mu)
            return NumpyVectorArray(self._mapping(U_array, mu), copy=False)
        else:
            assert mu is None
            return NumpyVectorArray(self._mapping(U_array), copy=False)


class NumpyLinearOperator(LinearOperatorInterface):
    '''Wraps a matrix as a proper linear discrete operator.

    The resulting operator will be parameter independent.

    Parameters
    ----------
    matrix
        The Matrix which is to be wrapped.
    name
        Name of the operator.
    '''

    type_source = type_range = NumpyVectorArray
    assembled = True

    def __init__(self, matrix, name=None):
        super(NumpyLinearOperator, self).__init__()
        assert matrix.ndim <= 2
        if matrix.ndim == 1:
            matrix = np.reshape(matrix, (1, -1))
        self.dim_source = matrix.shape[1]
        self.dim_range = matrix.shape[0]
        self.name = name
        self._matrix = matrix
        self.sparse = issparse(matrix)
        self.lock()

    def as_vector_array(self):
        return NumpyVectorArray(self._matrix, copy=True)

    def _assemble(self, mu=None):
        assert mu is None
        return self

    def assemble(self, mu=None, force=False):
        assert mu is None
        return self

    def apply(self, U, ind=None, mu=None):
        assert isinstance(U, NumpyVectorArray)
        assert mu is None
        U_array = U._array[:U._len] if ind is None else U._array[ind]
        return NumpyVectorArray(self._matrix.dot(U_array.T).T, copy=False)

    def __add__(self, other):
        if isinstance(other, NumpyLinearOperator):
            return NumpyLinearOperator(self._matrix + other._matrix)
        elif isinstance(other, Number):
            return NumpyLinearOperator(self._matrix + other)
        else:
            return NotImplemented

    __radd__ = __add__

    def __mul__(self, other):
        return NumpyLinearOperator(self._matrix * other)
