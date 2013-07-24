# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from itertools import izip
from numbers import Number

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import bicgstab

from pymor.core import defaults
from pymor.la import NumpyVectorArray
from pymor.operators import OperatorBase, MatrixBasedOperatorBase, LincombOperatorBase, LincombOperator


class NumpyMatrixBasedOperator(MatrixBasedOperatorBase):

    type_source = type_range = NumpyVectorArray

    @staticmethod
    def lincomb(operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
        if not all(isinstance(op, NumpyMatrixBasedOperator) for op in operators):
            return LincombOperator(operators, coefficients, num_coefficients=num_coefficients,
                                   coefficients_name=coefficients_name, name=name)
        else:
            return NumpyLincombMatrixOperator(operators, coefficients, num_coefficients=num_coefficients,
                                              coefficients_name=coefficients_name, name=name)

    @property
    def invert_options(self):
        if self.sparse is None:
            raise ValueError('Sparsity unkown, assemble first.')
        elif self.sparse:
            return OrderedDict((('bicgstab', {'type': 'bicgstab',
                                              'tol': defaults.bicgstab_tol,
                                              'maxiter': defaults.bicgstab_maxiter}),))
        else:
            return OrderedDict((('linsolve', {'type': 'linsolve'}),))

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        return self.assemble(mu).apply_inverse(U, ind=ind, options=options)


class NumpyGenericOperator(OperatorBase):
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
            self.build_parameter_type(parameter_type, local_global=True)
        self.lock()

    def apply(self, U, ind=None, mu=None):
        assert isinstance(U, NumpyVectorArray)
        assert U.dim == self.dim_source
        mu = self.parse_parameter(mu)
        U_array = U._array[:U._len] if ind is None else U._array[ind]
        if self.parametric:
            return NumpyVectorArray(self._mapping(U_array, mu=mu), copy=False)
        else:
            return NumpyVectorArray(self._mapping(U_array), copy=False)


class NumpyMatrixOperator(NumpyMatrixBasedOperator):
    '''Wraps a matrix as a proper linear discrete operator.

    The resulting operator will be parameter independent.

    Parameters
    ----------
    matrix
        The Matrix which is to be wrapped.
    name
        Name of the operator.
    '''

    assembled = True

    def __init__(self, matrix, name=None):
        super(NumpyMatrixOperator, self).__init__()
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
        mu = self.parse_parameter(mu)
        return self

    def assemble(self, mu=None, force=False):
        mu = self.parse_parameter(mu)
        return self

    def apply(self, U, ind=None, mu=None):
        assert isinstance(U, NumpyVectorArray)
        mu = self.parse_parameter(mu)
        U_array = U._array[:U._len] if ind is None else U._array[ind]
        return NumpyVectorArray(self._matrix.dot(U_array.T).T, copy=False)

    def apply_inverse(self, U, ind=None, mu=None, options=None):

        def check_options(options, sparse):
            if not options:
                return True
            assert 'type' in options
            if sparse:
                assert options['type'] == 'bicgstab'
                assert options.viewkeys() <= set(('type', 'tol', 'maxiter'))
            else:
                assert options['type'] == 'solve'
                assert options.viewkeys() <= set(('type',))
            return True

        if options is None:
            options = {}
        elif isinstance(options, str):
            options = {'type': options}

        assert isinstance(U, NumpyVectorArray)
        assert self.dim_range == U.dim
        assert check_options(options, self.sparse)

        U = U._array[:U._len] if ind is None else U._array[ind]
        if U.shape[1] == 0:
            return NumpyVectorArray(U)
        R = np.empty((len(U), self.dim_source))

        if self.sparse:
            tol =  options.get('tol', defaults.bicgstab_tol)
            maxiter = options.get('maxiter', defaults.bicgstab_maxiter)
            for i, UU in enumerate(U):
                R[i], _ = bicgstab(self._matrix, UU, tol=tol, maxiter=maxiter)
        else:
            for i, UU in enumerate(U):
                R[i] = np.linalg.solve(self._matrix, UU)

        return NumpyVectorArray(R)


class NumpyLincombMatrixOperator(NumpyMatrixBasedOperator, LincombOperatorBase):

    def __init__(self, operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
        assert all(isinstance(op, NumpyMatrixBasedOperator) for op in operators)
        LincombOperatorBase.__init__(self, operators=operators, coefficients=coefficients,
                                     num_coefficients=num_coefficients,
                                     coefficients_name=coefficients_name, name=name)
        self.sparse = all(op.sparse for op in operators)
        self.lock()

    def _assemble(self, mu=None):
        mu = self.parse_parameter(mu)
        ops = [op.assemble(mu) for op in self.operators]
        coeffs = self.evaluate_coefficients(mu)
        if self.sparse:
            matrix = sum(op._matrix * c for op, c in izip(ops, coeffs))
        else:
            matrix = ops[0]._matrix.copy()
            if coeffs[0] != 1:
                matrix *= coeffs[0]
            for op, c in izip(ops[1:], coeffs[1:]):
                matrix += (op._matrix * c)
        return NumpyMatrixOperator(matrix)
