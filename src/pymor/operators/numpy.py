# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Laier <m_laie01@uni-muenster.de>

"""This module provides the following |NumPy| based |Operators|:

  - |NumpyMatrixOperator| wraps a 2D |NumPy array| as a |Operator|.
  - |NumpyMatrixBasedOperator| should be used as base class for all |Operators|
    which assemble into a |NumpyMatrixOperator|.
  - |NumpyGenericOperator| wraps an arbitrary Python function between
    |NumPy arrays| as an |Operator|.
"""

from __future__ import absolute_import, division, print_function

from itertools import izip

import numpy as np
from scipy.sparse import issparse
from scipy.io import mmwrite, savemat

from pymor.core.defaults import defaults_sid
from pymor.core.exceptions import InversionError
from pymor.core.interfaces import abstractmethod
from pymor.la import numpysolvers
from pymor.la.numpyvectorarray import NumpyVectorArray, NumpyVectorSpace
from pymor.operators.basic import OperatorBase


class NumpyGenericOperator(OperatorBase):
    """Wraps an arbitrary Python function between |NumPy arrays| as a an |Operator|.

    Parameters
    ----------
    mapping
        The function to wrap. If `parameter_type` is `None`, the function is of
        the form `mapping(U)` and is expected to be vectorized. In particular::

            mapping(U).shape == U.shape[:-1] + (dim_range,).

        If `parameter_type` is not `None`, the function has to have the signature
        `mapping(U, mu)`.
    dim_source
        Dimension of the operator's source.
    dim_range
        Dimension of the operator's range.
    linear
        Set to `True` if the provided `mapping` is linear.
    parameter_type
        The |ParameterType| of the |Parameters| the mapping accepts.
    name
        Name of the operator.
    """

    def __init__(self, mapping, dim_source=1, dim_range=1, linear=False, parameter_type=None, name=None):
        self.source = NumpyVectorSpace(dim_source)
        self.range = NumpyVectorSpace(dim_range)
        self.name = name
        self._mapping = mapping
        self.linear = linear
        if parameter_type is not None:
            self.build_parameter_type(parameter_type, local_global=True)

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        assert U.check_ind(ind)
        U_array = U._array[:U._len] if ind is None else U._array[ind]
        if self.parametric:
            mu = self.parse_parameter(mu)
            return NumpyVectorArray(self._mapping(U_array, mu=mu), copy=False)
        else:
            return NumpyVectorArray(self._mapping(U_array), copy=False)


class NumpyMatrixBasedOperator(OperatorBase):
    """Base class for operators which assemble into a |NumpyMatrixOperator|.

    Attributes
    ----------
    sparse
        `True` if the operator assembles into a sparse matrix, `False` if the
        operator assembles into a dense matrix, `None` if unknown.
    """

    linear = True
    sparse = None

    @abstractmethod
    def _assemble(self, mu=None):
        pass

    def assemble(self, mu=None):
        """Assembles the operator for a given |Parameter|.

        Parameters
        ----------
        mu
            The |Parameter| for which to assemble the operator.

        Returns
        -------
        The assembled **parameter independent** |Operator|.
        """
        if hasattr(self, '_assembled_operator'):
            if self._defaults_sid != defaults_sid():
                self.logger.warn('Re-assembling since state of global defaults has changed.')
                op = self._assembled_operator = NumpyMatrixOperator(self._assemble())
                self._defaults_sid = defaults_sid()
                return op
            else:
                return self._assembled_operator
        elif self.parameter_type is None:
            op = self._assembled_operator = NumpyMatrixOperator(self._assemble())
            self._defaults_sid = defaults_sid()
            return op
        else:
            return NumpyMatrixOperator(self._assemble(self.parse_parameter(mu)))

    def apply(self, U, ind=None, mu=None):
        return self.assemble(mu).apply(U, ind=ind)

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        return self.assemble(mu).apply_adjoint(U, ind=ind, source_product=source_product, range_product=range_product)

    def as_vector(self, mu=None):
        return self.assemble(mu).as_vector()

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        return self.assemble(mu).apply_inverse(U, ind=ind, options=options)

    @property
    def invert_options(self):
        if self.sparse is None:
            raise ValueError('Sparsity unkown, assemble first.')
        else:
            return numpysolvers.invert_options(sparse=self.sparse)

    def export_matrix(self, filename, matrix_name=None, output_format='matlab', mu=None):
        """Save the matrix of the operator to a file.

        Parameters
        ----------
        filename
            Name of output file.
        matrix_name
            The name, the output matrix is given. (Comment field is used in
            case of Matrix Market output_format.) If `None`, the |Operator|'s `name`
            is used.
        output_format
            Output file format. Either `matlab` or `matrixmarket`.
        """
        assert output_format in {'matlab', 'matrixmarket'}
        matrix = self.assemble(mu)._matrix
        matrix_name = matrix_name or self.name
        if output_format is 'matlab':
            savemat(filename, {matrix_name: matrix})
        else:
            mmwrite(filename, matrix, comment=matrix_name)

    def __getstate__(self):
        d = self.__dict__.copy()
        if '_assembled_operator' in d:
            del d['_assembled_operator']
        return d


class NumpyMatrixOperator(NumpyMatrixBasedOperator):
    """Wraps a 2D |NumPy Array| as an |Operator|.

    Parameters
    ----------
    matrix
        The |NumPy array| which is to be wrapped.
    name
        Name of the operator.
    """

    calculate_sid = False

    def __init__(self, matrix, name=None):
        assert matrix.ndim <= 2
        if matrix.ndim == 1:
            matrix = np.reshape(matrix, (1, -1))
        self.source = NumpyVectorSpace(matrix.shape[1])
        self.range = NumpyVectorSpace(matrix.shape[0])
        self.name = name
        self._matrix = matrix
        self.sparse = issparse(matrix)
        self.calculate_sid = hasattr(matrix, 'sid')

    def _assemble(self, mu=None):
        pass

    def assemble(self, mu=None):
        return self

    def as_vector(self, mu=None):
        if self.source.dim != 1 and self.range.dim != 1:
            raise TypeError('This operator does not represent a vector or linear functional.')
        return NumpyVectorArray(self._matrix.ravel(), copy=True)

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        assert U.check_ind(ind)
        U_array = U._array[:U._len] if ind is None else U._array[ind]
        return NumpyVectorArray(self._matrix.dot(U_array.T).T, copy=False)

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        assert U in self.range
        assert U.check_ind(ind)
        assert source_product is None or source_product.source == source_product.range == self.source
        assert range_product is None or range_product.source == range_product.range == self.range
        if range_product:
            PrU = range_product.apply(U, ind=ind).data
        else:
            PrU = U.data if ind is None else U.data[ind]
        ATPrU = NumpyVectorArray(self._matrix.T.dot(PrU.T).T, copy=False)
        if source_product:
            return source_product.apply_inverse(ATPrU)
        else:
            return ATPrU

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        assert U in self.range
        assert U.check_ind(ind)
        if U.dim == 0:
            if (self.source.dim == 0
                    or isinstance(options, str) and options.startswith('least_squares')
                    or isinstance(options, dict) and options['type'].startswith('least_squares')):
                return NumpyVectorArray(np.zeros((U.len_ind(ind), self.source.dim)))
            else:
                raise InversionError
        U = U.data if ind is None else \
            U.data[ind] if hasattr(ind, '__len__') else U.data[ind:ind + 1]
        return NumpyVectorArray(numpysolvers.apply_inverse(self._matrix, U, options=options), copy=False)

    def projected_to_subbasis(self, dim_source=None, dim_range=None, name=None):
        """Project the operator to a subbasis.

        The purpose of this method is to further project an operator that has been
        obtained through :meth:`~pymor.operators.interfaces.OperatorInterface.projected`
        to subbases of the original projection bases, i.e. ::

            op.projected(s_basis, r_basis, prod).projected_to_subbasis(dim_source, dim_range)

        should be the same as ::

            op.projected(s_basis.copy(range(dim_source)), r_basis.copy(range(dim_range)), prod)

        For a |NumpyMatrixOperator| this amounts to extracting the upper-left
        (dim_range, dim_source) corner of the matrix it wraps.

        Parameters
        ----------
        dim_source
            Dimension of the source subbasis.
        dim_range
            Dimension of the range subbasis.

        Returns
        -------
        The projected |Operator|.
        """
        assert dim_source is None or dim_source <= self.source.dim
        assert dim_range is None or dim_range <= self.range.dim
        name = name or '{}_projected_to_subbasis'.format(self.name)
        # copy instead of just slicing the matrix to ensure contiguous memory
        return NumpyMatrixOperator(self._matrix[:dim_range, :dim_source].copy(), name=name)

    def assemble_lincomb(self, operators, coefficients, name=None):
        if not all(isinstance(op, NumpyMatrixOperator) for op in operators):
            return None

        if coefficients[0] == 1:
            matrix = operators[0]._matrix.copy()
        else:
            matrix = operators[0]._matrix * coefficients[0]
        for op, c in izip(operators[1:], coefficients[1:]):
            if c == 1:
                try:
                    matrix += op._matrix
                except NotImplementedError:
                    matrix = matrix + op._matrix
            elif c == -1:
                try:
                    matrix -= op._matrix
                except NotImplementedError:
                    matrix = matrix - op._matrix
            else:
                try:
                    matrix += (op._matrix * c)
                except NotImplementedError:
                    matrix = matrix + (op._matrix * c)
        return NumpyMatrixOperator(matrix)

    def __getstate__(self):
        if hasattr(self._matrix, 'factorization'):  # remove unplicklable SuperLU factorization
            del self._matrix.factorization
        return self.__dict__
