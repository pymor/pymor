# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Laier <m_laie01@uni-muenster.de>

"""This module provides some |NumPy| based |Operators| as well as base classes
providing some common functionality for the implementation of new |Operators|.

There are three |NumPy|-based |Operators| of interest:

  - |NumpyMatrixOperator| wraps a 2D |NumPy array| as a proper |Operator|.
  - |NumpyMatrixBasedOperator| should be used as base class for all |Operators|
    which assemble into a |NumpyMatrixOperator|.
  - |NumpyGenericOperator| wraps an arbitrary Python function between
    |NumPy arrays| as an |Operator|.

If you are developing new |Operators| not based on |NumPy arrays|, you should
consider deriving from :class:`OperatorBase`.
"""

from __future__ import absolute_import, division, print_function

from itertools import izip
from numbers import Number

import numpy as np
from scipy.sparse import issparse
from scipy.io import mmwrite, savemat

from pymor.core import abstractmethod
from pymor.core.defaults import defaults_sid
from pymor.core.exceptions import InversionError
from pymor.la import genericsolvers
from pymor.la.interfaces import VectorArrayInterface
from pymor.la.numpyvectorarray import NumpyVectorArray, NumpyVectorSpace
from pymor.la import numpysolvers
from pymor.operators.interfaces import OperatorInterface
from pymor.parameters import ParameterFunctionalInterface


class OperatorBase(OperatorInterface):
    """Base class for |Operators| providing some default implementations."""

    def apply2(self, V, U, pairwise, U_ind=None, V_ind=None, mu=None, product=None):
        mu = self.parse_parameter(mu)
        assert isinstance(V, VectorArrayInterface)
        assert isinstance(U, VectorArrayInterface)
        U_ind = None if U_ind is None else np.array(U_ind, copy=False, dtype=np.int, ndmin=1)
        V_ind = None if V_ind is None else np.array(V_ind, copy=False, dtype=np.int, ndmin=1)
        if pairwise:
            lu = len(U_ind) if U_ind is not None else len(U)
            lv = len(V_ind) if V_ind is not None else len(V)
            assert lu == lv
        AU = self.apply(U, ind=U_ind, mu=mu)
        if product is not None:
            AU = product.apply(AU)
        return V.dot(AU, ind=V_ind, pairwise=pairwise)

    def jacobian(self, U, mu=None):
        if self.linear:
            if self.parametric:
                return self.assemble(mu)
            else:
                return self
        else:
            raise NotImplementedError

    @staticmethod
    def lincomb(operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
        op = LincombOperator(operators, coefficients, num_coefficients, coefficients_name, name=None)
        if op.parametric:
            return op
        else:
            return op.assemble()

    def assemble(self, mu=None):
        if self.parametric:
            from pymor.operators.constructions import FixedParameterOperator

            return FixedParameterOperator(self, mu=mu, name=self.name + '_assembled')
        else:
            return self

    def __sub__(self, other):
        if isinstance(other, Number):
            assert other == 0.
            return self
        return self.lincomb([self, other], [1, -1])

    def __add__(self, other):
        if isinstance(other, Number):
            assert other == 0.
            return self
        return self.lincomb([self, other], [1, 1])

    __radd__ = __add__

    def __mul__(self, other):
        assert isinstance(other, Number)
        return self.lincomb([self], [other])

    def __str__(self):
        return '{}: R^{} --> R^{}  (parameter type: {}, class: {})'.format(
            self.name, self.source.dim, self.range.dim, self.parameter_type,
            self.__class__.__name__)

    @property
    def invert_options(self):
        if self.linear:
            return genericsolvers.invert_options()
        else:
            return None

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        if self.parametric:
            return self.assemble(mu).apply_inverse(U, ind=ind, options=options)
        else:
            return genericsolvers.apply_inverse(self, U.copy(ind), options=options)

    def as_vector(self, mu=None):
        if not self.linear:
            raise TypeError('This nonlinear operator does not represent a vector or linear functional.')
        elif self.source.dim == 1 and self.source.type is NumpyVectorArray:
            return self.apply(NumpyVectorArray(1), mu=mu)
        elif self.range.dim == 1 and self.range.type is NumpyVectorArray:
            raise NotImplementedError
        else:
            raise TypeError('This operator does not represent a vector or linear functional.')

    def projected(self, source_basis, range_basis, product=None, name=None):
        name = name or '{}_projected'.format(self.name)
        if self.linear:
            if self.parametric:
                self.logger.warn('Using inefficient generic linear projection operator')
                # Since the bases are not immutable and we do not own them,
                # the ProjectedLinearOperator will have to create copies of them.
                return ProjectedLinearOperator(self, source_basis, range_basis, product, copy=True, name=name)
            else:
                # Here we do not need copies since the operator is immediately thrown away.
                return (ProjectedLinearOperator(self, source_basis, range_basis, product, copy=False, name=name)
                        .assemble())
        else:
            self.logger.warn('Using inefficient generic projection operator')
            return ProjectedOperator(self, source_basis, range_basis, product, copy=True, name=name)


class NumpyGenericOperator(OperatorBase):
    """Wraps an arbitrary Python function between |NumPy arrays| as a proper
    |Operator|.

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
        The |ParameterType| the mapping accepts.
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

    def apply_transposed(self, U, ind=None, mu=None):
        return self.assemble(mu).apply_transposed(U, ind=ind)

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
        """Save matrix of operator to a file.

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
    """Wraps a 2D |NumPy Array| as a proper |Operator|.

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
        assert isinstance(U, NumpyVectorArray)
        U_array = U._array[:U._len] if ind is None else U._array[ind]
        return NumpyVectorArray(self._matrix.dot(U_array.T).T, copy=False)

    def apply_transposed(self, U, ind=None, mu=None):
        assert isinstance(U, NumpyVectorArray)
        U_array = U._array[:U._len] if ind is None else U._array[ind]
        return NumpyVectorArray(self._matrix.T.dot(U_array.T).T, copy=False)

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        assert U in self.range
        U = U._array[:U._len] if ind is None else U._array[ind]
        if U.shape[1] == 0:
            return NumpyVectorArray(U)
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
        return NumpyMatrixOperator(self._matrix[:dim_range, :dim_source], name=name)

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


class ProjectedOperator(OperatorBase):
    """Genric |Operator| for representing the projection of an |Operator| to a subspace.

    This class is not intended to be instantiated directly. Instead, you should use
    the :meth:`~pymor.operators.interfaces.OperatorInterface.projected` method of the given
    |Operator|.

    Parameters
    ----------
    operator
        The |Operator| to project.
    source_basis
        See :meth:`~pymor.operators.interfaces.OperatorInterface.projected`.
    range_basis
        See :meth:`~pymor.operators.interfaces.OperatorInterface.projected`.
    product
        See :meth:`~pymor.operators.interfaces.OperatorInterface.projected`.
    copy
        If `True`, make a copy of the provided `source_basis` and `range_basis`. This is
        usually necessary, as |VectorArrays| are not immutable.
    name
        Name of the projected operator.
    """

    linear = False

    def __init__(self, operator, source_basis, range_basis, product=None, copy=True, name=None):
        assert isinstance(operator, OperatorInterface)
        assert source_basis is None and issubclass(operator.source.type, NumpyVectorArray) \
            or source_basis in operator.source
        assert range_basis is None and issubclass(operator.range.type, NumpyVectorArray) \
            or range_basis in operator.range
        assert product is None \
            or (isinstance(product, OperatorInterface)
                and range_basis is not None
                and operator.range == product.source
                and product.range == product.source)
        self.build_parameter_type(inherits=(operator,))
        self.source = NumpyVectorSpace(len(source_basis) if operator.source.dim > 0 else 0)
        self.range = NumpyVectorSpace(len(range_basis) if range_basis is not None else operator.range.dim)
        self.name = name
        self.operator = operator
        self.source_basis = source_basis.copy() if source_basis is not None and copy else source_basis
        self.range_basis = range_basis.copy() if range_basis is not None and copy else range_basis
        self.product = product

    def apply(self, U, ind=None, mu=None):
        mu = self.parse_parameter(mu)
        if self.source_basis is None:
            if self.range_basis is None:
                return self.operator.apply(U, ind=ind, mu=mu)
            elif self.product is None:
                return NumpyVectorArray(self.operator.apply2(self.range_basis, U, U_ind=ind, mu=mu, pairwise=False).T)
            else:
                V = self.operator.apply(U, ind=ind, mu=mu)
                return NumpyVectorArray(self.product.apply2(V, self.range_basis, pairwise=False))
        else:
            U_array = U._array[:U._len] if ind is None else U._array[ind]
            UU = self.source_basis.lincomb(U_array)
            if self.range_basis is None:
                return self.operator.apply(UU, mu=mu)
            elif self.product is None:
                return NumpyVectorArray(self.operator.apply2(self.range_basis, UU, mu=mu, pairwise=False).T)
            else:
                V = self.operator.apply(UU, mu=mu)
                return NumpyVectorArray(self.product.apply2(V, self.range_basis, pairwise=False))

    def projected_to_subbasis(self, dim_source=None, dim_range=None, name=None):
        """See :meth:`NumpyMatrixOperator.projected_to_subbasis`."""
        assert dim_source is None or dim_source <= self.source.dim
        assert dim_range is None or dim_range <= self.range.dim
        assert dim_source is None or self.source_basis is not None, 'not implemented'
        assert dim_range is None or self.range_basis is not None, 'not implemented'
        name = name or '{}_projected_to_subbasis'.format(self.name)
        source_basis = self.source_basis if dim_source is None \
            else self.source_basis.copy(ind=range(dim_source))
        range_basis = self.range_basis if dim_range is None \
            else self.range_basis.copy(ind=range(dim_range))
        return ProjectedOperator(self.operator, source_basis, range_basis, product=None, copy=False, name=name)

    def jacobian(self, U, mu=None):
        assert len(U) == 1
        mu = self.parse_parameter(mu)
        if self.source_basis is None:
            J = self.operator.jacobian(U, mu=mu)
        else:
            J = self.operator.jacobian(self.source_basis.lincomb(U.data), mu=mu)
        return ProjectedLinearOperator(J, source_basis=self.source_basis, range_basis=self.range_basis,
                                       product=self.product, copy=False, name=self.name + '_jacobian').assemble()


class ProjectedLinearOperator(NumpyMatrixBasedOperator):
    """Genric |Operator| for representing the projection of a linear |Operator| to a subspace.

    This class is not intended to be instantiated directly. Instead, you should use
    the :meth:`~pymor.operators.interfaces.OperatorInterface.projected` method of the given
    |Operator|.

    Parameters
    ----------
    operator
        The |Operator| to project.
    source_basis
        See :meth:`~pymor.operators.interfaces.OperatorInterface.projected`.
    range_basis
        See :meth:`~pymor.operators.interfaces.OperatorInterface.projected`.
    product
        See :meth:`~pymor.operators.interfaces.OperatorInterface.projected`.
    copy
        If `True`, make a copy of the provided `source_basis` and `range_basis`. This is
        usually necessary, as |VectorArrays| are not immutable.
    name
        Name of the projected operator.
    """

    sparse = False

    def __init__(self, operator, source_basis, range_basis, product=None, name=None, copy=True):
        assert isinstance(operator, OperatorInterface)
        assert source_basis is None and issubclass(operator.source.type, NumpyVectorArray) \
            or source_basis in operator.source
        assert range_basis is None and issubclass(operator.range.type, NumpyVectorArray) \
            or range_basis in operator.range
        assert product is None \
            or (isinstance(product, OperatorInterface)
                and range_basis is not None
                and operator.range == product.source
                and product.range == product.source)
        assert operator.linear
        self.build_parameter_type(inherits=(operator,))
        self.source = NumpyVectorSpace(len(source_basis) if source_basis is not None else operator.source.dim)
        self.range = NumpyVectorSpace(len(range_basis) if range_basis is not None else operator.range.dim)
        self.name = name
        self.operator = operator
        self.source_basis = source_basis.copy() if source_basis is not None and copy else source_basis
        self.range_basis = range_basis.copy() if range_basis is not None and copy else range_basis
        self.product = product

    def _assemble(self, mu=None):
        if self.source_basis is None:
            if self.range_basis is None:
                return self.operator.assemble(mu=mu)
            elif self.product is None:
                return self.operator.apply2(self.range_basis,
                                            NumpyVectorArray(np.eye(self.operator.source.dim)),
                                            pairwise=False, mu=mu)
            else:
                V = self.operator.apply(NumpyVectorArray(np.eye(self.operator.source.dim)), mu=mu)
                return self.product.apply2(self.range_basis, V, pairwise=False)
        else:
            if self.range_basis is None:
                return self.operator.apply(self.source_basis, mu=mu).data.T
            elif self.product is None:
                return self.operator.apply2(self.range_basis, self.source_basis, mu=mu, pairwise=False)
            else:
                V = self.operator.apply(self.source_basis, mu=mu)
                return self.product.apply2(self.range_basis, V, pairwise=False)

    def projected_to_subbasis(self, dim_source=None, dim_range=None, name=None):
        """See :meth:`NumpyMatrixOperator.projected_to_subbasis`."""
        assert dim_source is None or dim_source <= self.source.dim
        assert dim_range is None or dim_range <= self.dim_range
        assert dim_source is None or self.source_basis is not None, 'not implemented'
        assert dim_range is None or self.range_basis is not None, 'not implemented'
        name = name or '{}_projected_to_subbasis'.format(self.name)
        source_basis = self.source_basis if dim_source is None \
            else self.source_basis.copy(ind=range(dim_source))
        range_basis = self.range_basis if dim_range is None \
            else self.range_basis.copy(ind=range(dim_range))
        return ProjectedLinearOperator(self.operator, source_basis, range_basis, product=None, copy=False, name=name)


class LincombOperator(OperatorBase):
    """A generic |LincombOperator| representing a linear combination of arbitrary |Operators|.

    Parameters
    ----------
    operators
        List of |Operators| whose linear combination is formed.
    coefficients
        `None` or a list of linear coefficients.
    num_coefficients
        If `coefficients` is `None`, the number of linear coefficients (starting
        at index 0) which are given by the |Parameter| component with name
        `'coefficients_name'`. The missing coefficients are set to `1`.
    coefficients_name
        If `coefficients` is `None`, the name of the |Parameter| component providing
        the linear coefficients.
    name
        Name of the operator.
    """

    def __init__(self, operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
        assert coefficients is None or len(operators) == len(coefficients)
        assert len(operators) > 0
        assert all(isinstance(op, OperatorInterface) for op in operators)
        assert coefficients is None or all(isinstance(c, (ParameterFunctionalInterface, Number)) for c in coefficients)
        assert all(op.source == operators[0].source for op in operators[1:])
        assert all(op.range == operators[0].range for op in operators[1:])
        assert coefficients is None or num_coefficients is None
        assert coefficients is None or coefficients_name is None
        assert coefficients is not None or coefficients_name is not None
        assert coefficients_name is None or isinstance(coefficients_name, str)
        self.source = operators[0].source
        self.range = operators[0].range
        self.operators = operators
        self.coefficients = coefficients
        self.coefficients_name = coefficients_name
        self.linear = all(op.linear for op in operators)
        self.name = name
        if coefficients is None:
            self.num_coefficients = num_coefficients if num_coefficients is not None else len(operators)
            self.pad_coefficients = len(operators) - self.num_coefficients
            self.build_parameter_type({'coefficients': self.num_coefficients}, inherits=list(operators),
                                      global_names={'coefficients': coefficients_name})
        else:
            self.build_parameter_type(inherits=list(operators) +
                                      [f for f in coefficients if isinstance(f, ParameterFunctionalInterface)])

    def evaluate_coefficients(self, mu):
        """Compute the linear coefficients of the linear combination for a given parameter.

        Parameters
        ----------
        mu
            |Parameter| for which to compute the linear coefficients.

        Returns
        -------
        List of linear coefficients.
        """
        mu = self.parse_parameter(mu)
        if self.coefficients is None:
            if self.pad_coefficients:
                return np.concatenate((self.local_parameter(mu)['coefficients'], np.ones(self.pad_coefficients)))
            else:
                return self.local_parameter(mu)['coefficients']

        else:
            return np.array([c.evaluate(mu) if hasattr(c, 'evaluate') else c for c in self.coefficients])

    def apply(self, U, ind=None, mu=None):
        mu = self.parse_parameter(mu)
        coeffs = self.evaluate_coefficients(mu)
        Vs = [op.apply(U, ind=ind, mu=mu) for op in self.operators]
        R = Vs[0]
        R.scal(coeffs[0])
        for V, c in izip(Vs[1:], coeffs[1:]):
            R.axpy(c, V)
        return R

    def assemble(self, mu=None):
        operators = [op.assemble(mu) for op in self.operators]
        coefficients = self.evaluate_coefficients(mu)
        op = operators[0].assemble_lincomb(operators, coefficients, name=self.name + '_assembled')
        if op is None:
            return LincombOperator(operators, coefficients, name=self.name + '_assembled')
        else:
            return op

    def jacobian(self, U, mu=None):
        jacobians = [op.jacobian(U, mu) for op in self.operators]
        coefficients = self.evaluate_coefficients(mu)
        jac = jacobians[0].assemble_lincomb(jacobians, coefficients, name=self.name + '_jacobian')
        if jac is None:
            return LincombOperator(jacobians, coefficients, name=self.name + '_jacobian')
        else:
            return jac

    def as_vector(self, mu=None):
        coefficients = self.evaluate_coefficients(mu)
        vectors = [op.as_vector(mu) for op in self.operators]
        R = vectors[0]
        R.scal(coefficients[0])
        for c, v in izip(coefficients[1:], vectors[1:]):
            R.axpy(c, v)
        return R

    def projected(self, source_basis, range_basis, product=None, name=None):
        proj_operators = [op.projected(source_basis=source_basis, range_basis=range_basis, product=product)
                          for op in self.operators]
        name = name or self.name + '_projected'
        num_coefficients = getattr(self, 'num_coefficients', None)
        return LincombOperator(operators=proj_operators, coefficients=self.coefficients,
                               num_coefficients=num_coefficients,
                               coefficients_name=self.coefficients_name, name=name)

    def projected_to_subbasis(self, dim_source=None, dim_range=None, name=None):
        """See :meth:`NumpyMatrixOperator.projected_to_subbasis`."""
        assert dim_source is None or dim_source <= self.source.dim
        assert dim_range is None or dim_range <= self.range.dim
        proj_operators = [op.projected_to_subbasis(dim_source=dim_source, dim_range=dim_range)
                          for op in self.operators]
        name = name or '{}_projected_to_subbasis'.format(self.name)
        num_coefficients = getattr(self, 'num_coefficients', None)
        return LincombOperator(operators=proj_operators, coefficients=self.coefficients,
                               num_coefficients=num_coefficients,
                               coefficients_name=self.coefficients_name, name=name)
