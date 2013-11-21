# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from itertools import izip
from numbers import Number

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import bicgstab

from pymor import defaults
from pymor.core import abstractmethod
from pymor.core.exceptions import InversionError
from pymor.la.interfaces import VectorArrayInterface
from pymor.la.numpyvectorarray import NumpyVectorArray
from pymor.operators.interfaces import OperatorInterface, LincombOperatorInterface
from pymor.parameters import ParameterFunctionalInterface


class OperatorBase(OperatorInterface):

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

    @staticmethod
    def lincomb(operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
        return LincombOperator(operators, coefficients, num_coefficients, coefficients_name, name=None)

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
            self.name, self.dim_source, self.dim_range, self.parameter_type,
            self.__class__.__name__)

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        raise InversionError('No inversion algorithm available.')

    def as_vector(self, mu=None):
        if not self.linear:
            raise TypeError('This is nonlinear operator does not represent a vector or linear functional.')
        elif self.dim_source == 1 and self.type_source is NumpyVectorArray:
            return self.apply(NumpyVectorArray(1), mu)
        elif self.dim_range == 1 and self.type_range is NumpyVectorArray:
            raise NotImplementedError
        else:
            raise TypeError('This is operator does not represent a vector or linear functional.')

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


class MatrixBasedOperatorBase(OperatorBase):

    linear = True
    sparse = None

    _assembled = False

    @property
    def assembled(self):
        return self._assembled

    @abstractmethod
    def _assemble(self, mu=None):
        pass

    def assemble(self, mu=None):
        '''Assembles the matrix of the operator for given parameter.

        Parameters
        ----------
        mu
            The parameter for which to assemble the operator.

        Returns
        -------
        The assembled parameter independent `MatrixBasedOperator`.
        '''
        if self._assembled:
            assert self.check_parameter(mu)
            return self._last_op
        elif self.parameter_type is None:
            assert self.check_parameter(mu)
            self._last_op = self._assemble()
            self._assembled = True
            return self._last_op
        else:
            mu_s = self.strip_parameter(mu)
            if mu_s == self._last_mu:
                return self._last_op
            else:
                self._last_mu = mu_s.copy()
                self._last_op = self._assemble(mu)
                return self._last_op

    def apply(self, U, ind=None, mu=None):
        if not self._assembled:
            return self.assemble(mu).apply(U, ind=ind)
        elif self._last_op is not self:
            return self._last_op.apply(U, ind=ind)
        else:
            raise NotImplementedError

    def as_vector(self, mu=None):
        if not self._assembled:
            return self.assemble(mu).as_vector()
        elif self._last_op is not self:
            return self._last_op.as_vector()
        else:
            return super(MatrixBasedOperatorBase, self).as_vector(self, mu)

    _last_mu = None
    _last_op = None


class LincombOperatorBase(OperatorBase, LincombOperatorInterface):

    def __init__(self, operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
        assert coefficients is None or len(operators) == len(coefficients)
        assert len(operators) > 0
        assert all(isinstance(op, OperatorInterface) for op in operators)
        assert coefficients is None or all(isinstance(c, (ParameterFunctionalInterface, Number)) for c in coefficients)
        assert all(op.dim_source == operators[0].dim_source for op in operators[1:])
        assert all(op.dim_range == operators[0].dim_range for op in operators[1:])
        assert all(op.type_source == operators[0].type_source for op in operators[1:])
        assert all(op.type_range == operators[0].type_range for op in operators[1:])
        assert coefficients is None or num_coefficients is None
        assert coefficients is None or coefficients_name is None
        assert coefficients is not None or coefficients_name is not None
        assert coefficients_name is None or isinstance(coefficients_name, str)
        self.dim_source = operators[0].dim_source
        self.dim_range = operators[0].dim_range
        self.type_source = operators[0].type_source
        self.type_range = operators[0].type_range
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
        mu = self.parse_parameter(mu)
        if self.coefficients is None:
            if self.pad_coefficients:
                return np.concatenate((self.local_parameter(mu)['coefficients'], np.ones(self.pad_coefficients)))
            else:
                return self.local_parameter(mu)['coefficients']

        else:
            return np.array([c.evaluate(mu) if hasattr(c, 'evaluate') else c for c in self.coefficients])

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
        name = name or '{}_projected'.format(self.name)
        num_coefficients = getattr(self, 'num_coefficients', None)
        return type(proj_operators[0]).lincomb(operators=proj_operators, coefficients=self.coefficients,
                                               num_coefficients=num_coefficients,
                                               coefficients_name=self.coefficients_name, name=name)

    def projected_to_subbasis(self, dim_source=None, dim_range=None, name=None):
        assert dim_source is None or dim_source <= self.dim_source
        assert dim_range is None or dim_range <= self.dim_range
        proj_operators = [op.projected_to_subbasis(dim_source=dim_source, dim_range=dim_range)
                          for op in self.operators]
        name = name or '{}_projected_to_subbasis'.format(self.name)
        num_coefficients = getattr(self, 'num_coefficients', None)
        return type(proj_operators[0]).lincomb(operators=proj_operators, coefficients=self.coefficients,
                                               num_coefficients=num_coefficients,
                                               coefficients_name=self.coefficients_name, name=name)


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
    linear = False

    def __init__(self, mapping, dim_source=1, dim_range=1, parameter_type=None, name=None):
        self.dim_source = dim_source
        self.dim_range = dim_range
        self.name = name
        self._mapping = mapping
        if parameter_type is not None:
            self.build_parameter_type(parameter_type, local_global=True)

    def apply(self, U, ind=None, mu=None):
        assert isinstance(U, NumpyVectorArray)
        assert U.dim == self.dim_source
        U_array = U._array[:U._len] if ind is None else U._array[ind]
        if self.parametric:
            mu = self.parse_parameter(mu)
            return NumpyVectorArray(self._mapping(U_array, mu=mu), copy=False)
        else:
            assert self.check_parameter(mu)
            return NumpyVectorArray(self._mapping(U_array), copy=False)


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
            return OrderedDict((('solve', {'type': 'solve'}),))

    def apply(self, U, ind=None, mu=None):
        if self._assembled:
            assert isinstance(U, NumpyVectorArray)
            assert self.check_parameter(mu)
            U_array = U._array[:U._len] if ind is None else U._array[ind]
            return NumpyVectorArray(self._last_op._matrix.dot(U_array.T).T, copy=False)
        else:
            return self.assemble(mu).apply(U, ind=ind)

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        if self._assembled:
            return self._last_op.apply_inverse(U, ind=ind, options=options)
        else:
            return self.assemble(mu).apply_inverse(U, ind=ind, options=options)


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
    calculate_sid = False

    def __init__(self, matrix, name=None):
        assert matrix.ndim <= 2
        if matrix.ndim == 1:
            matrix = np.reshape(matrix, (1, -1))
        self.dim_source = matrix.shape[1]
        self.dim_range = matrix.shape[0]
        self.name = name
        self._matrix = matrix
        self.sparse = issparse(matrix)
        self.calculate_sid = hasattr(matrix, 'sid')

    def _assemble(self, mu=None):
        assert self.check_parameter(mu)
        return self

    def assemble(self, mu=None):
        assert self.check_parameter(mu)
        return self

    def as_vector(self, mu=None):
        if self.dim_source != 1 and self.dim_range != 1:
            raise TypeError('This is operator does not represent a vector or linear functional.')
        assert self.check_parameter(mu)
        return NumpyVectorArray(self._matrix.ravel(), copy=True)

    def apply(self, U, ind=None, mu=None):
        assert isinstance(U, NumpyVectorArray)
        assert self.check_parameter(mu)
        U_array = U._array[:U._len] if ind is None else U._array[ind]
        return NumpyVectorArray(self._matrix.dot(U_array.T).T, copy=False)

    def apply_inverse(self, U, ind=None, mu=None, options=None):

        def check_options(options, sparse):
            if not options:
                return True
            assert 'type' in options
            if sparse:
                assert options['type'] == 'bicgstab'
                assert options.viewkeys() <= {'type', 'tol', 'maxiter'}
            else:
                assert options['type'] == 'solve'
                assert options.viewkeys() <= {'type'}
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
            tol = options.get('tol', defaults.bicgstab_tol)
            maxiter = options.get('maxiter', defaults.bicgstab_maxiter)
            for i, UU in enumerate(U):
                R[i], info = bicgstab(self._matrix, UU, tol=tol, maxiter=maxiter)
                if info != 0:
                    if info > 0:
                        raise InversionError('bicgstab failed to converge after {} iterations'.format(info))
                    else:
                        raise InversionError('bicgstab failed with error code {} (illegal input or breakdown)'.
                                             format(info))
        else:
            for i, UU in enumerate(U):
                try:
                    R[i] = np.linalg.solve(self._matrix, UU)
                except np.linalg.LinAlgError as e:
                    raise InversionError('{}: {}'.format(str(type(e)), str(e)))

        return NumpyVectorArray(R)

    def projected_to_subbasis(self, dim_source=None, dim_range=None, name=None):
        assert dim_source is None or dim_source <= self.dim_source
        assert dim_range is None or dim_range <= self.dim_range
        name = name or '{}_projected_to_subbasis'.format(self.name)
        return NumpyMatrixOperator(self._matrix[:dim_range, :dim_source], name=name)


class NumpyLincombMatrixOperator(LincombOperatorBase, NumpyMatrixBasedOperator):

    def __init__(self, operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
        assert all(isinstance(op, NumpyMatrixBasedOperator) for op in operators)
        LincombOperatorBase.__init__(self, operators=operators, coefficients=coefficients,
                                     num_coefficients=num_coefficients,
                                     coefficients_name=coefficients_name, name=name)
        self.sparse = all(op.sparse for op in operators)

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


class ProjectedOperator(OperatorBase):
    '''Projection of an operator to a subspace.

    Given an operator L, a scalar product ( ⋅, ⋅), and vectors b_1, ..., b_N,
    c_1, ..., c_M, the projected operator is defined by ::

        [ L_P(b_j) ]_i = ( c_i, L(b_j) )

    for all i,j.

    In particular, if b_i = c_i are orthonormal w.r.t. the product, then
    L_P is the coordinate representation of the orthogonal projection
    of L onto the subspace spanned by the b_i (with b_i as basis).

    From another point of view, if L represents the matrix of a bilinear form and
    ( ⋅, ⋅ ) is the euclidean scalar product, then L_P represents the matrix of
    the bilinear form restricted to the span of the b_i.

    It is not checked whether the b_i and c_j are linear independent.

    Parameters
    ----------
    operator
        The `Operator` to project.
    source_basis
        The b_1, ..., b_N as a `VectorArray` or `None`. If `None`, `operator.type_source`
        has to be a subclass of `NumpyVectorArray`.
    range_basis
        The c_1, ..., c_M as a `VectorArray`. If `None`, `operator.type_source`
        has to be a subclass of `NumpyVectorArray`.

    product
        An `Operator` representing the scalar product.
        If None, the euclidean product is chosen.
    name
        Name of the projected operator.
    '''

    type_source = type_range = NumpyVectorArray

    def __init__(self, operator, source_basis, range_basis, product=None, copy=True, name=None):
        assert isinstance(operator, OperatorInterface)
        assert isinstance(source_basis, operator.type_source) or issubclass(operator.type_source, NumpyVectorArray)
        assert issubclass(operator.type_range, type(range_basis)) or issubclass(operator.type_range, NumpyVectorArray)
        assert source_basis is None or source_basis.dim == operator.dim_source
        assert range_basis is None or range_basis.dim == operator.dim_range
        assert product is None \
            or (isinstance(product, OperatorInterface)
                and range_basis is not None
                and issubclass(operator.type_range, product.type_source)
                and issubclass(product.type_range, type(product))
                and product.dim_range == product.dim_source == operator.dim_range)
        super(ProjectedOperator, self).__init__()
        self.build_parameter_type(inherits=(operator,))
        self.dim_source = len(source_basis) if operator.dim_source > 0 else 0
        self.dim_range = len(range_basis) if range_basis is not None else operator.dim_range
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
            U_array = U._array if ind is None else U._array[ind]
            UU = self.source_basis.lincomb(U_array)
            if self.range_basis is None:
                return self.operator.apply(UU, mu=mu)
            elif self.product is None:
                return NumpyVectorArray(self.operator.apply2(self.range_basis, UU, mu=mu, pairwise=False).T)
            else:
                V = self.operator.apply(UU, mu=mu)
                return NumpyVectorArray(self.product.apply2(V, self.range_basis, pairwise=False))

    def projected_to_subbasis(self, dim_source=None, dim_range=None, name=None):
        assert dim_source is None or dim_source <= self.dim_source
        assert dim_range is None or dim_range <= self.dim_range
        assert dim_source is None or self.source_basis is not None, 'not implemented'
        assert dim_range is None or self.range_basis is not None, 'not implemented'
        name = name or '{}_projected_to_subbasis'.format(self.name)
        source_basis = self.source_basis if dim_source is None \
            else self.source_basis.copy(ind=range(dim_source))
        range_basis = self.range_basis if dim_range is None \
            else self.range_basis.copy(ind=range(dim_range))
        return ProjectedOperator(self.operator, source_basis, range_basis, product=None, copy=False, name=name)


class ProjectedLinearOperator(NumpyMatrixBasedOperator):
    '''Projection of an linear operator to a subspace.

    The same as ProjectedOperator, but the resulting operator is again a
    `LinearOperator`.

    See also `ProjectedOperator`.

    Parameters
    ----------
    operator
        The `DiscreteLinearOperator` to project.
    source_basis
        The b_1, ..., b_N as a 2d-array.
    range_basis
        The c_1, ..., c_M as a 2d-array. If None, `range_basis=source_basis`.
    product
        Either an 2d-array or a `Operator` representing the scalar product.
        If None, the euclidean product is chosen.
    name
        Name of the projected operator.
    '''

    sparse = False

    def __init__(self, operator, source_basis, range_basis, product=None, name=None, copy=True):
        assert isinstance(operator, OperatorInterface)
        assert isinstance(source_basis, operator.type_source) or issubclass(operator.type_source, NumpyVectorArray)
        assert issubclass(operator.type_range, type(range_basis)) or issubclass(operator.type_range, NumpyVectorArray)
        assert source_basis is None or source_basis.dim == operator.dim_source
        assert range_basis is None or range_basis.dim == operator.dim_range
        assert product is None \
            or (isinstance(product, OperatorInterface)
                and range_basis is not None
                and issubclass(operator.type_range, product.type_source)
                and issubclass(product.type_range, type(product))
                and product.dim_range == product.dim_source == operator.dim_range)
        assert operator.linear
        super(ProjectedLinearOperator, self).__init__()
        self.build_parameter_type(inherits=(operator,))
        self.dim_source = len(source_basis) if source_basis is not None else operator.dim_source
        self.dim_range = len(range_basis) if range_basis is not None else operator.dim_range
        self.name = name
        self.operator = operator
        self.source_basis = source_basis.copy() if source_basis is not None and copy else source_basis
        self.range_basis = range_basis.copy() if range_basis is not None and copy else range_basis
        self.product = product

    def _assemble(self, mu=None):
        mu = self.parse_parameter(mu)
        if self.source_basis is None:
            if self.range_basis is None:
                return self.operator.assemble(mu=mu)
            elif self.product is None:
                return NumpyMatrixOperator(self.operator.apply2(self.range_basis,
                                                                NumpyVectorArray(np.eye(self.operator.dim_source)),
                                                                pairwise=False, mu=mu),
                                           name='{}_assembled'.format(self.name))
            else:
                V = self.operator.apply(NumpyVectorArray(np.eye(self.operator.dim_source)), mu=mu)
                return NumpyMatrixOperator(self.product.apply2(self.range_basis, V, pairwise=False),
                                           name='{}_assembled'.format(self.name))
        else:
            if self.range_basis is None:
                M = self.operator.apply(self.source_basis, mu=mu).data.T
                return NumpyMatrixOperator(M, name='{}_assembled'.format(self.name))
            elif self.product is None:
                return NumpyMatrixOperator(self.operator.apply2(self.range_basis, self.source_basis, mu=mu,
                                                                pairwise=False),
                                           name='{}_assembled'.format(self.name))
            else:
                V = self.operator.apply(self.source_basis, mu=mu)
                return NumpyMatrixOperator(self.product.apply2(self.range_basis, V, pairwise=False),
                                           name='{}_assembled'.format(self.name))

    def projected_to_subbasis(self, dim_source=None, dim_range=None, name=None):
        assert dim_source is None or dim_source <= self.dim_source
        assert dim_range is None or dim_range <= self.dim_range
        assert dim_source is None or self.source_basis is not None, 'not implemented'
        assert dim_range is None or self.range_basis is not None, 'not implemented'
        name = name or '{}_projected_to_subbasis'.format(self.name)
        source_basis = self.source_basis if dim_source is None \
            else self.source_basis.copy(ind=range(dim_source))
        range_basis = self.range_basis if dim_range is None \
            else self.range_basis.copy(ind=range(dim_range))
        return ProjectedLinearOperator(self.operator, source_basis, range_basis, product=None, copy=False, name=name)


class LincombOperator(LincombOperatorBase):

    def __init__(self, operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
        super(LincombOperator, self).__init__(operators=operators, coefficients=coefficients,
                                              num_coefficients=num_coefficients,
                                              coefficients_name=coefficients_name, name=name)

    def apply(self, U, ind=None, mu=None):
        mu = self.parse_parameter(mu)
        coeffs = self.evaluate_coefficients(mu)
        Vs = [op.apply(U, ind=ind, mu=mu) for op in self.operators]
        R = Vs[0]
        R.scal(coeffs[0])
        for V, c in izip(Vs[1:], coeffs[1:]):
            R.axpy(c, V)
        return R
