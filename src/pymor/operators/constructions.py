# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from numbers import Number

import numpy as np

from pymor.la import VectorArrayInterface, NumpyVectorArray
from pymor.operators import OperatorInterface, OperatorBase
from pymor.operators import NumpyMatrixBasedOperator, NumpyMatrixOperator


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

    def __init__(self, operator, source_basis, range_basis, product=None, name=None):
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
        self.source_basis = source_basis
        self.range_basis = range_basis
        self.product = product
        self.lock()

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

    def __init__(self, operator, source_basis, range_basis, product=None, name=None):
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
        self.dim_source = len(source_basis) if operator.dim_source > 0 else 0
        self.dim_range = len(range_basis) if range_basis is not None else operator.dim_range
        self.name = name
        self.operator = operator
        self.source_basis = source_basis
        self.range_basis = range_basis
        self.product = product
        self.lock()

    def _assemble(self, mu=None):
        mu = self.parse_parameter(mu)
        if self.source_basis is None:
            if self.range_basis is None:
                return self.operator.assemble(mu=mu)
            elif product is None:
                return NumpyMatrixOperator(self.operator.apply2(self.range_basis,
                                                                NumpyVectorArray(np.eye(self.operator.dim_source)),
                                                                mu=mu), name='{}_assembled'.format(self.name))
            else:
                V = self.operator.apply(NumpyVectorArray(np.eye(self.operator.dim_source)), mu=mu)
                return NumpyMatrixOperator(self.product.apply2(self.range_basis, V, pairwise=False),
                                           name='{}_assembled'.format(self.name))
        else:
            if self.range_basis is None:
                M = self.operator.apply(self.source_basis, mu=mu).data.T
                return NumpyMatrixOperator(M, name='{}_assembled'.format(self.name))
            elif self.product is None:
                return NumpyMatrixOperator(self.operator.apply2(self.range_basis, self.source_basis, mu=mu, pairwise=False),
                                           name='{}_assembled'.format(self.name))
            else:
                V = self.operator.apply(self.source_basis, mu=mu)
                return NumpyMatrixOperator(self.product.apply2(self.range_basis, V, pairwise=False),
                                           name='{}_assembled'.format(self.name))


def project_operator(operator, source_basis, range_basis, product=None, name=None):
    '''Project operators to subspaces.

    Replaces `Operators` by `ProjectedOperators` and `LinearOperators`
    by `ProjectedLinearOperators`.
    Moreover, `LinearAffinelyDecomposedOperators` are projected by recursively
    projecting each of its components.

    See also `ProjectedOperator`.

    Parameters
    ----------
    operator
        The `Operator` to project.
    source_basis
        The b_1, ..., b_N as a 2d-array.
    range_basis
        The c_1, ..., c_M as a 2d-array.
    product
        Either an 2d-array or a `Operator` representing the scalar product.
        If None, the euclidean product is chosen.
    name
        Name of the projected operator.
    '''
    assert operator is None or isinstance(operator, OperatorInterface)

    name = name or '{}_projected'.format(operator.name)

    if operator is None:
        return None
    elif hasattr(operator, 'projected'):
        return operator.projected(source_basis=source_basis, range_basis=range_basis, product=product, name=name)
    elif operator.linear:
        return ProjectedLinearOperator(operator, source_basis, range_basis, product, name)
    else:
        return ProjectedOperator(operator, source_basis, range_basis, product, name)


def rb_project_operator(operator, rb, product=None, name=None):
    assert operator is None or isinstance(operator, OperatorInterface)

    if operator is None:
        return None
    if operator.dim_source > 0:
        assert operator.dim_source == rb.dim
        source_basis = rb
    else:
        source_basis = None
    if operator.dim_range > 1:
        assert operator.dim_range == rb.dim
        range_basis = rb
    else:
        range_basis = None

    return project_operator(operator, source_basis, range_basis, product=product, name=name)


class Concatenation(OperatorBase):

    def __init__(self, second, first, name=None):
        assert isinstance(second, OperatorInterface)
        assert isinstance(first, OperatorInterface)
        assert second.dim_source == first.dim_range
        assert second.type_source == first.type_range
        super(Concatenation, self).__init__()
        self.first = first
        self.second = second
        self.build_parameter_type(inherits=(second, first))
        self.dim_source = first.dim_source
        self.dim_range = second.dim_range
        self.type_source = first.type_source
        self.type_range = second.type_range
        self.linear = second.linear and first.linear
        self.name = name
        self.lock()

    def apply(self, U, ind=None, mu=None):
        mu = self.parse_parameter(mu)
        return self.second.apply(self.first.apply(U, ind=ind, mu=mu), mu=mu)


class ComponentProjection(OperatorBase):

    type_range = NumpyVectorArray

    def __init__(self, components, dim, type_source, name=None):
        assert all(0 <= c < dim for c in components)
        self.components = components
        self.dim_source = dim
        self.dim_range = len(components)
        self.type_source = type_source
        self.name = name
        self.lock()

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)
        assert isinstance(U, self.type_source)
        assert U.dim == self.dim_source
        return NumpyVectorArray(U.components(self.components, ind), copy=False)


class ConstantOperator(OperatorBase):

    type_source = NumpyVectorArray

    dim_source = 0

    def __init__(self, value, name=None):
        assert isinstance(value, VectorArrayInterface)
        assert len(value) == 1

        super(ConstantOperator, self).__init__()
        self.dim_range = value.dim
        self.type_range = type(value)
        self.name = name
        self._value = value.copy()
        self.lock()

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)
        assert isinstance(U, (NumpyVectorArray, Number))
        if isinstance(U, Number):
            assert U == 0.
            assert ind == None
            return self._value.copy()
        else:
            assert U.dim == 0
            if ind is not None:
                raise NotImplementedError
            return self._value.copy()

    def as_vector(self):
        '''Returns the image of the operator as a VectorArray of length 1.'''
        return self._value.copy()

    def __add__(self, other):
        if isinstance(other, ConstantOperator):
            return ConstantOperator(self._vector + other._vector)
        elif isinstance(other, Number):
            return ConstantOperator(self._vector + other)
        else:
            return NotImplemented

    __radd__ = __add__

    def __mul__(self, other):
        return ConstantOperator(self._vector * other)


