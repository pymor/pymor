# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np

from pymor.la import NumpyVectorArray
from pymor.operators import OperatorInterface, OperatorBase
from pymor.operators.numpy import NumpyMatrixBasedOperator, NumpyMatrixOperator


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
        The b_1, ..., b_N as a `VectorArray`.
    range_basis
        The c_1, ..., c_M as a `VectorArray`. If None, `range_basis=source_basis`.
    product
        An `Operator` representing the scalar product.
        If None, the euclidean product is chosen.
    name
        Name of the projected operator.
    '''

    type_source = type_range = NumpyVectorArray

    def __init__(self, operator, source_basis, range_basis, product=None, name=None):
        assert source_basis is not None or operator.dim_source == 0
        assert range_basis is not None
        assert isinstance(operator, OperatorInterface)
        assert source_basis is None or operator.dim_source == source_basis.dim
        assert operator.dim_range == range_basis.dim
        super(ProjectedOperator, self).__init__()
        self.build_parameter_type(inherits=(operator,))
        self.dim_source = len(source_basis) if operator.dim_source > 0 else 0
        self.dim_range = len(range_basis)
        self.name = name
        self.operator = operator
        self.source_basis = source_basis
        self.range_basis = range_basis
        self.product = product
        self.lock()

    def apply(self, U, ind=None, mu=None):
        mu = self.parse_parameter(mu)
        if self.source_basis is not None:
            U_array = U._array if ind is None else U._array[ind]
            V = self.source_basis.lincomb(U_array)
            if self.product is None:
                return NumpyVectorArray(self.operator.apply2(self.range_basis, V, mu=mu, pairwise=False).T)
            else:
                V = self.operator.apply(V, mu=mu)
                return NumpyVectorArray(self.product.apply2(V, self.range_basis, pairwise=False))
        else:
            V = self.operator.apply(U, ind=ind, mu=mu)
            if self.product is None:
                return NumpyVectorArray(V.dot(self.range_basis, pairwise=False))
            else:
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
        assert operator.linear
        assert operator.dim_source == source_basis.dim
        assert operator.dim_range == range_basis.dim
        super(ProjectedLinearOperator, self).__init__()
        self.build_parameter_type(inherits=(operator,))
        self.dim_source = len(source_basis)
        self.dim_range = len(range_basis)
        self.name = name
        self.operator = operator
        self.source_basis = source_basis
        self.range_basis = range_basis
        self.product = product
        self.lock()

    def _assemble(self, mu=None):
        if self.product is None:
            return NumpyMatrixOperator(self.operator.apply2(self.range_basis, self.source_basis, mu=mu, pairwise=False),
                                       name='{}_assembled'.format(self.name))
        else:
            AU = self.operator.apply(self.source_basis, mu=mu)
            return NumpyMatrixOperator(self.product.apply2(self.range_basis, AU, pairwise=False),
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

    name = name or '{}_projected'.format(operator.name)

    if hasattr(operator, 'projected'):
        return operator.projected(source_basis=source_basis, range_basis=range_basis, product=product, name=name)
    elif operator.linear:
        return ProjectedLinearOperator(operator, source_basis, range_basis, product, name)
    else:
        return ProjectedOperator(operator, source_basis, range_basis, product, name)


def rb_project_operator(operator, rb, product=None, name=None):
    if operator.dim_source > 0:
        assert operator.dim_source == rb.dim
        source_basis = rb
    else:
        source_basis = None
    if operator.dim_range > 1:
        assert operator.dim_range == rb.dim
        range_basis = rb
    elif operator.dim_range == 1:
        if operator.type_range is not NumpyVectorArray:
            raise NotImplementedError
        range_basis = NumpyVectorArray(np.ones((1,1)))
    else:
        raise NotImplementedError
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
