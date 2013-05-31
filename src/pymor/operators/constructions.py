# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.la import NumpyVectorArray
from pymor.operators.interfaces import OperatorInterface, LinearOperatorInterface
from pymor.operators.affine import LinearAffinelyDecomposedOperator
from pymor.operators.numpy import NumpyLinearOperator


class ProjectedOperator(OperatorInterface):
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
        self.build_parameter_type(operator.parameter_type, local_global=True)
        self.dim_source = len(source_basis) if operator.dim_source > 0 else 0
        self.dim_range = len(range_basis)
        self.name = name
        self.operator = operator
        self.source_basis = source_basis
        self.range_basis = range_basis
        self.product = product

    def apply(self, U, ind=None, mu=None):
        if self.source_basis is not None:
            U_array = U._array if ind is None else U._array[ind]
            V = self.source_basis.lincomb(U_array)
            if self.product is None:
                return NumpyVectorArray(self.operator.apply2(self.range_basis, V, mu=self.map_parameter(mu),
                                                             pairwise=False).T)
            else:
                V = self.operator.apply(V, mu=self.map_parameter(mu))
                return NumpyVectorArray(self.product.apply2(V, self.range_basis, pairwise=False))
        else:
            V = self.operator.apply(U, ind=ind, mu=self.map_parameter(mu))
            if self.product is None:
                return NumpyVectorArray(V.prod(self.range_basis))
            else:
                return NumpyVectorArray(self.product.apply2(V, self.range_basis, pairwise=False))


class ProjectedLinearOperator(LinearOperatorInterface):
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

    type_source = type_range = NumpyVectorArray
    sparse = False

    def __init__(self, operator, source_basis, range_basis, product=None, name=None):
        assert isinstance(operator, LinearOperatorInterface)
        assert operator.dim_source == source_basis.dim
        assert operator.dim_range == range_basis.dim
        super(ProjectedLinearOperator, self).__init__()
        self.build_parameter_type(operator.parameter_type, local_global=True)
        self.dim_source = len(source_basis)
        self.dim_range = len(range_basis)
        self.name = name
        self.operator = operator
        self.source_basis = source_basis
        self.range_basis = range_basis
        self.product = product

    def _assemble(self, mu=None):
        if self.product is None:
            return NumpyLinearOperator(self.operator.apply2(self.range_basis, self.source_basis,
                                                            mu=self.map_parameter(mu), pairwise=False),
                                       name='{}_assembled'.format(self.name))
        else:
            AU = self.operator.apply(self.source_basis, mu=self.map_parameter(mu))
            return NumpyLinearOperator(self.product.apply2(self.range_basis, AU, pairwise=False),
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

    if isinstance(operator, LinearAffinelyDecomposedOperator):
        proj_operators = tuple(project_operator(op, source_basis, range_basis, product,
                                                name='{}_projected'.format(op.name))
                               for op in operator.operators)
        if operator.operator_affine_part is not None:
            proj_operator_ap = project_operator(operator.operator_affine_part, source_basis, range_basis, product,
                                                name='{}_projected'.format(operator.operator_affine_part.name))
        else:
            proj_operator_ap = None
        proj_operator = LinearAffinelyDecomposedOperator(proj_operators, proj_operator_ap, operator.functionals, name)
        proj_operator.rename_parameter(operator.parameter_name_map)
        return proj_operator

    elif isinstance(operator, LinearOperatorInterface):
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


class LincombOperator(OperatorInterface):
    '''Operator representing the sum operators.

    Given operators L_1, ..., L_K, this defines the operator given by ::

        L(u) = L_1(u) + ... + L_K(u)

    Parameters
    ----------
    operators
        List of the `Operators` L_1, ..., L_K.
    name
        Name of the operator.
    '''

    def __init__(self, operators, factors=None, name=None):
        assert all(isinstance(op, OperatorInterface) for op in operators)
        assert all(op.dim_source == operators[0].dim_source for op in operators)
        assert all(op.dim_range == operators[0].dim_range for op in operators)
        assert all(op.type_source == operators[0].type_source for op in operators)
        assert all(op.type_range == operators[0].type_range for op in operators)
        super(LincombOperator, self).__init__()
        self.build_parameter_type(inherits={'operators': operators})
        self.operators = operators
        self.factors = np.ones(len(operators)) if factors is None else factors
        self.dim_source = operators[0].dim_source
        self.dim_range = operators[0].dim_range
        self.type_source = operators[0].type_source
        self.type_range = operators[0].type_range
        self.name = name or '+'.join(op.name for op in operators)

    def apply(self, U, ind=None, mu=None):
        return sum(op.apply(U, ind=ind, mu=self.map_parameter(mu, 'operators', i)) * self.factors[i]
                   for i, op in enumerate(self.operators))


class LinearLincombOperator(LinearOperatorInterface):
    '''Linear operator representing the sum linear operators.

    Given linear operators L_1, ..., L_K, this defines the linear operator given by ::

        L(u) = L_1(u) + ... + L_K(u)

    Parameters
    ----------
    operators
        List of the `LinearOperators` L_1, ..., L_K.
    name
        Name of the operator.
    '''

    def __init__(self, operators, factors=None, name=None):
        assert all(isinstance(op, LinearOperatorInterface) for op in operators)
        assert all(op.dim_source == operators[0].dim_source for op in operators)
        assert all(op.dim_range == operators[0].dim_range for op in operators)
        assert all(op.type_source == operators[0].type_source for op in operators)
        assert all(op.type_range == operators[0].type_range for op in operators)
        super(LinearLincombOperator, self).__init__()
        self.build_parameter_type(inherits={'operators': operators})
        self.operators = operators
        self.factors = np.ones(len(operators)) if factors is None else factors
        self.dim_source = operators[0].dim_source
        self.dim_range = operators[0].dim_range
        self.type_source = operators[0].type_source
        self.type_range = operators[0].type_range
        self.name = name or '+'.join(op.name for op in operators)

    def apply(self, U, ind=None, mu=None):
        if self.assembled:
            assert mu == None
            return sum(op.apply(U, ind=ind, mu=None) * self.factors[i]
                       for i, op in enumerate(self.operators))
        else:
            return self.assemble(mu).apply(U, ind=ind)

    def _assemble(self, mu=None):
        if self.assembled:
            return self
        M = self.operators[0].assemble(self.map_parameter(mu, 'operators', 0)) * self.factors[0]
        for i, op in enumerate(self.operators[1:]):
            M = M + op.assemble(self.map_parameter(mu, 'operators', i + 1)) * self.factors[i + 1]
        M.assembled = True
        return M


def add_operators(operators, name=None):
    '''Operator representing the sum operators.

    Given operators L_1, ..., L_K, this defines the operator given by ::

        L(u) = L_1(u) + ... + L_K(u)

    If all L_k are linear, a LinearSumOperator is returned, otherwise a
    SumOperator.

    Parameters
    ----------
    operators
        List of the `Operators` L_1, ..., L_k.
    name
        Name of the operator.
    '''
    if all(isinstance(op, LinearOperatorInterface) for op in operators):
        return LinearLincombOperator(operators, name=name)
    else:
        return LincombOperator(operators, name=name)
