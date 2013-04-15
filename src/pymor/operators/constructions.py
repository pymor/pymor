# -*- coding: utf-8 -*-
# pymor (http://www.pymor.org)
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from .interfaces import OperatorInterface, LinearOperatorInterface
from .affine import LinearAffinelyDecomposedOperator
from .basic import GenericLinearOperator


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
        The b_1, ..., b_N as a 2d-array.
    range_basis
        The c_1, ..., c_M as a 2d-array. If None, `range_basis=source_basis`.
    product
        Either an 2d-array or a `Operator` representing the scalar product.
        If None, the euclidean product is chosen.
    name
        Name of the projected operator.
    '''

    def __init__(self, operator, source_basis, range_basis=None, product=None, name=None):
        if range_basis is None:
            range_basis = np.ones((1, 1)) if operator.dim_range == 1 else source_basis
        assert isinstance(operator, OperatorInterface)
        assert operator.dim_source == source_basis.shape[1]
        assert operator.dim_range == range_basis.shape[1]
        super(ProjectedOperator, self).__init__()
        self.build_parameter_type(operator.parameter_type, local_global=True)
        self.dim_source = source_basis.shape[0]
        self.dim_range = range_basis.shape[0]
        self.name = name
        self.operator = operator
        self.source_basis = source_basis
        self.range_basis = range_basis
        self.product = product

    def apply(self, U, mu={}):
        V = np.dot(U, self.source_basis)
        AV = self.operator.apply(V, self.map_parameter(mu))
        if self.product is None:
            return np.dot(AV, self.range_basis.T)
        elif isinstance(self.product, OperatorInterface):
            return self.product.apply2(AV, self.range_basis, pairwise=False)
        else:
            return np.dot(np.dot(AV, self.product), self.range_basis.T)


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

    def __init__(self, operator, source_basis, range_basis=None, product=None, name=None):
        if range_basis is None:
            range_basis = np.ones((1, 1)) if operator.dim_range == 1 else source_basis
        assert isinstance(operator, LinearOperatorInterface)
        assert operator.dim_source == source_basis.shape[1]
        assert operator.dim_range == range_basis.shape[1]
        super(ProjectedLinearOperator, self).__init__()
        self.build_parameter_type(operator.parameter_type, local_global=True)
        self.dim_source = source_basis.shape[0]
        self.dim_range = range_basis.shape[0]
        self.name = name
        self.operator = operator
        self.source_basis = source_basis
        self.range_basis = range_basis
        self.product = product

    def assemble(self, mu={}):
        M = self.operator.matrix(self.map_parameter(mu))
        MB = M.dot(self.source_basis.T)
        if self.product is None:
            return np.dot(self.range_basis, MB)
        elif isinstance(self.product, OperatorInterface):
            return self.product.apply2(self.range_basis, MB.T, pairwise=False)
        else:
            return np.dot(self.range_basis, np.dot(self.product, MB))


def project_operator(operator, source_basis, range_basis=None, product=None, name=None):
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
        The c_1, ..., c_M as a 2d-array. If None, `range_basis=source_basis`.
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
        proj_operator.rename_parameter(operator.parameter_user_map)
        return proj_operator

    elif isinstance(operator, LinearOperatorInterface):
        proj_operator = ProjectedLinearOperator(operator, source_basis, range_basis, product, name)
        if proj_operator.parameter_type == {}:
            return GenericLinearOperator(proj_operator.matrix(), name)
        else:
            return proj_operator

    else:
        return ProjectedOperator(operator, source_basis, range_basis, product, name)


class SumOperator(OperatorInterface):
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

    def __init__(self, operators, name=None):
        assert all(isinstance(op, OperatorInterface) for op in operators)
        assert all(op.dim_source == operators[0].dim_source for op in operators)
        assert all(op.dim_range == operators[0].dim_range for op in operators)
        super(SumOperator, self).__init__()
        self.build_parameter_type(inherits={'operators': operators})
        self.operators = operators
        self.dim_source = operators[0].dim_source
        self.dim_range = operators[0].dim_range
        self.name = name or '+'.join(op.name for op in operators)

    def apply(self, U, mu={}):
        return np.sum([op.apply(U, self.map_parameter(mu, 'operators', i)) for i, op in enumerate(self.operators)],
                      axis=0)


class LinearSumOperator(LinearOperatorInterface):
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

    def __init__(self, operators, name=None):
        assert all(isinstance(op, LinearOperatorInterface) for op in operators)
        assert all(op.dim_source == operators[0].dim_source for op in operators)
        assert all(op.dim_range == operators[0].dim_range for op in operators)
        super(LinearSumOperator, self).__init__()
        self.build_parameter_type(inherits={'operators': operators})
        self.operators = operators
        self.dim_source = operators[0].dim_source
        self.dim_range = operators[0].dim_range
        self.name = name or '+'.join(op.name for op in operators)

    def assemble(self, mu={}):
        M = self.operators[0].matrix(self.map_parameter(mu, 'operators', 0))
        for i, op in enumerate(self.operators[1:]):
            M = M + op.matrix(self.map_parameter(mu, 'operators', i + 1))
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
        return LinearSumOperator(operators, name)
    else:
        return SumOperator(operators, name)
