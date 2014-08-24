# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.linalg import solve_triangular


from pymor.la import NumpyVectorArray, NumpyVectorSpace
from pymor.la.interfaces import VectorArrayInterface
from pymor.operators import OperatorInterface, OperatorBase


class EmpiricalInterpolatedOperator(OperatorBase):
    """Interpolate an |Operator| using Empirical Operator Interpolation.

    Let `L` be an |Operator|, `0 <= c_1, ..., c_M <= L.range.dim` indices
    of interpolation DOFs and `b_1, ..., b_M in R^(L.range.dim)` collateral
    basis vectors. If moreover `ψ_j(U)` denotes the j-th component of `U`, the
    empirical interpolation `L_EI` of `L` w.r.t. the given data is given by ::

      |                M
      |   L_EI(U, μ) = ∑ b_i⋅λ_i     such that
      |               i=1
      |
      |   ψ_(c_i)(L_EI(U, μ)) = ψ_(c_i)(L(U, μ))   for i=0,...,M

    Since the original operator only has to be evaluated at the given interpolation
    DOFs, |EmpiricalInterpolatedOperator| calls `operator.restricted(interpolation_dofs)`
    to obtain a restricted version of the operator which is stored and later used
    to quickly obtain the required evaluations. (The second return value of the `restricted`
    method has to be an array of source DOFs -- determined by the operator's stencil --
    required to evaluate the restricted operator.) If the operator fails to have
    a `restricted` method, the full operator will be evaluated (which will lead to
    the same result, but without any speedup).

    The interpolation DOFs and the collateral basis can be generated using
    the algorithms provided in the :mod:`pymor.algorithms.ei` module.


    Parameters
    ----------
    operator
        The |Operator| to interpolate. The operator must implement a `restricted`
        method as described above.
    interpolation_dofs
        List or 1D |NumPy array| of the interpolation DOFs `c_1, ..., c_M`.
    collateral_basis
        |VectorArray| containing the collateral basis `b_1, ..., b_M`.
    triangular
        If `True`, assume that ψ_(c_i)(b_j) = 0  for i < j, which means
        that the interpolation matrix is triangular.
    name
        Name of the operator.
    """

    def __init__(self, operator, interpolation_dofs, collateral_basis, triangular, name=None):
        assert isinstance(operator, OperatorInterface)
        assert isinstance(collateral_basis, VectorArrayInterface)
        assert collateral_basis in operator.range

        self.build_parameter_type(inherits=(operator,))
        self.source = operator.source
        self.range = operator.range
        self.linear = operator.linear
        self.name = name or '{}_interpolated'.format(operator.name)

        interpolation_dofs = np.array(interpolation_dofs, dtype=np.int32)
        self.interpolation_dofs = interpolation_dofs
        self.triangular = triangular

        if len(interpolation_dofs) > 0:
            if hasattr(operator, 'restricted'):
                self.restricted_operator, self.source_dofs  = operator.restricted(interpolation_dofs)
            else:
                self.logger.warn('Operator has no "restricted" method. The full operator will be evaluated.')
                self.operator = operator
            interpolation_matrix = collateral_basis.components(interpolation_dofs).T
            self.interpolation_matrix = interpolation_matrix
            self.collateral_basis = collateral_basis.copy()

    def apply(self, U, ind=None, mu=None):
        mu = self.parse_parameter(mu)
        if len(self.interpolation_dofs) == 0:
            count = len(ind) if ind is not None else len(U)
            return self.range.zeros(count=count)

        if hasattr(self, 'restricted_operator'):
            U_components = NumpyVectorArray(U.components(self.source_dofs, ind=ind), copy=False)
            AU = self.restricted_operator.apply(U_components, mu=mu)
        else:
            AU = NumpyVectorArray(self.operator.apply(U, mu=mu).components(self.interpolation_dofs), copy=False)
        try:
            if self.triangular:
                interpolation_coefficients = solve_triangular(self.interpolation_matrix, AU.data.T,
                                                              lower=True, unit_diagonal=True).T
            else:
                interpolation_coefficients = np.linalg.solve(self.interpolation_matrix, AU._array.T).T
        except ValueError:  # this exception occurs when AU contains NaNs ...
            interpolation_coefficients = np.empty((len(AU), len(self.collateral_basis))) + np.nan
        return self.collateral_basis.lincomb(interpolation_coefficients)

    def projected(self, source_basis, range_basis, product=None, name=None):
        assert source_basis is not None or self.source.dim == 0
        assert source_basis is None or source_basis in self.source
        assert range_basis in self.range

        if not hasattr(self, 'restricted_operator'):
            return super(EmpiricalInterpolatedOperator, self).projected(source_basis, range_basis, product, name)

        if product is None:
            projected_collateral_basis = NumpyVectorArray(self.collateral_basis.dot(range_basis, pairwise=False))
        else:
            projected_collateral_basis = NumpyVectorArray(product.apply2(self.collateral_basis, range_basis,
                                                                         pairwise=False))

        return ProjectedEmpiciralInterpolatedOperator(self.restricted_operator, self.interpolation_matrix,
                                                      NumpyVectorArray(source_basis.components(self.source_dofs),
                                                                       copy=False),
                                                      projected_collateral_basis, self.triangular, name)

    def jacobian(self, U, mu=None):
        mu = self.parse_parameter(mu)
        if hasattr(self, 'operator'):
            return EmpiricalInterpolatedOperator(self.operator.jacobian(U, mu=mu), self.interpolation_dofs,
                                                 self.collateral_basis, self.triangular, self.name + '_jacobian')
        else:
            raise NotImplementedError


class ProjectedEmpiciralInterpolatedOperator(OperatorBase):
    """Project an |EmpiricalInterpolatedOperator|.

    Not intended to be used directly. Instead use :meth:`~pymor.operators.interfaces.OperatorInterface.projected`.
    """

    def __init__(self, restricted_operator, interpolation_matrix, source_basis_dofs,
                 projected_collateral_basis, triangular, name=None):
        self.source = NumpyVectorSpace(len(source_basis_dofs))
        self.range = NumpyVectorSpace(projected_collateral_basis.dim)
        self.linear = restricted_operator.linear
        self.build_parameter_type(inherits=(restricted_operator,))
        self.restricted_operator = restricted_operator
        self.interpolation_matrix = interpolation_matrix
        self.source_basis_dofs = source_basis_dofs
        self.projected_collateral_basis = projected_collateral_basis
        self.triangular = triangular
        self.name = name or '{}_projected'.format(restricted_operator.name)

    def apply(self, U, ind=None, mu=None):
        mu = self.parse_parameter(mu)
        U_array = U._array if ind is None else U._array[ind]
        U_components = self.source_basis_dofs.lincomb(U_array)
        AU = self.restricted_operator.apply(U_components, mu=mu)
        try:
            if self.triangular:
                interpolation_coefficients = solve_triangular(self.interpolation_matrix, AU.data.T,
                                                              lower=True, unit_diagonal=True).T
            else:
                interpolation_coefficients = np.linalg.solve(self.interpolation_matrix, AU._array.T).T
        except ValueError:  # this exception occurs when AU contains NaNs ...
            interpolation_coefficients = np.empty((len(AU), len(self.projected_collateral_basis))) + np.nan
        return self.projected_collateral_basis.lincomb(interpolation_coefficients)

    def projected_to_subbasis(self, dim_source=None, dim_range=None, dim_collateral=None, name=None):
        assert dim_source is None or dim_source <= self.source.dim
        assert dim_range is None or dim_range <= self.range.dim
        assert dim_collateral is None or dim_collateral <= self.restricted_operator.range.dim
        name = name or '{}_projected_to_subbasis'.format(self.name)

        interpolation_matrix = self.interpolation_matrix[:dim_collateral, :dim_collateral]

        if dim_collateral is not None:
            restricted_operator, source_dofs = self.restricted_operator.restricted(np.arange(dim_collateral))
        else:
            restricted_operator = self.restricted_operator

        old_pcb = self.projected_collateral_basis
        projected_collateral_basis = NumpyVectorArray(old_pcb.data[:dim_collateral, :dim_range], copy=False)

        old_sbd = self.source_basis_dofs
        source_basis_dofs = NumpyVectorArray(old_sbd.data[:dim_source], copy=False) if dim_collateral is None \
            else NumpyVectorArray(old_sbd.data[:dim_source, source_dofs], copy=False)

        return ProjectedEmpiciralInterpolatedOperator(restricted_operator, interpolation_matrix,
                                                      source_basis_dofs, projected_collateral_basis, self.triangular,
                                                      name=name)
