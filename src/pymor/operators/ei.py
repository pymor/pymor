# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.linalg import solve_triangular


from pymor.la.interfaces import VectorArrayInterface
from pymor.la import NumpyVectorArray
from pymor.operators.interfaces import OperatorInterface, LinearOperatorInterface
from pymor.tools import float_cmp_all


class EmpiricalInterpolatedOperator(OperatorInterface):

    def __init__(self, operator, interpolation_dofs, collateral_basis, name=None):
        assert isinstance(operator, OperatorInterface)
        assert isinstance(collateral_basis, VectorArrayInterface)
        assert operator.dim_source == operator.dim_range == collateral_basis.dim
        assert operator.type_range == type(collateral_basis)
        assert hasattr(operator, 'restricted')

        self.build_parameter_type(inherits={'operator': operator})
        self.dim_source = operator.dim_source
        self.dim_range = operator.dim_range
        self.type_source = operator.type_source
        self.type_range = operator.type_range
        self.name = name or '{}_interpolated'.format(operator.name)

        interpolation_dofs = np.array(interpolation_dofs, dtype=np.int32)
        self.interpolation_dofs = interpolation_dofs

        if len(interpolation_dofs) > 0:
            restricted_operator, source_dofs  = operator.restricted(interpolation_dofs)
            interpolation_matrix = collateral_basis.components(interpolation_dofs).T
            assert float_cmp_all(interpolation_matrix.diagonal(), 1)
            # assert float_cmp_all(interpolation_matrix, np.triu(interpolation_matrix))
            self.restricted_operator = restricted_operator
            self.source_dofs = source_dofs
            self.interpolation_matrix = interpolation_matrix
            self.collateral_basis = collateral_basis.copy()

    def apply(self, U, ind=None, mu=None):
        if len(self.interpolation_dofs) == 0:
            count = len(ind) if ind is not None else len(U)
            return self.type_range.zeros(dim=self.dim_range, count=count)

        U_components = NumpyVectorArray(U.components(self.source_dofs, ind=ind), copy=False)
        AU = self.restricted_operator.apply(U_components, mu=self.map_parameter(mu, 'operator'))
        interpolation_coefficients = solve_triangular(self.interpolation_matrix, AU._array.T,
                                                      lower=True, unit_diagonal=True).T
        # interpolation_coefficients = np.linalg.solve(self.interpolation_matrix, AU._array.T).T
        # interpolation_coefficients = AU._array
        return self.collateral_basis.lincomb(interpolation_coefficients)

    def projected(self, source_basis, range_basis, product=None, name=None):
        assert source_basis is not None or self.dim_source == 0
        assert range_basis is not None
        assert source_basis is None or self.dim_source == source_basis.dim
        assert self.dim_range == range_basis.dim
        assert source_basis is None or self.type_source == type(source_basis)
        assert self.type_range == type(range_basis)

        class ProjectedEmpiciralInterpolatedOperator(OperatorInterface):

            def __init__(self, restricted_operator, interpolation_matrix, source_basis_dofs,
                         projected_collateral_basis, name=None):
                self.dim_source = len(source_basis_dofs)
                self.dim_range = projected_collateral_basis.dim
                self.type_source = self.type_range = NumpyVectorArray
                self.build_parameter_type(inherits={'operator':restricted_operator})
                self.restricted_operator = restricted_operator
                self.interpolation_matrix = interpolation_matrix
                self.source_basis_dofs = source_basis_dofs
                self.projected_collateral_basis = projected_collateral_basis
                self.name = name or '{}_projected'.format(restricted_operator.name)

            def apply(self, U, ind=None, mu=None):
                U_components = self.source_basis_dofs.lincomb(U._array, ind=ind)
                AU = self.restricted_operator.apply(U_components, mu=self.map_parameter(mu, 'operator'))
                interpolation_coefficients = solve_triangular(self.interpolation_matrix, AU._array.T,
                                                              lower=True, unit_diagonal=True).T
                return self.projected_collateral_basis.lincomb(interpolation_coefficients)

        if product is None:
            projected_collateral_basis = NumpyVectorArray(self.collateral_basis.prod(range_basis, pairwise=False))
        else:
            projected_collateral_basis = NumpyVectorArray(product.apply2(self.collateral_basis, range_basis,
                                                                         pairwise=False))

        return ProjectedEmpiciralInterpolatedOperator(self.restricted_operator, self.interpolation_matrix,
                                                      NumpyVectorArray(source_basis.components(self.source_dofs),
                                                                       copy=False),
                                                      projected_collateral_basis, name)
