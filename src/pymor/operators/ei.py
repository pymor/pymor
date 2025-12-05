# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import weakref

import numpy as np
from scipy.linalg import lstsq, lu_factor, lu_solve, qr, solve, solve_triangular

from pymor.operators.constructions import (
    ComponentProjectionOperator,
    ConcatenationOperator,
    VectorArrayOperator,
    ZeroOperator,
)
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


class EmpiricalInterpolatedOperator(Operator):
    """Interpolate an |Operator| using Empirical Operator Interpolation.

    Let `L` be an |Operator|, `0 <= c_1, ..., c_M < L.range.dim` indices
    of interpolation DOFs and let `b_1, ..., b_M in R^(L.range.dim)` be collateral
    basis vectors. If moreover `ψ_j(U)` denotes the j-th component of `U`, the
    empirical interpolation `L_EI` of `L` w.r.t. the given data is given by ::

      |                M
      |   L_EI(U, μ) = ∑ b_i⋅λ_i     such that
      |               i=1
      |
      |   ψ_(c_i)(L_EI(U, μ)) = ψ_(c_i)(L(U, μ))   for i=0,...,M

    Since the original operator only has to be evaluated at the given interpolation
    DOFs, |EmpiricalInterpolatedOperator| calls
    :meth:`~pymor.operators.interface.Operator.restricted`
    to obtain a restricted version of the operator which is used
    to quickly obtain the required evaluations. If the `restricted` method, is not
    implemented, the full operator will be evaluated (which will lead to
    the same result, but without any speedup).

    The interpolation DOFs and the collateral basis can be generated using
    the algorithms provided in the :mod:`pymor.algorithms.ei` module.


    Parameters
    ----------
    operator
        The |Operator| to interpolate.
    interpolation_dofs
        List or 1D |NumPy array| of the interpolation DOFs `c_1, ..., c_M`.
    collateral_basis
        |VectorArray| containing the collateral basis `b_1, ..., b_M`.
    triangular
        If `True`, assume that ψ_(c_i)(b_j) = 0  for i < j, which means
        that the interpolation matrix is triangular.
    solver
        The |Solver| for the operator.
    name
        Name of the operator.
    """

    def __init__(self, operator, interpolation_dofs, collateral_basis, triangular,
                 least_squares=False, solver=None, name=None):
        assert isinstance(operator, Operator)
        assert isinstance(collateral_basis, VectorArray)
        assert collateral_basis in operator.range
        assert (len(interpolation_dofs) == len(collateral_basis)) or least_squares

        self.source = operator.source
        self.range = operator.range
        self.linear = operator.linear
        self.solver = solver
        self.name = name or f'{operator.name}_interpolated'

        self._operator = weakref.ref(operator)
        interpolation_dofs = np.array(interpolation_dofs, dtype=np.int32)
        self.interpolation_dofs = interpolation_dofs
        self.triangular = triangular
        self.least_squares = least_squares

        if len(interpolation_dofs) > 0:
            try:
                self.restricted_operator, self.source_dofs = operator.restricted(interpolation_dofs)
            except NotImplementedError:
                self.logger.warning('Operator has no "restricted" method. The full operator will be evaluated.')
                self._operator = operator
            interpolation_matrix = collateral_basis.dofs(interpolation_dofs)
            self.interpolation_matrix = interpolation_matrix
        self.collateral_basis = collateral_basis.copy()

    @property
    def operator(self):
        if hasattr(self, 'restricted_operator'):
            return self._operator()
        else:
            return self._operator

    def apply(self, U, mu=None):
        assert self.parameters.assert_compatible(mu)
        if len(self.interpolation_dofs) == 0:
            return self.range.zeros(len(U))

        if hasattr(self, 'restricted_operator'):
            U_dofs = NumpyVectorSpace.make_array(U.dofs(self.source_dofs))
            AU = self.restricted_operator.apply(U_dofs, mu=mu)
        else:
            AU = NumpyVectorSpace.make_array(self.operator.apply(U, mu=mu).dofs(self.interpolation_dofs))
        try:
            if self.least_squares:
                interpolation_coefficients = lstsq(self.interpolation_matrix, AU.to_numpy())[0]
            elif self.triangular:
                interpolation_coefficients = solve_triangular(self.interpolation_matrix, AU.to_numpy(),
                                                              lower=True, unit_diagonal=True)
            else:
                interpolation_coefficients = solve(self.interpolation_matrix, AU.to_numpy())
        except ValueError:  # this exception occurs when AU contains NaNs ...
            interpolation_coefficients = np.empty((len(self.collateral_basis), len(AU))) + np.nan
        return self.collateral_basis.lincomb(interpolation_coefficients)

    def jacobian(self, U, mu=None):
        assert self.parameters.assert_compatible(mu)

        if len(self.interpolation_dofs) == 0:
            if isinstance(self.source, NumpyVectorSpace) and isinstance(self.range, NumpyVectorSpace):
                return NumpyMatrixOperator(np.zeros((self.range.dim, self.source.dim)),
                                           solver=self._jacobian_solver,
                                           name=self.name + '_jacobian')
            else:
                return ZeroOperator(self.range, self.source, solver=self._jacobian_solver, name=self.name + '_jacobian')
        elif hasattr(self, 'operator'):
            return EmpiricalInterpolatedOperator(self.operator.jacobian(U, mu=mu), self.interpolation_dofs,
                                                 self.collateral_basis, self.triangular,
                                                 solver=self._jacobian_solver, name=self.name + '_jacobian')
        else:
            restricted_source = self.restricted_operator.source
            U_dofs = restricted_source.make_array(U.dofs(self.source_dofs))
            JU = self.restricted_operator.jacobian(U_dofs, mu=mu) \
                                         .apply(restricted_source.make_array(np.eye(len(self.source_dofs))))
            try:
                if self.triangular:
                    interpolation_coefficients = solve_triangular(self.interpolation_matrix, JU.to_numpy(),
                                                                  lower=True, unit_diagonal=True)
                else:
                    interpolation_coefficients = solve(self.interpolation_matrix, JU.to_numpy())
            except ValueError:  # this exception occurs when AU contains NaNs ...
                interpolation_coefficients = np.empty((len(self.collateral_basis), len(JU))) + np.nan
            J = self.collateral_basis.lincomb(interpolation_coefficients)
            if isinstance(J.space, NumpyVectorSpace):
                J = NumpyMatrixOperator(J.to_numpy())
            else:
                J = VectorArrayOperator(J)
            return ConcatenationOperator([J, ComponentProjectionOperator(self.source_dofs, self.source)],
                                         solver=self._jacobian_solver, name=self.name + '_jacobian')

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_operator']
        return d


class ProjectedEmpiricalInterpolatedOperator(Operator):
    """A projected |EmpiricalInterpolatedOperator|."""

    def __init__(self, restricted_operator, interpolation_matrix, source_basis_dofs,
                 projected_collateral_basis, triangular, least_squares=False, solver=None, name=None):

        name = name or f'{restricted_operator.name}_projected'

        self.__auto_init(locals())
        self.source = NumpyVectorSpace(len(source_basis_dofs))
        self.range = projected_collateral_basis.space
        self.linear = restricted_operator.linear
        if self.least_squares:
            self.least_squares_qr = qr(self.interpolation_matrix, mode='economic')
        elif not triangular:
            self.lu_factors = lu_factor(self.interpolation_matrix)

    def apply(self, U, mu=None):
        assert self.parameters.assert_compatible(mu)
        U_dofs = self.source_basis_dofs.lincomb(U.to_numpy())
        AU = self.restricted_operator.apply(U_dofs, mu=mu)
        try:
            if self.least_squares:
                Q, R = self.least_squares_qr
                interpolation_coefficients = solve_triangular(R, Q.T @ AU.to_numpy(), lower=False)
            elif self.triangular:
                interpolation_coefficients = solve_triangular(self.interpolation_matrix, AU.to_numpy(),
                                                              lower=True, unit_diagonal=True)
            else:
                interpolation_coefficients = lu_solve(self.lu_factors, AU.to_numpy(), check_finite=False)
        except ValueError:  # this exception occurs when AU contains NaNs ...
            interpolation_coefficients = np.empty((len(self.projected_collateral_basis), len(AU))) + np.nan
        return self.projected_collateral_basis.lincomb(interpolation_coefficients)

    def jacobian(self, U, mu=None):
        assert len(U) == 1
        assert self.parameters.assert_compatible(mu)

        if self.interpolation_matrix.shape[0] == 0:
            return NumpyMatrixOperator(np.zeros((self.range.dim, self.source.dim)), solver=self._jacobian_solver,
                                       name=self.name + '_jacobian')

        U_dofs = self.source_basis_dofs.lincomb(U.to_numpy()[:, 0])
        J = self.restricted_operator.jacobian(U_dofs, mu=mu).apply(self.source_basis_dofs)
        try:
            if self.triangular:
                interpolation_coefficients = solve_triangular(self.interpolation_matrix, J.to_numpy(),
                                                              lower=True, unit_diagonal=True)
            else:
                interpolation_coefficients = solve(self.interpolation_matrix, J.to_numpy())
        except ValueError:  # this exception occurs when J contains NaNs ...
            interpolation_coefficients = (np.empty((len(self.projected_collateral_basis),
                                                    len(self.source_basis_dofs)))
                                          + np.nan)
        M = self.projected_collateral_basis.lincomb(interpolation_coefficients)
        if isinstance(M.space, NumpyVectorSpace):
            return NumpyMatrixOperator(M.to_numpy(), solver=self._jacobian_solver)
        else:
            return VectorArrayOperator(M, solver=self._jacobian_solver)

    def with_cb_dim(self, dim):
        assert dim <= self.restricted_operator.range.dim

        interpolation_matrix = self.interpolation_matrix[:dim, :dim]

        restricted_operator, source_dofs = self.restricted_operator.restricted(np.arange(dim))

        old_pcb = self.projected_collateral_basis
        projected_collateral_basis = NumpyVectorSpace.make_array(old_pcb.to_numpy()[:, :dim])

        old_sbd = self.source_basis_dofs
        source_basis_dofs = NumpyVectorSpace.make_array(old_sbd.to_numpy()[source_dofs, :])

        return ProjectedEmpiricalInterpolatedOperator(restricted_operator, interpolation_matrix,
                                                      source_basis_dofs, projected_collateral_basis, self.triangular,
                                                      solver=self.solver, name=self.name)
