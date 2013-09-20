# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number
import math as m

import numpy as np
from scipy.linalg import solve_triangular, cho_factor, cho_solve

from pymor.core import getLogger, BasicInterface
from pymor.core.cache import CacheableInterface, cached
from pymor.la import VectorArrayInterface
from pymor.operators.ei import EmpiricalInterpolatedOperator


def ei_greedy(evaluations, error_norm=None, target_error=None, max_interpolation_dofs=None,
              projection='orthogonal', product=None):

    assert projection in ('orthogonal', 'ei')
    assert isinstance(evaluations, VectorArrayInterface) or all(isinstance(ev, VectorArrayInterface) for ev in evaluations)
    if isinstance(evaluations, VectorArrayInterface):
        evaluations = (evaluations,)

    logger = getLogger('pymor.algorithms.ei.ei_greedy')
    logger.info('Generating Interpolation Data ...')

    interpolation_dofs = np.zeros((0,), dtype=np.int32)
    interpolation_matrix = np.zeros((0,0))
    collateral_basis = type(next(iter(evaluations))).empty(dim=next(iter(evaluations)).dim)
    gramian_inverse = None
    max_errs = []

    def interpolate(U, ind=None):
        coefficients = solve_triangular(interpolation_matrix, U.components(interpolation_dofs, ind=ind).T,
                                        lower=True, unit_diagonal=True).T
        # coefficients = np.linalg.solve(interpolation_matrix, U.components(interpolation_dofs, ind=ind).T).T
        return collateral_basis.lincomb(coefficients)

    # compute the maximum projection error and error vector for the current interpolation data
    def projection_error():
        max_err = -1.

        # precompute gramian_inverse if needed
        if projection == 'orthogonal' and len(interpolation_dofs) > 0:
            if product is None:
                gramian = collateral_basis.gramian()
            else:
                gramian = product.apply2(collateral_basis, collateral_basis, pairwise=False)
            gramian_cholesky = cho_factor(gramian, overwrite_a=True)

        for AU in evaluations:
            if len(interpolation_dofs) > 0:
                if projection == 'ei':
                    AU_interpolated = interpolate(AU)
                    ERR =  AU - AU_interpolated
                else:
                    if product is None:
                        coefficients = cho_solve(gramian_cholesky, collateral_basis.dot(AU, pairwise=False)).T
                    else:
                        coefficients = cho_solve(gramian_cholesky, product.apply2(collateral_basis, AU, pairwise=False)).T
                    AU_projected = collateral_basis.lincomb(coefficients)
                    ERR = AU - AU_projected
            else:
                ERR = AU
            errs = ERR.l2_norm() if error_norm is None else error_norm(ERR)
            local_max_err_ind = np.argmax(errs)
            local_max_err = errs[local_max_err_ind]
            if local_max_err > max_err:
                max_err = local_max_err
                if len(interpolation_dofs) == 0 or projection == 'ei':
                    new_vec = ERR.copy(ind=local_max_err_ind)
                else:
                    new_vec = AU.copy(ind=local_max_err_ind)
                    new_vec -= interpolate(AU, ind=local_max_err_ind)

        return max_err, new_vec

    # main loop
    while True:
        max_err, new_vec = projection_error()

        logger.info('Maximum interpolation error with {} interpolation DOFs: {}'.format(len(interpolation_dofs),
                                                                                        max_err))
        if target_error is not None and max_err <= target_error:
            logger.info('Target error reached! Stopping extension loop.')
            break

        # compute new interpolation dof and collateral basis vector
        new_dof = new_vec.amax()[0]
        if new_dof in interpolation_dofs:
            logger.info('DOF {} selected twice for interplation! Stopping extension loop.'.format(new_dof))
            break
        new_vec *= 1 / new_vec.components([new_dof])[0]
        interpolation_dofs = np.hstack((interpolation_dofs, new_dof))
        collateral_basis.append(new_vec, remove_from_other=True)
        interpolation_matrix = collateral_basis.components(interpolation_dofs).T
        max_errs.append(max_err)

        triangularity_error = np.max(np.abs(interpolation_matrix - np.tril(interpolation_matrix)))
        logger.info('Interpolation matrix is not lower triangular with maximum error of {}'
                    .format(triangularity_error))

        if len(interpolation_dofs) >= max_interpolation_dofs:
            logger.info('Maximum number of interpolation DOFs reached. Stopping extension loop.')
            max_err, _ = projection_error()
            logger.info('Final maximum interpolation error with {} interpolation DOFs: {}'.format(
                len(interpolation_dofs), max_err))
            break

        logger.info('')

    data = {'errors': max_errs}

    return interpolation_dofs, collateral_basis, data


# This class provides cached evaulations of the operator on the solutions.
# Should be replaced by something simpler in the future.
class EvaluationProvider(CacheableInterface):

    def __init__(self, discretization, operator, sample, caching='memory'):
        CacheableInterface.__init__(self, region=caching)
        self.discretization = discretization
        self.sample = sample
        self.operator = operator

    @cached
    def data(self, k):
        mu = self.sample[k]
        return self.operator.apply(self.discretization.solve(mu), mu=mu)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, ind):
        if not 0 <= ind < len(self.sample):
            raise IndexError
        return self.data(ind)


def interpolate_operators(discretization, operator_name, parameter_sample, error_norm=None,
                          target_error=None, max_interpolation_dofs=None,
                          projection='orthogonal', product=None):


    sample = tuple(parameter_sample)
    operator = discretization.operators[operator_name]

    evaluations = EvaluationProvider(discretization, operator, sample)
    dofs, basis, data = ei_greedy(evaluations, error_norm, target_error, max_interpolation_dofs,
                                  projection=projection, product=product)

    ei_operator = EmpiricalInterpolatedOperator(operator, dofs, basis)
    ei_operators = discretization.operators.copy()
    ei_operators[operator_name] = ei_operator
    ei_discretization = discretization.with_(operators=ei_operators, name='{}_ei'.format(discretization.name))

    data.update({'dofs': dofs, 'basis': basis})
    return ei_discretization, data
