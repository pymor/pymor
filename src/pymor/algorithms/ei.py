# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number
import math as m

import numpy as np
from scipy.linalg import solve_triangular

from pymor.core import getLogger, BasicInterface
from pymor.core.cache import Cachable, cached, DEFAULT_DISK_CONFIG
from pymor.la import VectorArrayInterface
from pymor.operators.ei import EmpiricalInterpolatedOperator


def generate_ei_data(evaluations, error_norm=None, target_error=None, max_interpolation_dofs=None,
                     projection='orthogonal', product=None):

    assert projection in ('orthogonal', 'ei')
    assert projection != 'orthogonal' or product is not None
    assert isinstance(evaluations, VectorArrayInterface) or all(isinstance(ev, VectorArrayInterface) for ev in evaluations)
    if isinstance(evaluations, VectorArrayInterface):
        evaluations = (evaluations,)

    logger = getLogger('pymor.algorithms.ei.generate_ei_data')
    logger.info('Generating Interpolation Data ...')

    interpolation_dofs = np.zeros((0,), dtype=np.int32)
    collateral_basis = type(next(iter(evaluations))).empty(dim=next(iter(evaluations)).dim)
    max_errs = []

    def interpolate(U, collateral_basis, interpolation_dofs, interpolation_matrix, ind=None):
        coefficients = solve_triangular(interpolation_matrix, U.components(interpolation_dofs, ind=ind).T,
                                        lower=True, unit_diagonal=True).T
        # coefficients = np.linalg.solve(interpolation_matrix, U.components(interpolation_dofs, ind=ind).T).T
        return collateral_basis.lincomb(coefficients)

    while True:
        max_err = -1.
        if projection == 'orthogonal' and len(interpolation_dofs) > 0:
            if product is None:
                gramian = collateral_basis.gramian()
            else:
                gramian = product.apply2(collateral_basis, collateral_basis, pairwise=False)
            gramian_inverse = np.linalg.inv(gramian)

        for AU in evaluations:
            if len(interpolation_dofs) > 0:
                if projection == 'ei':
                    AU_interpolated = interpolate(AU, collateral_basis, interpolation_dofs, interpolation_matrix)
                    ERR = AU - AU_interpolated
                else:
                    if product is None:
                        coefficients = gramian_inverse.dot(collateral_basis.dot(AU, pairwise=False)).T
                    else:
                        gramian = product
                        coefficients = gramian_inverse.dot(product.apply2(collateral_basis, AU, pairwise=False)).T
                    AU_projected = collateral_basis.lincomb(coefficients)
                    ERR = AU - AU_projected
            else:
                ERR = AU
            errs = discretization.l2_norm(ERR) if error_norm is None else error_norm(ERR)
            local_max_err_ind = np.argmax(errs)
            local_max_err = errs[local_max_err_ind]
            if local_max_err > max_err:
                max_err = local_max_err
                if len(interpolation_dofs) == 0 or projection == 'ei':
                    new_vec = ERR.copy(ind=local_max_err_ind)
                else:
                    new_vec = AU.copy(ind=local_max_err_ind)
                    new_vec -= interpolate(AU, collateral_basis, interpolation_dofs, interpolation_matrix,
                                           ind=local_max_err_ind)

        logger.info('Maximum interpolation error with {} interpolation DOFs: {}'.format(len(interpolation_dofs),
                                                                                        max_err))
        if target_error is not None and max_err <= target_error:
            logger.info('Target error reached! Stopping extension loop.')
            break

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
            break

        logger.info('')

    data = {'errors': max_errs}

    return interpolation_dofs, collateral_basis, data


def interpolate_operators(discretization, operator_names, parameter_sample, error_norm=None,
                          target_error=None, max_interpolation_dofs=None,
                          projection='orthogonal', product=None, separately=False):


    class EvaluationProvider(BasicInterface, Cachable):

        # evil hack to prevent deadlock ...
        from tempfile import gettempdir
        from os.path import join
        DEFAULT_MEMORY_CONFIG = {"backend": 'LimitedMemory', 'arguments.max_kbytes': 20000}
        DISK_CONFIG = {"backend": 'LimitedFile',
                       "arguments.filename": join(gettempdir(), 'pymor.ei_cache.dbm'),
                       'arguments.max_keys': 2000}

        def __init__(self, discretization, operator, sample, operator_sample):
            Cachable.__init__(self, config=self.DEFAULT_MEMORY_CONFIG)
            self.discretization = discretization
            self.sample = sample
            self.operator = operator
            self.operator_sample = operator_sample

        @cached
        def data(self, k):
            from scipy.io import loadmat
            mu = self.sample[k]
            mu_op = self.operator_sample[k]
            return self.operator.apply(self.discretization.solve(mu=mu), mu=mu_op)

        def __len__(self):
            return len(self.sample)

        def __getitem__(self, ind):
            if not 0 <= ind < len(self.sample):
                raise IndexError
            return self.data(ind)

    if isinstance(operator_names, str):
        operator_names = (operator_names,)

    if len(operator_names) > 1:
        raise NotImplementedError

    sample = tuple(parameter_sample)
    operator_sample = tuple(discretization.map_parameter(mu, operator_names[0]) for mu in sample)
    operator = discretization.operators[operator_names[0]]

    evaluations = EvaluationProvider(discretization, operator, sample, operator_sample)
    dofs, basis, data = generate_ei_data(evaluations, error_norm, target_error, max_interpolation_dofs,
                                         projection=projection, product=product)

    ei_operator = EmpiricalInterpolatedOperator(operator, dofs, basis)
    ei_operators = discretization.operators.copy()
    ei_operators['operator'] = ei_operator
    ei_discretization = discretization.with_(operators=ei_operators, name='{}_interpolated'.format(discretization.name))

    return ei_discretization, data
