# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor import defaults
from pymor.core import getLogger
from pymor.core.exceptions import InversionError, NewtonError


def newton(operator, rhs, initial_guess=None, mu=None, error_norm=None,
           maxiter=None, reduction=None, abs_limit=None,
           return_stages=False, return_residuals=False):
    maxiter = defaults.newton_maxiter if maxiter is None else maxiter
    reduction = defaults.newton_reduction if reduction is None else reduction
    abs_limit = defaults.newton_abs_limit if abs_limit is None else abs_limit
    logger = getLogger('pymor.algorithms.newton')

    data = {}

    if initial_guess is None:
        initial_guess = operator.type_source.zeros(operator.dim_source)

    if return_stages:
        data['stages'] = operator.type_source.empty(operator.dim_source)

    if return_residuals:
        data['residuals'] = operator.type_range.empty(operator.dim_range)

    U = initial_guess.copy()
    residual = rhs - operator.apply(U, mu=mu)

    err = first_err = residual.l2_norm()[0] if error_norm is None else error_norm(residual)[0]
    logger.info('      Initial Residual: {:5e}'.format(err))

    iteration = 0
    while iteration < maxiter and err > abs_limit and err/first_err > reduction:  # or iteration == 0:
        if iteration > 0 and return_stages:
            data['stages'].append(U)
        if return_residuals:
            data['residuals'].append(residual)
        iteration += 1
        jacobian = operator.jacobian(U, mu=mu)
        try:
            correction = jacobian.apply_inverse(residual)
        except InversionError:
            raise NewtonError('Could not invert jacobian')
        U += correction
        residual = rhs - operator.apply(U, mu=mu)

        old_err = err
        err = residual.l2_norm()[0] if error_norm is None else error_norm(residual)[0]
        logger.info('Iteration {:2}: Residual: {:5e},  Reduction: {:5e}, Total Reduction: {:5e}'
                    .format(iteration, err, err / old_err, err / first_err))

    if err > abs_limit and err/first_err > reduction:
        raise NewtonError('Failed to converge')

    return U, data
