# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.algorithms.line_search import armijo

from pymor.core.defaults import defaults
from pymor.core.exceptions import InversionError, NewtonError
from pymor.core.logger import getLogger


@defaults('miniter', 'maxiter', 'rtol', 'atol', 'relax', 'stagnation_window', 'stagnation_threshold')
def newton(operator, rhs, initial_guess=None, mu=None, error_product=None, least_squares=False,
           miniter=0, maxiter=100, atol=0., rtol=0., relax=1., line_search_params=None,
           stagnation_window=3, stagnation_threshold=np.inf, error_measure='residual',
           return_stages=False, return_residuals=False):
    """Basic Newton algorithm.

    This method solves the nonlinear equation ::

        A(U, mu) = V

    for `U` using the Newton method.

    Parameters
    ----------
    operator
        The |Operator| `A`. `A` must implement the
        :meth:`~pymor.operators.interface.Operator.jacobian` interface method.
    rhs
        |VectorArray| of length 1 containing the vector `V`.
    initial_guess
        If not `None`, a |VectorArray| of length 1 containing an initial guess for the
        solution `U`.
    mu
        The |parameter values| for which to solve the equation.
    error_norm
        The product with which the norm of the residual is computed. If `None`, the
        Euclidean product is used.
    least_squares
        If `True`, use a least squares linear solver (e.g. for residual minimization).
    miniter
        Minimum amount of iterations to perform.
    maxiter
        Fail if the iteration count reaches this value without converging.
    atol
        Finish when the residual norm is below this threshold.
    rtol
        Finish when the residual norm has been reduced by this factor relative to the
        norm of the initial residual.
    relax
        If real valued, relaxation factor for Newton updates; otherwise 'armijo' to
        indicate that the :func:~pymor.algorithms.line_search.armijo line search algorithm
        shall be used.
    line_search_params
        Dictionary of additional parameters passed to the line search method.
    error_measure
        If 'resdiual', convergence depends on the norm of the residual. If
        'update', convergence depends on the norm of the update vector.
    stagnation_window
        Finish when the residual norm has not been reduced by a factor of
        `stagnation_threshold` during the last `stagnation_window` iterations.
    stagnation_threshold
        See `stagnation_window`.
    return_stages
        If `True`, return a |VectorArray| of the intermediate approximations of `U`
        after each iteration.
    return_residuals
        If `True`, return a |VectorArray| of all residual vectors which have been computed
        during the Newton iterations.

    Returns
    -------
    U
        |VectorArray| of length 1 containing the computed solution
    data
        Dict containing the following fields:

            :error_sequence:  |NumPy array| containing the residual norms after each iteration.
            :stages:          See `return_stages`.
            :residuals:       See `return_residuals`.

    Raises
    ------
    NewtonError
        Raised if the Netwon algorithm failed to converge.
    """
    assert error_measure in ('residual', 'update')

    logger = getLogger('pymor.algorithms.newton')

    data = {}

    if initial_guess is None:
        initial_guess = operator.source.zeros()

    if return_stages:
        data['stages'] = operator.source.empty()

    if return_residuals:
        data['residuals'] = operator.range.empty()

    U = initial_guess.copy()
    residual = rhs - operator.apply(U, mu=mu)

    solution_norm = U.norm(error_product)[0]
    solution_norms = [solution_norm]
    correction_norms = [solution_norm]
    residual_norm = residual.norm(error_product)[0]
    residual_norms = [residual_norm]
    err = residual_norm if error_measure == 'residual' else solution_norm
    err_scale_factor = err
    logger.info(f'      Initial Residual: {residual_norm:5e}')

    iteration = 0
    error_sequence = residual_norms if error_measure == 'residual' else correction_norms
    while True:
        if iteration >= miniter:
            if err < atol:
                logger.info(f'Absolute limit of {atol} reached. Stopping.')
                break
            if residual_norm < rtol * err_scale_factor:
                logger.info(f'Prescribed total reduction of {rtol} reached. Stopping.')
                break
            if (len(error_sequence) >= stagnation_window + 1
                    and err > stagnation_threshold * max(error_sequence[-stagnation_window - 1:])):
                logger.info(f'Error is stagnating (threshold: {stagnation_threshold:5e}, window: {stagnation_window}). '
                            f'Stopping.')
                break
            if iteration >= maxiter:
                raise NewtonError('Failed to converge')
        if iteration > 0 and return_stages:
            data['stages'].append(U)
        if return_residuals:
            data['residuals'].append(residual)
        iteration += 1
        jacobian = operator.jacobian(U, mu=mu)
        try:
            correction = jacobian.apply_inverse(residual, least_squares=least_squares)
        except InversionError:
            raise NewtonError('Could not invert jacobian')
        if isinstance(relax, Number):
            step_size = relax
        elif relax == 'armijo':
            logger.info(f'Using Armijo as line search method')
            def res(x):
                residual_vec = rhs - operator.apply(x, mu=mu)
                return residual_vec.norm(error_product)[0]
            if error_product is None:
                grad = - (jacobian.apply(residual) + jacobian.apply_adjoint(residual))
            else:
                grad = - (jacobian.apply_adjoint(error_product.apply(residual)) + jacobian.apply(error_product.apply_adjoint(residual)))
            step_size = armijo(res, U, correction, grad=grad, initial_value=residual_norm, **(line_search_params or {}))
        else:
            raise ValueError('Unknown line search method')
        U.axpy(step_size, correction)
        residual = rhs - operator.apply(U, mu=mu)

        solution_norm = U.norm(error_product)[0]
        solution_norms.append(solution_norm)
        correction_norm = correction.norm(error_product)[0]
        correction_norms.append(correction_norm)
        residual_norm = residual.norm(error_product)[0]
        residual_norms.append(residual_norm)
        err = residual_norm if error_measure == 'residual' else correction_norm
        if error_measure == 'update':
            err_scale_factor = solution_norm
        logger.info(f'Iteration {iteration:2}: Residual: {residual_norm:5e},  '
                    f'Reduction: {residual_norm / residual_norms[-2]:5e}, Total Reduction: {residual_norm / residual_norms[0]:5e}')
        if not np.isfinite(residual_norm):
            raise NewtonError('Failed to converge')

    data['error_sequence'] = np.array(error_sequence)
    data['solution_norms'] = np.array(solution_norms)
    data['update_norms']   = np.array(correction_norms)
    data['residual_norms'] = np.array(residual_norms)

    return U, data
