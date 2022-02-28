# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.algorithms.line_search import armijo

from pymor.core.defaults import defaults
from pymor.core.exceptions import InversionError, NewtonError
from pymor.core.logger import getLogger


@defaults('miniter', 'maxiter', 'rtol', 'atol', 'relax', 'stagnation_window', 'stagnation_threshold')
def newton(operator, rhs, initial_guess=None, mu=None, range_product=None, source_product=None, least_squares=False,
           miniter=0, maxiter=100, atol=0., rtol=1e-7, relax='armijo', line_search_params=None,
           stagnation_window=3, stagnation_threshold=np.inf, error_measure='update',
           return_stages=False, return_residuals=False):
    """Newton algorithm.

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
    range_product
        The inner product `Operator` on `operator.range` with which the norm
        of the resiudal is computed. If `None`, the Euclidean inner product
        is used.
    source_product
        The inner product `Operator` on `operator.source` with which the norm
        of the solution and update vectors is computed. If `None`, the Euclidean inner
        product is used.
    least_squares
        If `True`, use a least squares linear solver (e.g. for residual minimization).
    miniter
        Minimum amount of iterations to perform.
    maxiter
        Fail if the iteration count reaches this value without converging.
    atol
        Finish when the error measure is below this threshold.
    rtol
        Finish when the error measure has been reduced by this factor
        relative to the norm of the initial residual resp. the norm of the current solution.
    relax
        If real valued, relaxation factor for Newton updates; otherwise `'armijo'` to
        indicate that the :func:`~pymor.algorithms.line_search.armijo` line search algorithm
        shall be used.
    line_search_params
        Dictionary of additional parameters passed to the line search method.
    stagnation_window
        Finish when the error measure has not been reduced by a factor of
        `stagnation_threshold` during the last `stagnation_window` iterations.
    stagnation_threshold
        See `stagnation_window`.
    error_measure
        If `'residual'`, convergence depends on the norm of the residual. If
        `'update'`, convergence depends on the norm of the update vector.
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

            :solution_norms:  |NumPy array| of the solution norms after each iteration.
            :update_norms:    |NumPy array| of the norms of the update vectors for each iteration.
            :residual_norms:  |NumPy array| of the residual norms after each iteration.
            :stages:          See `return_stages`.
            :residuals:       See `return_residuals`.

    Raises
    ------
    NewtonError
        Raised if the Newton algorithm failed to converge.
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

    # compute norms
    solution_norm = U.norm(source_product)[0]
    solution_norms = [solution_norm]
    update_norms = []
    residual_norm = residual.norm(range_product)[0]
    residual_norms = [residual_norm]

    # select error measure for convergence criteria
    err = residual_norm if error_measure == 'residual' else np.inf
    err_scale_factor = err
    errs = residual_norms if error_measure == 'residual' else update_norms

    logger.info(f'     norm:{solution_norm:.3e}                                 res:{residual_norm:.3e}')

    iteration = 0
    while True:
        # check for convergence / failure of convergence
        if iteration >= miniter:
            if residual_norm == 0:
                # handle the corner case where error_norm == update, U is the exact solution
                # and the jacobian of operator is not invertible at the exact solution
                logger.info('Norm of residual exactly zero. Converged.')
                break
            if err < atol:
                logger.info(f'Absolute tolerance of {atol} for norm of {error_measure} reached. Converged.')
                break
            if err < rtol * err_scale_factor:
                logger.info(f'Relative tolerance of {rtol} for norm of {error_measure} reached. Converged.')
                break
            if (len(errs) >= stagnation_window + 1
                    and err > stagnation_threshold * max(errs[-stagnation_window - 1:])):
                logger.info(f'Norm of {error_measure} is stagnating (threshold: {stagnation_threshold:5e}, '
                            f'window: {stagnation_window}). Converged.')
                break
            if iteration >= maxiter:
                raise NewtonError(f'Failed to converge after {iteration} iterations.')

        iteration += 1

        # store convergence history
        if iteration > 0 and return_stages:
            data['stages'].append(U)
        if return_residuals:
            data['residuals'].append(residual)

        # compute update
        jacobian = operator.jacobian(U, mu=mu)
        try:
            update = jacobian.apply_inverse(residual, least_squares=least_squares)
        except InversionError as e:
            raise NewtonError('Could not invert jacobian.') from e

        # compute step size
        if isinstance(relax, Number):
            step_size = relax
        elif relax == 'armijo':
            def res(x):
                residual_vec = rhs - operator.apply(x, mu=mu)
                return residual_vec.norm(range_product)[0]

            if range_product is None:
                grad = - (jacobian.apply(residual) + jacobian.apply_adjoint(residual))
            else:
                grad = - (jacobian.apply_adjoint(range_product.apply(residual))
                          + jacobian.apply(range_product.apply_adjoint(residual)))
            step_size = armijo(res, U, update, grad=grad, initial_value=residual_norm, **(line_search_params or {}))
        else:
            raise ValueError('Unknown line search method.')

        # update solution and residual
        U.axpy(step_size, update)
        residual = rhs - operator.apply(U, mu=mu)

        # compute norms
        solution_norm = U.norm(source_product)[0]
        solution_norms.append(solution_norm)
        update_norm = update.norm(source_product)[0] * step_size
        update_norms.append(update_norm)
        residual_norm = residual.norm(range_product)[0]
        residual_norms.append(residual_norm)

        # select error measure for next iteration
        err = residual_norm if error_measure == 'residual' else update_norm
        if error_measure == 'update':
            err_scale_factor = solution_norm

        logger.info(f'it:{iteration} '
                    f'norm:{solution_norm:.3e} '
                    f'upd:{update_norm:.3e} '
                    f'rel_upd:{update_norm / solution_norm:.3e} '
                    f'res:{residual_norm:.3e} '
                    f'red:{residual_norm / residual_norms[-2]:.3e} '
                    f'tot_red:{residual_norm / residual_norms[0]:.3e}')

        if not np.isfinite(residual_norm) or not np.isfinite(solution_norm):
            raise NewtonError('Failed to converge.')

    logger.info('')

    data['solution_norms'] = np.array(solution_norms)
    data['update_norms']   = np.array(update_norms)
    data['residual_norms'] = np.array(residual_norms)

    return U, data
