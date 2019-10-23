# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.defaults import defaults
from pymor.core.exceptions import InversionError, NewtonError
from pymor.core.logger import getLogger


@defaults('miniter', 'maxiter', 'rtol', 'atol', 'relax', 'stagnation_window', 'stagnation_threshold')
def newton(operator, rhs, initial_guess=None, mu=None, error_norm=None, least_squares=False,
           miniter=0, maxiter=100, rtol=-1., atol=-1., relax=1.,
           stagnation_window=3, stagnation_threshold=np.inf,
           return_stages=False, return_residuals=False):
    """Basic Newton algorithm.

    This method solves the nonlinear equation ::

        A(U, mu) = V

    for `U` using the Newton method.

    Parameters
    ----------
    operator
        The |Operator| `A`. `A` must implement the
        :meth:`~pymor.operators.interfaces.OperatorInterface.jacobian` interface method.
    rhs
        |VectorArray| of length 1 containing the vector `V`.
    initial_guess
        If not `None`, a |VectorArray| of length 1 containing an initial guess for the
        solution `U`.
    mu
        The |Parameter| for which to solve the equation.
    error_norm
        The norm with which the norm of the residual is computed. If `None`, the
        Euclidean norm is used.
    least_squares
        If `True`, use a least squares linear solver (e.g. for residual minimization).
    miniter
        Minimum amount of iterations to perform.
    maxiter
        Fail if the iteration count reaches this value without converging.
    rtol
        Finish when the residual norm has been reduced by this factor relative to the
        norm of the initial residual.
    atol
        Finish when the residual norm is below this threshold.
    relax
        Relaxation factor for Newton updates.
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

    err = residual.l2_norm()[0] if error_norm is None else error_norm(residual)[0]
    logger.info(f'      Initial Residual: {err:5e}')

    iteration = 0
    error_sequence = [err]
    while True:
        if iteration >= miniter:
            if err <= atol:
                logger.info(f'Absolute limit of {atol} reached. Stopping.')
                break
            if err/error_sequence[0] <= rtol:
                logger.info(f'Prescribed total reduction of {rtol} reached. Stopping.')
                break
            if (len(error_sequence) >= stagnation_window + 1
                    and err/max(error_sequence[-stagnation_window - 1:]) >= stagnation_threshold):
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
        U.axpy(relax, correction)
        residual = rhs - operator.apply(U, mu=mu)

        err = residual.l2_norm()[0] if error_norm is None else error_norm(residual)[0]
        logger.info(f'Iteration {iteration:2}: Residual: {err:5e},  '
                    f'Reduction: {err / error_sequence[-1]:5e}, Total Reduction: {err / error_sequence[0]:5e}')
        error_sequence.append(err)
        if not np.isfinite(err):
            raise NewtonError('Failed to converge')

    data['error_sequence'] = np.array(error_sequence)

    return U, data
