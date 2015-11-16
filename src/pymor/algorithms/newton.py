# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core.defaults import defaults
from pymor.core.exceptions import InversionError, NewtonError
from pymor.core.logger import getLogger


@defaults('miniter', 'maxiter', 'rtol', 'atol', 'stagnation_window', 'stagnation_threshold')
def newton(operator, rhs, initial_guess=None, mu=None, error_norm=None, least_squares=False,
           miniter=0, maxiter=100, rtol=-1., atol=-1.,
           stagnation_window=3, stagnation_threshold=0.9,
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
        If `None`, a |VectorArray| of length 1 containing an initial guess for the solution
        `U`.
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
        Finish if the residual norm has been reduced by this factor relative to the
        norm of the initial residual.
    atol
        Finish if the residual norm is below this threshold.
    stagnation_window
        Finish if the residual norm has not been reduced by a factor of
        `stagnation_threshold` during the last `stagnation_window` iterations.
    stagnation_threshold
        See `stagnation_window`.
    return_stages
        If `True` return a |VectorArray| of the approximations of `U` after each iteration
        in the `data` dict.
    return_residuals
        If `True` return a |VectorArray| of all residual vectors which have been computed
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
    logger.info('      Initial Residual: {:5e}'.format(err))

    iteration = 0
    error_sequence = [err]
    while (iteration < miniter
           or (iteration < maxiter
               and err > atol and err/error_sequence[0] > rtol
               and (len(error_sequence) < stagnation_window + 1
                    or err/max(error_sequence[-stagnation_window - 1:]) < stagnation_threshold))):
        if iteration > 0 and return_stages:
            data['stages'].append(U)
        if return_residuals:
            data['residuals'].append(residual)
        iteration += 1
        jacobian = operator.jacobian(U, mu=mu)
        try:
            correction = jacobian.apply_inverse(residual, options='least_squares' if least_squares else None)
        except InversionError:
            raise NewtonError('Could not invert jacobian')
        U += correction
        residual = rhs - operator.apply(U, mu=mu)

        err = residual.l2_norm()[0] if error_norm is None else error_norm(residual)[0]
        logger.info('Iteration {:2}: Residual: {:5e},  Reduction: {:5e}, Total Reduction: {:5e}'
                    .format(iteration, err, err / error_sequence[-1], err / error_sequence[0]))
        error_sequence.append(err)

    if err <= atol:
        logger.info('Absolute limit of {} reached. Stopping.'.format(atol))
    elif err/error_sequence[0] <= rtol:
        logger.info('Prescribed total reduction of {} reached. Stopping.'.format(rtol))
    elif (len(error_sequence) >= stagnation_window + 1
          and err/max(error_sequence[-stagnation_window - 1:]) >= stagnation_threshold):
        logger.info('Error is stagnating (threshold: {:5e}, window: {}). Stopping.'.format(stagnation_threshold,
                                                                                           stagnation_window))
    else:
        raise NewtonError('Failed to converge')

    data['error_sequence'] = np.array(error_sequence)

    return U, data
