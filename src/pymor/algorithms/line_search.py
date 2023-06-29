# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.defaults import defaults
from pymor.core.logger import getLogger


@defaults('alpha_init', 'tau', 'beta', 'maxiter')
def armijo(f, starting_point, direction, grad=None, initial_value=None, alpha_init=1.0, tau=0.5, beta=0.0001,
           maxiter=10):
    """Unconstrained Armijo line search algorithm.

    This method computes a step size such that the unconstrained Armijo condition
    (see :cite:`NW06`, p. 33) is fulfilled.

    Parameters
    ----------
    f
        Real-valued function that can be evaluated for its value.
    starting_point
        A |VectorArray| of length 1 containing the starting point of the line search.
    direction
        Descent direction along which the line search is performed.
    grad
        Gradient of `f` in the point `starting_point` as either a |VectorArray| or a |NumPy array|.
    initial_value
        Value of `f` in the point `starting_point`.
    tau
        The fraction by which the step size is reduced in each iteration.
    beta
        Control parameter to adjust the required decrease of the function value of `f`.
    maxiter
        Use `alpha_init` as default if the iteration count reaches this value without
        finding a point fulfilling the Armijo condition.

    Returns
    -------
    alpha
        Step size computed according to the Armijo condition.
    iterations
        Number of total Armijo line search iterations.
    """
    def _unconstrained_armijo_condition(starting_point, initial_value, current_value, alpha, beta,
                                        direction, slope):
        return current_value < initial_value + alpha * beta * slope

    return _armijo(f, starting_point, direction, armijo_condition=_unconstrained_armijo_condition, grad=grad,
                   initial_value=initial_value, alpha_init=alpha_init, tau=tau, beta=beta, maxiter=maxiter)


@defaults('alpha_init', 'tau', 'beta', 'maxiter')
def constrained_armijo(f, starting_point, direction, armijo_condition=None, grad=None, initial_value=None,
                       alpha_init=1.0, tau=0.5, beta=0.0001, maxiter=10):
    """Constrained Armijo line search algorithm.

    This method computes a step size such that the given constrained Armijo condition
    is fulfilled.

    Parameters
    ----------
    f
        Real-valued function that can be evaluated for its value.
    starting_point
        A |VectorArray| of length 1 containing the starting point of the line search.
    direction
        Descent direction along which the line search is performed.
    armijo_condition
        A `callable` used as the Armijo termination condition during the step length computation.
    grad
        Gradient of `f` in the point `starting_point` as either a |VectorArray| or a |NumPy array|.
    initial_value
        Value of `f` in the point `starting_point`.
    tau
        The fraction by which the step size is reduced in each iteration.
    beta
        Control parameter to adjust the required decrease of the function value of `f`.
    maxiter
        Use `alpha_init` as default if the iteration count reaches this value without
        finding a point fulfilling the Armijo condition.

    Returns
    -------
    alpha
        Step size computed according to the Armijo condition.
    iterations
        Number of total Armijo line search iterations.
    """
    assert armijo_condition is not None, 'Please provide a callable Armijo condition!'

    return _armijo(f, starting_point, direction, armijo_condition=armijo_condition, grad=grad,
                   initial_value=initial_value, alpha_init=alpha_init, tau=tau, beta=beta, maxiter=maxiter)


@defaults('alpha_init', 'tau', 'beta', 'maxiter')
def _armijo(f, starting_point, direction, armijo_condition, grad=None, initial_value=None, alpha_init=1.0, tau=0.5,
            beta=0.0001, maxiter=10):
    """Armijo line search algorithm.

    This method computes a step size such that the Armijo condition (see :cite:`NW06`, p. 33)
    is fulfilled.

    Not intended to be used directly.

    Parameters
    ----------
    f
        Real-valued function that can be evaluated for its value.
    starting_point
        A |VectorArray| of length 1 containing the starting point of the line search.
    direction
        Descent direction along which the line search is performed.
    armijo_condition
        A `callable` used as the Armijo termination condition during the step length computation.
    grad
        Gradient of `f` in the point `starting_point` as either a |VectorArray| or a |NumPy array|.
    initial_value
        Value of `f` in the point `starting_point`.
    alpha_init
        Initial step size that is gradually reduced.
    tau
        The fraction by which the step size is reduced in each iteration.
    beta
        Control parameter to adjust the required decrease of the function value of `f`.
    maxiter
        Use `alpha_init` as default if the iteration count reaches this value without
        finding a point fulfilling the Armijo condition.

    Returns
    -------
    alpha
        Step size computed according to the Armijo condition.
    iterations
        Number of total Armijo line search iterations.
    """
    assert alpha_init > 0
    assert 0 < tau < 1
    assert maxiter > 0
    assert armijo_condition is not None
    assert callable(armijo_condition)

    # Start line search with step size of alpha_init
    alpha = alpha_init

    # Compute initial function value
    if initial_value is None:
        initial_value = f(starting_point)

    iterations = 1
    slope = 0.0

    # Compute slope if gradient is provided
    if grad is not None:
        if isinstance(grad, np.ndarray):
            slope = min(np.inner(grad, direction), 0.0)
        else:
            slope = min(grad.inner(direction), 0.0)

    while True:
        # Compute new function value
        current_value = f(starting_point + alpha * direction)
        # Check the Armijo condition
        if armijo_condition(starting_point, initial_value, current_value, alpha, beta, direction, slope):
            break

        # Check if maxiter is reached
        if iterations > maxiter:
            # Use default value as step size
            alpha = alpha_init
            # Log warning
            logger = getLogger('pymor.algorithms.line_search.armijo')
            logger.warning('Reached maximum number of line search steps; '
                           f'using initial step size of {alpha_init} instead.')
            break
        iterations += 1
        # Adjust step size
        alpha *= tau

    return alpha, iterations
