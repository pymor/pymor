# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.defaults import defaults


@defaults('alpha_init', 'tau', 'beta', 'maxiter')
def armijo(r, starting_point, correction, grad=None,
           alpha_init=1.0, tau=0.5, beta=0.0001, maxiter=100):
    """Armijo line search algorithm.

    This method computes a step size such that the Armijo-Goldstein condition is
    fulfilled.

    Parameters
    ----------
    r
        Real-valued function that can be evaluated for its value.
    starting_point
        A |VectorArray| of length 1 containing the starting point of the line search.
    correction
        Descent direction of `r` in the point `starting_point` along which the line
        search is performed.
    grad
        Gradient of `r` in the point `starting_point`.
    alpha_init
        Initial step size that is gradually reduced.
    tau
        The fraction by which the step size is reduced in each iteration.
    beta
        Control parameter to adjust the required decrease of the function value of `r`.
    maxiter
        Fail if the iteration count reaches this value without finding a point fulfilling
        the Armijo-Goldstein condition.

    Returns
    -------
    alpha
        Step size computed according to the Armijo-Goldstein condition.
    """
    assert alpha_init > 0
    assert 0 < tau < 1
    assert maxiter > 0

    # Start line search with step size of alpha_init
    alpha = alpha_init

    # Compute initial function value
    initial_residual = r(starting_point)

    iteration = 0
    slope = 0.0

    # Compute slope if gradient is provided
    if grad:
        slope = min(grad.dot(correction), 0.0)

    while True:
        # Compute new function value
        current_residual = r(starting_point + alpha * correction)
        # Check the Armijo-Goldstein condition
        if current_residual < initial_residual + alpha * beta * slope:
            break
        # Check if maxiter is reached
        if iteration >= maxiter:
            alpha = alpha_init
            break
        iteration += 1
        # Adjust step size
        alpha *= tau

    return alpha
