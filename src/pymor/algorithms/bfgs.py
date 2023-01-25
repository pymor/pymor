# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import warnings

import numpy as np

from pymor.algorithms.line_search import armijo
from pymor.core.defaults import defaults
from pymor.core.exceptions import BFGSError
from pymor.core.logger import getLogger
from pymor.parameters.base import Mu


def get_active_and_inactive_sets(parameter_space, mu, epsilon=1e-8):
    """Compute the active and inactive parameter index sets for constrained optimization.

    Parameters
    ----------
    parameter_space
        The |ParameterSpace| used for checking parameter ranges.
    mu
        The parameter to compute the active and inactive sets for.
    epsilon
        Tolerance threshold for checking boundary conditions.

    Returns
    -------
    active
        The set of active parameter indices.
    inactive
        The set of inactive parameter indices.
    """
    mu = mu if isinstance(mu, Mu) else parameter_space.parameters.parse(mu)
    active = np.array([])

    for key, dim in parameter_space.parameters.items():
        indices = np.zeros(dim)
        low_range = parameter_space.ranges[key][0] * np.ones(dim)
        high_range = parameter_space.ranges[key][1] * np.ones(dim)
        eps = epsilon * np.ones(dim)
        indices[np.where(np.logical_or(mu[key] - low_range <= eps, high_range - mu[key] <= eps))] = 1
        active = np.concatenate((active, indices))

    inactive = np.ones(active.shape) - active

    return active, inactive


def project_hessian(hessian, direction, active, inactive):
    """Project the Hessian matrix using an active and an inactive set.

    Parameters
    ----------
    hessian
        The Hessian matrix.
    direction
        The descent direction used to update the parameter.
    active
        The active set computed from the |ParameterSpace|.
    inactive
        The inactive set computed from the |ParameterSpace|.
    """
    direction_A = np.multiply(active, direction)
    direction_I = np.multiply(inactive, direction)
    return direction_A + np.multiply(inactive, hessian.dot(direction_I))


def update_hessian(hessian, mu, old_mu, gradient, old_gradient):
    """Update Hessian matrix using BFGS iteration.

    Parameters
    ----------
    hessian
        The current Hessian matrix.
    mu
        The current `mu` parameter.
    old_mu
        The previous `mu` parameter.
    gradient
        The current gradient with respect to the parameter.
    old_gradient
        The previous gradient with respect to the parameter.
    """
    gradient_difference = gradient - old_gradient
    mu_difference = mu - old_mu
    gradient_mu_coefficient = np.dot(gradient_difference, mu_difference)

    if gradient_mu_coefficient > 0.:
        hessian_grad_diff = np.dot(hessian, gradient_difference)
        quadratic_coefficient = np.dot(gradient_difference, hessian_grad_diff)

        hessian_part = np.outer(hessian_grad_diff, mu_difference) + np.outer(mu_difference, hessian_grad_diff)
        mu_part = np.outer(mu_difference, mu_difference)

        mu_coeff = (gradient_mu_coefficient + quadratic_coefficient) / gradient_mu_coefficient**2
        hessian_coeff = 1. / gradient_mu_coefficient

        hessian += mu_coeff * mu_part - hessian_coeff * hessian_part
    else:
        hessian = np.eye(hessian.shape[0])

    return hessian


@defaults('miniter', 'maxiter', 'atol', 'tol_sub', 'stagnation_window', 'stagnation_threshold')
def bfgs(model, parameter_space, initial_guess=None, miniter=0, maxiter=100, atol=1e-16,
         tol_sub=1e-8, line_search_params=None, stagnation_window=3, stagnation_threshold=np.inf,
         error_aware=False, error_criterion=None, beta=None, radius=None, return_stages=False):
    """BFGS algorithm.

    This method solves the optimization problem ::

        min J(mu), mu in C

    for an output functional depending on a box-constrained `mu` using the BFGS method.

    Parameters
    ----------
    model
        The |Model| used for the optimization.
    parameter_space
        The |ParameterSpace| for enforcing the box constraints on the parameter `mu`.
    initial_guess
        If not `None`, a |Mu| instance of length 1 containing an initial guess for the
        solution `mu`.
    miniter
        Minimum amount of iterations to perform.
    maxiter
        Fail if the iteration count reaches this value without converging.
    atol
        Finish when the absolute error measure is below this threshold.
    tol_sub
        Finish when the clipped parameter error measure is below this threshold.
    line_search_params
        Dictionary of additional parameters passed to the line search method.
    stagnation_window
        Finish when the parameter update has been stagnating within a tolerance of
        `stagnation_threshold` during the last `stagnation_window` iterations.
    stagnation_threshold
        See `stagnation_window`.
    error_aware
        If `True`, perform an additional error aware check during the line search phase.
        Intended for use with the trust region algorithm. Requires the parameters `beta`
        and `radius` to be set.
    beta
        Intended for use with the trust region algorithm. Indicates the factor for checking
        if the current value is close to the trust region boundary. See `error_aware`.
    radius
        Intended for use with the trust region algorithm. Indicates the radius of the
        current trust region. See `error_aware`.
    return_stages
        If `True`, return a `list` of the intermediate parameter values of `mu` after
        each iteration.

    Returns
    -------
    mu
        |Numpy array| containing the computed parameter.
    data
        Dict containing the following fields:

            :mus:             `list` of parameters after each iteration.
            :mu_norms:        |NumPy array| of the parameter norms after each iteration.
            :update_norms:    |NumPy array| of the norms of the update vectors for each iteration.
            :stages:          See `return_stages`.

    Raises
    ------
    BFGSError
        Raised if the BFGS algorithm failed to converge.
    """
    logger = getLogger('pymor.algorithms.bfgs')

    data = {}

    assert model.output_functional is not None
    output = lambda m: model.output(m)[0, 0]

    if initial_guess is None:
        initial_guess = parameter_space.sample_randomly(1)[0]
        mu = initial_guess.to_numpy()
    else:
        mu = initial_guess.to_numpy() if isinstance(initial_guess, Mu) else initial_guess

    if error_aware:
        assert error_criterion is not None and beta is not None and radius is not None
        assert callable(error_criterion)

    if return_stages:
        stages = []

    gradient = model.parameters.parse(model.output_d_mu(mu)).to_numpy()
    current_output = output(mu)
    eps = np.linalg.norm(gradient)

    # compute norms
    mu_norm = np.linalg.norm(mu)
    update_norms = []
    foc_norms = []
    line_search_iterations = []
    data['mus'] = [mu.copy()]

    hessian = np.eye(mu.size)

    first_order_criticity = output_diff = mu_diff = update_norm = 1e6
    iteration = 0
    while True:
        if iteration >= miniter:
            if output_diff < atol:
                logger.info(f'Absolute tolerance of {atol} for output error reached. Converged.')
                break
            if mu_diff < atol:
                logger.info(f'Absolute tolerance of {atol} for parameter difference reached. Converged.')
                break
            if first_order_criticity < tol_sub:
                logger.info(f'Absolute tolerance of {tol_sub} for first order criticity reached. Converged.')
                break
            if error_aware:
                if error_criterion(mu, current_output):
                    logger.info(f'Output error confidence reached for beta {beta} and radius {radius}. Converged.')
                    break
            if (iteration >= stagnation_window + 1 and not stagnation_threshold == np.inf
                    and all(np.isclose(
                        [max(update_norms[-stagnation_window - 1:]), min(update_norms[-stagnation_window - 1:])],
                        update_norm, atol=stagnation_threshold))):
                logger.info(f'Norm of update is stagnating (threshold: {stagnation_threshold:5e}, '
                            f'window: {stagnation_window}). Converged.')
                break
            if iteration >= maxiter:
                logger.info(f'Maximum iterations reached. Failed to converge after {iteration} iterations.')
                raise BFGSError

        iteration += 1

        # store convergence history
        if iteration > 0 and return_stages:
            stages.append(model.parameters.parse(mu))

        active, inactive = get_active_and_inactive_sets(parameter_space, mu, eps)

        # compute update to mu
        if sum(inactive) == 0.:
            direction = -gradient
        else:
            direction = project_hessian(hessian, -gradient, active, inactive)

        if error_aware:
            step_size, line_search_iteration = armijo(
                output, mu, direction, grad=gradient, initial_value=current_output,
                additional_criterion=error_criterion, **(line_search_params or {}))
        else:
            step_size, line_search_iteration = armijo(output, mu, direction, grad=gradient,
                               initial_value=current_output, **(line_search_params or {}))
        line_search_iterations.append(line_search_iteration + 1)

        # update mu
        old_mu = mu.copy()
        old_output = current_output.copy()
        mu_update = step_size * direction
        mu += mu_update
        mu = parameter_space.clip(mu).to_numpy()
        current_output = output(mu)
        data['mus'].append(mu.copy())

        # compute norms
        update_norm = np.linalg.norm(mu - old_mu)
        update_norms.append(update_norm)
        mu_norm = np.linalg.norm(mu)

        # update gradient
        old_gradient = gradient.copy()
        gradient = model.parameters.parse(model.output_d_mu(mu)).to_numpy()
        first_order_criticity = np.linalg.norm(mu - parameter_space.clip(mu - gradient).to_numpy())
        foc_norms.append(first_order_criticity)

        # set new active inactive threshhold
        eps_update_mu = mu - gradient
        eps = np.linalg.norm(eps_update_mu - mu)

        # update relative errors
        output_diff = abs(old_output - current_output) / abs(old_output)
        mu_diff = np.linalg.norm(mu - old_mu) / np.linalg.norm(old_mu)

        # update the hessian approximate
        hessian = update_hessian(hessian, mu, old_mu, gradient, old_gradient)

        with warnings.catch_warnings():
            # ignore division-by-zero warnings when solution_norm or output is zero
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            logger.info(f'it:{iteration} '
                        f'foc:{first_order_criticity:.3e} '
                        f'upd:{update_norm:.3e} '
                        f'rel_upd:{update_norm / mu_norm:.3e} ')

        if not np.isfinite(update_norm) or not np.isfinite(mu_norm):
            raise BFGSError('Failed to converge.')

    data['update_norms'] = np.array(update_norms)
    data['foc_norms'] = np.array(foc_norms)
    data['iterations'] = iteration
    data['line_search_iterations'] = np.array(line_search_iterations)
    if return_stages:
        data['stages'] = np.array(stages)

    return mu, data
