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


@defaults('miniter', 'maxiter', 'rtol_output', 'rtol_mu', 'tol_sub', 'stagnation_window', 'stagnation_threshold')
def error_aware_bfgs(model, parameter_space=None, initial_guess=None, miniter=0, maxiter=100, rtol_output=1e-16,
         rtol_mu=1e-16, tol_sub=1e-8, line_search_params=None, stagnation_window=3, stagnation_threshold=np.inf,
         error_aware=False, error_criterion=None):
    """BFGS algorithm.

    This method solves the optimization problem ::

        min J(mu), mu in C

    for an output functional depending on a box-constrained `mu` using the BFGS method.

    In contrast to :func:`scipy.optimize.minimize` with the `L-BFGS-B` methods, this BFGS
    implementation is explicitly designed to work with an error estimator. In particular, this
    implementation terminates if the higher level TR boundary from :mod:`pymor.algorithms.tr` is
    reached instead of continuing to optimize close to the boundary.

    Parameters
    ----------
    model
        The |Model| used for the optimization.
    parameter_space
        If not `None`, the |ParameterSpace| for enforcing the box constraints on the
        |parameter values| `mu`. Otherwise a |ParameterSpace| with infinite bounds.
    initial_guess
        If not `None`, a |Mu| instance containing an initial guess for the solution `mu`.
        Otherwise, random |parameter values| from the parameter space are chosen as the
        initial value.
    miniter
        Minimum amount of iterations to perform.
    maxiter
        Fail if the iteration count reaches this value without converging.
    rtol_output
        Finish when the relative error measure of the output is below this threshold.
    rtol_mu
        Finish when the relative error measure of the |parameter values| is below this threshold.
    tol_sub
        Finish when the first order criticality is below this threshold.
    line_search_params
        Dictionary of additional parameters passed to the Armijo line search method.
    stagnation_window
        Finish when the parameter update has not been enlarged by a factor of
        `stagnation_threshold` during the last `stagnation_window` iterations.
    stagnation_threshold
        See `stagnation_window`.
    error_aware
        If `True`, perform an additional error aware check during the line search phase.
        Intended for use with the trust region algorithm.
    error_criterion
        The additional error criterion used to check model confidence in the line search.
        This maps |parameter values| and an output value to a boolean indicating if the
        criterion is fulfilled. Refer to :func:`error_aware_line_search_criterion` in
        :mod:`pymor.algorithms.tr` for an example.

    Returns
    -------
    mu
        |Numpy array| containing the computed |parameter values|.
    data
        Dict containing the following fields:

            :mus:                       `list` of |parameter values| after each iteration.
            :foc_norms:                 |NumPy array| of the first order criticality norms
                                        after each iteration.
            :update_norms:              |NumPy array| of the norms of the update vectors
                                        after each iteration.
            :iterations:                Number of total BFGS iterations.
            :line_search_iterations:    |NumPy array| of the number of line search
                                        iterations per BFGS iteration.

    Raises
    ------
    BFGSError
        Raised if the BFGS algorithm failed to converge.
    """
    logger = getLogger('pymor.algorithms.bfgs.error_aware_bfgs')

    data = {}

    assert model.output_functional is not None
    assert model.output_functional.range.dim == 1
    output = lambda m: model.output(m)[0, 0]

    if parameter_space is None:
        logger.warn('No parameter space given. Assuming uniform parameter bounds of (-1, 1).')
        parameter_space = model.parameters.space(-1., 1.)

    if initial_guess is None:
        initial_guess = parameter_space.sample_randomly(1)[0]
        mu = initial_guess.to_numpy()
    else:
        mu = initial_guess.to_numpy() if isinstance(initial_guess, Mu) else initial_guess
        assert model.parameters.assert_compatible(model.parameters.parse(mu))

    if error_aware:
        assert error_criterion is not None
        assert callable(error_criterion)

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

    first_order_criticality = output_diff = mu_diff = update_norm = np.inf
    iteration = 0
    while True:
        if iteration >= miniter:
            if output_diff < rtol_output:
                logger.info(f'Relative tolerance of {rtol_output} for output difference reached. Converged.')
                break
            if mu_diff < rtol_mu:
                logger.info(f'Relative tolerance of {rtol_mu} for parameter difference reached. Converged.')
                break
            if first_order_criticality < tol_sub:
                logger.info(f'Absolute tolerance of {tol_sub} for first order criticality reached. Converged.')
                break
            if error_aware:
                if error_criterion(mu, current_output):
                    logger.info('Output error confidence reached. Converged.')
                    break
            if (iteration >= stagnation_window + 1
                    and stagnation_threshold * update_norm < min(update_norms[-stagnation_window - 1:])):
                logger.info(f'Norm of update is stagnating (threshold: {stagnation_threshold:5e}, '
                            f'window: {stagnation_window}). Converged.')
                break
            if iteration >= maxiter:
                logger.info(f'Maximum iterations reached. Failed to converge after {iteration} iterations.')
                raise BFGSError('Failed to converge after the maximum amount of iterations.')

        iteration += 1

        active, inactive = _get_active_and_inactive_sets(parameter_space, mu, eps)

        # compute update to mu
        if np.all(active):
            direction = -gradient
        else:
            direction = _compute_hessian_action(hessian, -gradient, active, inactive)

        step_size, line_search_iteration = armijo(
            output, mu, direction, grad=gradient, initial_value=current_output,
            additional_criterion=error_criterion if error_aware else None, **(line_search_params or {}))
        line_search_iterations.append(line_search_iteration)

        # update mu
        old_mu = mu.copy()
        old_output = current_output.copy()
        mu += step_size * direction
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
        first_order_criticality = np.linalg.norm(mu - parameter_space.clip(mu - gradient).to_numpy())
        foc_norms.append(first_order_criticality)

        # set new active inactive threshhold
        eps = np.linalg.norm(gradient)

        # update relative errors
        output_diff = abs(old_output - current_output) / abs(old_output)
        mu_diff = np.linalg.norm(mu - old_mu) / np.linalg.norm(old_mu)

        # update the hessian approximate
        hessian = _update_hessian(hessian, mu, old_mu, gradient, old_gradient)

        with warnings.catch_warnings():
            # ignore division-by-zero warnings when solution_norm or output is zero
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            logger.info(f'it:{iteration} '
                        f'foc:{first_order_criticality:.3e} '
                        f'upd:{update_norm:.3e} '
                        f'rel_upd:{update_norm / mu_norm:.3e} ')

        if not np.isfinite(update_norm) or not np.isfinite(mu_norm):
            raise BFGSError('Failed to converge.')

    data['update_norms'] = np.array(update_norms)
    data['foc_norms'] = np.array(foc_norms)
    data['iterations'] = iteration
    data['line_search_iterations'] = np.array(line_search_iterations)

    return mu, data


def _get_active_and_inactive_sets(parameter_space, mu, epsilon=1e-8):
    """Compute the active and inactive parameter index sets for constrained optimization.

    Parameters
    ----------
    parameter_space
        The |ParameterSpace| used for checking parameter ranges.
    mu
        The |parameter values| to compute the active and inactive sets for.
    epsilon
        Tolerance threshold for checking boundary conditions.

    Returns
    -------
    active
        The binary mask corresponding to the set of active parameter indices.
    inactive
        The binary mask corresponding to the set of inactive parameter indices.
    """
    mu = mu if isinstance(mu, Mu) else parameter_space.parameters.parse(mu)
    active = []

    for key, dim in parameter_space.parameters.items():
        indices = np.zeros(dim)
        eps = epsilon * np.ones(dim)
        low_range, high_range = parameter_space.ranges[key]
        indices = np.logical_or(mu[key] - low_range <= eps, high_range - mu[key] <= eps)
        active.append(indices)

    active = np.concatenate(active)
    inactive = ~active

    return active, inactive


def _compute_hessian_action(hessian, direction, active, inactive):
    """Compute the Hessian applied to the direction.

    The active set specifies which indices of the direction are actively constrained.

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

    Returns
    -------
    action
        The Hessian applied to the projected direction.
    """
    direction_A = active * direction
    direction_I = inactive * direction
    action = direction_A + inactive * (hessian @ direction_I)
    return action


def _update_hessian(hessian, mu, old_mu, gradient, old_gradient):
    """Update Hessian matrix using BFGS iteration.

    Parameters
    ----------
    hessian
        The current Hessian matrix.
    mu
        The current `mu` |parameter values| as a |NumPy array|.
    old_mu
        The previous `mu` |parameter values| as a |NumPy array|.
    gradient
        The current gradient with respect to the parameter.
    old_gradient
        The previous gradient with respect to the parameter.

    Returns
    -------
    hessian
        The next BFGS Hessian matrix iterate.
    """
    gradient_difference = gradient - old_gradient
    mu_difference = mu - old_mu
    gradient_mu_coefficient = np.inner(gradient_difference, mu_difference)

    if gradient_mu_coefficient > 0.:
        hessian_grad_diff = hessian @ gradient_difference
        quadratic_coefficient = np.inner(gradient_difference, hessian_grad_diff)

        hessian_part = np.outer(hessian_grad_diff, mu_difference) + np.outer(mu_difference, hessian_grad_diff)
        mu_part = np.outer(mu_difference, mu_difference)

        mu_coeff = (gradient_mu_coefficient + quadratic_coefficient) / gradient_mu_coefficient**2
        hessian_coeff = 1. / gradient_mu_coefficient

        hessian += mu_coeff * mu_part - hessian_coeff * hessian_part
    else:
        hessian = np.eye(hessian.shape[0])

    return hessian
