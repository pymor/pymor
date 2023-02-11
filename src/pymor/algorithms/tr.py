# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import warnings
from copy import deepcopy

import numpy as np

from pymor.algorithms.bfgs import bfgs
from pymor.core.base import BasicObject
from pymor.core.defaults import defaults
from pymor.core.exceptions import TRError
from pymor.core.logger import getLogger
from pymor.parameters.base import Mu


class TRSurrogate(BasicObject):
    """Surrogate for the :func:`trust_region`.

    Not intended to be used directly.
    """

    def __init__(self, reductor, initial_guess):
        self.__auto_init(locals())

        # generate a first rom if none was given
        self.fom = reductor.fom
        self.extend(initial_guess)
        self.accept()

        assert self.rom.output_functional is not None, 'Please provide an output functional with your model!'

        # initialize placeholders for extension
        self.new_reductor = None
        self.new_rom = None

    def fom_output(self, mu):
        return self.fom.output(mu)[0, 0]

    def fom_gradient(self, mu):
        return self.fom.parameters.parse(self.fom.output_d_mu(mu)).to_numpy()

    def rom_output(self, mu):
        return self.rom.output(mu)[0, 0]

    def estimate_output_error(self, mu):
        return self.rom.estimate_output_error(mu)

    def extend(self, mu):
        """Try to extend the current ROM for a new parameter.

        Parameters
        ----------
        mu
            The `Mu` instance for which an extension is computed.
        """
        with self.logger.block('Trying to extend the basis...'):
            U_h_mu = self.fom.solve(mu)
            self.new_reductor = deepcopy(self.reductor)
            try:
                self.new_reductor.extend_basis(U_h_mu)
                self.new_rom = self.new_reductor.reduce()
            except Exception:
                self.new_reductor = self.reductor
                self.new_rom = self.rom

    def new_rom_output(self, mu):
        assert self.new_rom is not None, 'No new ROM found. Did you forget to call surrogate.extend()?'
        return self.new_rom.output(mu)[0, 0]

    def accept(self):
        """Accept the new ROM.

        This function is intended to be called after :func:`extend` was called.
        """
        assert self.new_rom is not None, 'No new ROM found. Did you forget to call surrogate.extend()?'
        self.rom = self.new_rom
        self.reductor = self.new_reductor
        self.new_rom = None
        self.new_reductor = None

    def reject(self):
        """Reject the new ROM.

        This function is intended to be called after :func:`extend` was called.
        """
        self.new_rom = None
        self.new_reductor = None

    def rb_size(self):
        return len(self.reductor.bases['RB'])


@defaults('beta', 'radius', 'shrink_factor', 'miniter', 'maxiter', 'miniter_subproblem', 'maxiter_subproblem',
          'tol', 'radius_tol', 'rtol', 'tol_sub', 'stagnation_window', 'stagnation_threshold')
def trust_region(reductor, parameter_space=None, initial_guess=None, beta=.95, radius=.1,
                 shrink_factor=.5, miniter=0, maxiter=30, miniter_subproblem=0, maxiter_subproblem=400, tol=1e-6,
                 radius_tol=.75, rtol=1e-16, tol_sub=1e-8, line_search_params=None, stagnation_window=3,
                 stagnation_threshold=np.inf):
    """TR algorithm.

    This method solves the optimization problem ::

        min J(mu), mu in C

    for an output functional depending on a box-constrained `mu` using an
    adaptive trust region method.

    The main idea for the algorithm can be found in :cite:`YM13`, and an application to
    box-constrained parameters with possible enlarging of the trust radius in :cite:`K21`.

    Parameters
    ----------
    reductor
        The `reductor` used to generate the reduced order models and estimate the output error.
    parameter_space
        If not `None`, the |ParameterSpace| for enforcing the box constraints on the
        parameter `mu`. Otherwise a |ParameterSpace| with no constraints.
    initial_guess
        If not `None`, a |Mu| instance of length 1 containing an initial guess for
        the solution `mu`. Otherwise, a random parameter from the parameter space is chosen
        as the initial value.
    beta
        The factor to check if the current parameter is close to the trust region boundary.
    radius
        The radius of the initial trust region.
    shrink_factor
        The factor by which the trust region is shrunk. If the trust region radius is increased,
        it is increased by `1. / shrink_factor`.
    miniter
        Minimum amount of iterations to perform.
    maxiter
        Fail if the iteration count reaches this value without converging.
    miniter_subproblem
        Minimum amount of iterations to perform in the BFGS subproblem.
    maxiter_subproblem
        Fail the BFGS subproblem if the iteration count reaches this value without converging.
    tol
        Finish when the clipped parameter error measure is below this threshold.
    radius_tol
        Threshold for increasing the trust region radius upon extending the reduced order model.
    rtol
        Finish the subproblem when the relative error measure is below this threshold.
    tol_sub
        Finish when the subproblem clipped parameter error measure is below this threshold.
    line_search_params
        Dictionary of additional parameters passed to the line search method.
    stagnation_window
        Finish when the parameter update has been stagnating within a tolerance of
        `stagnation_threshold` during the last `stagnation_window` iterations.
    stagnation_threshold
        See `stagnation_window`.

    Returns
    -------
    mu
        |Numpy array| containing the computed parameter.
    data
        Dict containing the following fields:

            :mus:             `list` of parameters after each iteration.
            :mu_norms:        |NumPy array| of the solution norms after each iteration.
            :subproblem_data: `list` of data generated by the individual subproblems.

    Raises
    ------
    TRError
        Raised if the TR algorithm failed to converge.
    """
    assert shrink_factor != 0.

    logger = getLogger('pymor.algorithms.tr')

    if parameter_space is None:
        parameter_space = reductor.fom.parameters.space(-np.inf, np.inf)

    if initial_guess is None:
        initial_guess = parameter_space.sample_randomly(1)[0]
        mu = initial_guess.to_numpy()
    else:
        mu = initial_guess.to_numpy() if isinstance(initial_guess, Mu) else initial_guess

    def error_aware_line_search_criterion(new_mu, current_value):
        output_error = surrogate.estimate_output_error(new_mu)
        if output_error / abs(current_value) >= beta * radius:
            return True
        return False

    surrogate = TRSurrogate(reductor, initial_guess)

    data = {'subproblem_data': []}

    # compute norms
    mu_norm = np.linalg.norm(mu)
    update_norms = []
    foc_norms = []
    data['mus'] = [mu.copy()]

    old_rom_output = surrogate.rom_output(mu)
    old_fom_output = surrogate.fom_output(mu)

    first_order_criticity = 1e6
    iteration = 0
    while True:
        with logger.block(f'Starting adaptive TR algorithm iteration {iteration + 1} with radius {radius}...'):
            rejected = False

            if iteration >= miniter:
                if first_order_criticity < tol:
                    logger.info(
                        f'TR converged in {iteration} iterations because first order criticity tolerance of {tol}' \
                        f' was reached. The reduced basis is of size {surrogate.rb_size()}.')
                    break
                if iteration >= maxiter:
                    logger.info(f'Maximum iterations reached. Failed to converge after {iteration} iterations.')
                    raise TRError('Failed to converge.')

            iteration += 1

            # solve the subproblem using bfgs
            old_mu = mu.copy()

            with logger.block(f'Solving subproblem for mu {mu} with BFGS...'):
                mu, sub_data = bfgs(
                    surrogate.rom, parameter_space, initial_guess=mu, miniter=miniter_subproblem,
                    maxiter=maxiter_subproblem, rtol=rtol, tol_sub=tol_sub,
                    line_search_params=line_search_params, stagnation_window=stagnation_window,
                    stagnation_threshold=stagnation_threshold, error_aware=True,
                    error_criterion=error_aware_line_search_criterion, beta=beta, radius=radius)

            # first BFGS iterate is AGC point
            index = 1 if len(sub_data['mus']) > 1 else 0
            compare_output = surrogate.rom_output(sub_data['mus'][index])
            estimate_output = surrogate.estimate_output_error(mu)
            current_output = surrogate.rom_output(mu)

            with logger.block('Running output checks for TR parameters.'):
                if current_output + estimate_output < compare_output:
                    surrogate.extend(mu)
                    current_fom_output = surrogate.fom_output(mu)
                    fom_output_diff = old_fom_output - current_fom_output
                    rom_output_diff = old_rom_output - current_output
                    if fom_output_diff >= radius_tol * rom_output_diff:
                        # increase the radius if the model confidence is high enough
                        radius /= shrink_factor
                elif current_output - estimate_output > compare_output:
                    # reject new mu
                    rejected = True
                    # shrink the radius
                    radius *= shrink_factor
                else:
                    surrogate.extend(mu)
                    current_output = surrogate.new_rom_output(mu)
                    if current_output <= compare_output:
                        current_fom_output = surrogate.fom_output(mu)
                        fom_output_diff = old_fom_output - current_fom_output
                        rom_output_diff = old_rom_output - current_output
                        if fom_output_diff >= radius_tol * rom_output_diff:
                            # increase the radius if the model confidence is high enough
                            radius /= shrink_factor
                    else:
                        # reject new mu
                        rejected = True
                        # shrink the radius
                        radius *= shrink_factor

            # handle parameter rejection
            if not rejected:
                data['mus'].append(mu.copy())
                mu_norm = np.linalg.norm(mu)
                update_norms.append(np.linalg.norm(mu - old_mu))

                data['subproblem_data'].append(sub_data)

                with logger.block('Computing first order criticity...'):
                    gradient = surrogate.fom_gradient(mu)
                    first_order_criticity = np.linalg.norm(mu - parameter_space.clip(mu - gradient).to_numpy())
                    foc_norms.append(first_order_criticity)

                surrogate.accept()
                old_rom_output = current_output
            else:
                mu = old_mu
                surrogate.reject()

            with warnings.catch_warnings():
                # ignore division-by-zero warnings when solution_norm or output is zero
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                logger.info(f'it:{iteration} '
                            f'foc:{first_order_criticity:.3e} '
                            f'radius:{radius:.3e}')

            if not np.isfinite(mu_norm):
                raise TRError('Failed to converge.')

    logger.info('')

    data['update_norms'] = np.array(update_norms)
    data['foc_norms'] = np.array(foc_norms)
    data['iterations'] = iteration

    return mu, data
