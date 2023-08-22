# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from copy import deepcopy

import numpy as np

from pymor.algorithms.bfgs import error_aware_bfgs
from pymor.core.base import BasicObject, abstractmethod
from pymor.core.defaults import defaults
from pymor.core.exceptions import ExtensionError, TRError
from pymor.core.logger import getLogger
from pymor.models.basic import StationaryModel
from pymor.parameters.base import Mu


@defaults('beta', 'radius', 'shrink_factor', 'miniter', 'maxiter', 'miniter_subproblem', 'maxiter_subproblem',
          'tol_criticality', 'radius_tol', 'rtol_output', 'rtol_mu', 'tol_sub', 'stagnation_window',
          'stagnation_threshold')
def trust_region(fom, surrogate, parameter_space=None, initial_guess=None, beta=.95, radius=.1,
                 shrink_factor=.5, miniter=0, maxiter=30, miniter_subproblem=0, maxiter_subproblem=400,
                 tol_criticality=1e-6, radius_tol=.75, rtol_output=1e-16, rtol_mu=1e-16, tol_sub=1e-8,
                 armijo_alpha=1e-4, line_search_params=None, stagnation_window=3, stagnation_threshold=np.inf):
    """Error-aware trust region algorithm.

    This method solves the optimization problem ::

        min J(mu), mu in C

    for a model with an output :math:`J` depending on a box-constrained `mu` using
    an adaptive trust region method.

    The main idea for the algorithm can be found in :cite:`YM13`, and an application to
    box-constrained parameters with possible enlarging of the trust radius in :cite:`KMOSV21`.

    This method contrasts itself from :func:`scipy.optimize.minimize` in the computation of the
    trust region: `scipy` TR implementations use a metric distance, whereas this function uses an
    error estimator obtained from the surrogate. Additionally, the cheap model function
    surrogate here is only updated for each outer iteration, not entirely reconstructed.

    Parameters
    ----------
    fom
        The |Model| with output `J` used for the optimization.
    surrogate
        The :class:`TRSurrogate` used to generate the surrogate model and estimate the output error.
    parameter_space
        If not `None`, the |ParameterSpace| for enforcing the box constraints on the
        |parameter values| `mu`. Otherwise a |ParameterSpace| with lower bound -1
        and upper bound 1 is used.
    initial_guess
        If not `None`, |parameter values| containing an initial guess for the
        solution `mu`. Otherwise, random |parameter values| from the `parameter_space` are
        chosen as the initial value.
    beta
        The factor used to check if the current |parameter values| are close to the
        trust region boundary.
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
    tol_criticality
        Finish when the current |parameter values| fulfill the approximate first order critical
        optimality condition with a value below this threshold.
    radius_tol
        Threshold for increasing the trust region radius upon extending the reduced order model.
    rtol_output
        See `rtol_output` of :func:`~pymor.algorithms.bfgs.error_aware_bfgs`.
    rtol_mu
        See `rtol_mu` of :func:`~pymor.algorithms.bfgs.error_aware_bfgs`.
    tol_sub
        See `tol_sub` of :func:`~pymor.algorithms.bfgs.error_aware_bfgs`.
    armijo_alpha
        Threshold for the constrained Armijo condition.
        See :func:`~pymor.algorithms.line_search.constrained_armijo`.
    line_search_params
        Dictionary of additional parameters passed to the Armijo line search method.
    stagnation_window
        Finish when the parameter update has not been enlarged by a factor of
        `stagnation_threshold` during the last `stagnation_window` iterations.
    stagnation_threshold
        See `stagnation_window`.

    Returns
    -------
    mu
        |Numpy array| containing the computed optimal |parameter values|.
    data
        Dict containing additional information of the optimization call.

    Raises
    ------
    TRError
        Raised if the TR algorithm failed to converge.
    """
    assert shrink_factor > 0.
    assert fom.dim_output == 1

    logger = getLogger('pymor.algorithms.tr.trust_region')
    logger.info(f'Started error-aware adaptive TR algorithm for {fom.output_functional}.')

    if parameter_space is None:
        logger.warn('No parameter space given. Assuming uniform parameter bounds of (-1, 1).')
        parameter_space = fom.parameters.space(-1., 1.)

    if initial_guess is None:
        initial_guess = parameter_space.sample_randomly(1)[0]
        mu = initial_guess.to_numpy()
    else:
        mu = initial_guess.to_numpy() if isinstance(initial_guess, Mu) else initial_guess

    def error_aware_bfgs_criterion(new_mu, current_value):
        output_error = surrogate.estimate_output_error(new_mu)
        return output_error / abs(current_value) >= beta * radius

    def error_aware_line_search_criterion(starting_point, initial_value, current_value, step,
                                          line_search_beta, direction, slope):
        initial_mu = starting_point
        new_mu = initial_mu + step * direction

        # check if the new parameter is outside of the parameter space's bounds
        if not np.allclose(new_mu - parameter_space.clip(new_mu).to_numpy(), 0.):
            return False

        # check the convergence conditions of the line search
        # the first condition corresponds to the usual armijo descent criterion
        # the second condition checks if the new relative error is still within the trust region
        output_error = surrogate.estimate_output_error(new_mu)
        return (current_value <= initial_value - (armijo_alpha / step) * np.linalg.norm(new_mu - initial_mu)**2
            and output_error / abs(current_value) <= radius)

    data = {'subproblem_data': []}

    # compute norms
    mu_norm = np.linalg.norm(mu)
    update_norms = []
    foc_norms = []
    data['mus'] = [mu.copy()]

    old_rom_output = surrogate.output(mu)
    old_fom_output = fom.output(mu)

    first_order_criticality = np.inf
    iteration = 0
    while True:
        with logger.block(f'Starting adaptive TR algorithm iteration {iteration + 1} with radius {radius}...'):
            rejected = False

            if iteration >= miniter:
                if first_order_criticality < tol_criticality:
                    logger.info(
                        f'TR converged in {iteration} iterations because first order criticality tolerance of' \
                        f' {tol_criticality} was reached. The reduced basis is of size {surrogate.rb_size()}.')
                    break
                if iteration >= maxiter:
                    logger.info(f'Maximum iterations reached. Failed to converge after {iteration} iterations.')
                    raise TRError('Failed to converge after the maximum amount of iterations.')

            iteration += 1

            # solve the subproblem using bfgs
            old_mu = mu.copy()

            with logger.block(f'Solving subproblem for mu {mu} with BFGS...'):
                mu, sub_data = error_aware_bfgs(
                    surrogate, parameter_space, initial_guess=mu, miniter=miniter_subproblem,
                    maxiter=maxiter_subproblem, rtol_output=rtol_output, rtol_mu=rtol_mu, tol_sub=tol_sub,
                    line_search_params=line_search_params, stagnation_window=stagnation_window,
                    stagnation_threshold=stagnation_threshold, error_aware=True,
                    error_criterion=error_aware_bfgs_criterion,
                    line_search_error_criterion=error_aware_line_search_criterion)

            # first BFGS iterate is AGC point
            index = 1 if len(sub_data['mus']) > 1 else 0
            compare_output = surrogate.output(sub_data['mus'][index])
            estimate_output = surrogate.estimate_output_error(mu)
            current_output = surrogate.output(mu)

            with logger.block('Running output checks for TR parameters.'):
                if current_output + estimate_output < compare_output:
                    surrogate.extend(mu)
                    # fom.output is potentially for free after enrichment, e.g. using caching
                    current_fom_output = fom.output(mu)
                    fom_output_diff = old_fom_output - current_fom_output
                    rom_output_diff = old_rom_output - current_output
                    if fom_output_diff >= radius_tol * rom_output_diff:
                        # increase the radius if the model confidence is high enough
                        # model confidence is deemed high if a small decrease in the ROM's output
                        # results in a larger decrease of the FOM's output
                        radius /= shrink_factor

                    msg = 'Estimated output smaller than previous output.'
                elif current_output - estimate_output > compare_output:
                    # reject new mu
                    rejected = True
                    # shrink the radius
                    radius *= shrink_factor

                    msg = 'Estimated output larger than previous output.'
                else:
                    surrogate.extend(mu)
                    current_output = surrogate.new_output(mu)
                    if current_output <= compare_output:
                        # fom.output is potentially for free after enrichment, e.g. using caching
                        current_fom_output = fom.output(mu)
                        fom_output_diff = old_fom_output - current_fom_output
                        rom_output_diff = old_rom_output - current_output
                        if fom_output_diff >= radius_tol * rom_output_diff:
                            # increase the radius if the model confidence is high enough
                            radius /= shrink_factor

                        msg = 'Updated model output smaller than previous output.'
                    else:
                        # reject new mu
                        rejected = True
                        # shrink the radius
                        radius *= shrink_factor

                        msg = 'Updated model output larger than previous output.'

            # handle parameter rejection
            if not rejected:
                data['mus'].append(mu.copy())
                mu_norm = np.linalg.norm(mu)
                update_norms.append(np.linalg.norm(mu - old_mu))

                data['subproblem_data'].append(sub_data)

                with logger.block('Computing first order criticality...'):
                    # fom.output_d_mu is potentially for free after enrichment, e.g. using caching
                    gradient = fom.parameters.parse(fom.output_d_mu(mu)).to_numpy()
                    first_order_criticality = np.linalg.norm(mu - parameter_space.clip(mu - gradient).to_numpy())
                    foc_norms.append(first_order_criticality)

                surrogate.accept()
                logger.info(f'Current mu iterate accepted: {msg}')

                old_rom_output = current_output
            else:
                mu = old_mu
                surrogate.reject()
                logger.info(f'Current mu iterate rejected: {msg}')

            if not np.isfinite(mu_norm):
                raise TRError('Failed to converge.')

    logger.info('')

    data['update_norms'] = np.array(update_norms)
    data['foc_norms'] = np.array(foc_norms)
    data['iterations'] = iteration
    data['rom_output_evaluations'] = surrogate.rom_output_evaluations
    data['rom_output_d_mu_evaluations'] = surrogate.rom_output_d_mu_evaluations
    data['rom_output_estimations'] = surrogate.rom_output_estimations
    data['enrichments'] = surrogate.enrichments
    data['rom'] = surrogate.rom

    return mu, data

def coercive_rb_trust_region(reductor, primal_dual=False, parameter_space=None, initial_guess=None,
                             beta=.95, radius=.1, shrink_factor=.5, miniter=0, maxiter=30, miniter_subproblem=0,
                             maxiter_subproblem=400, tol_criticality=1e-6, radius_tol=.75, rtol_output=1e-16,
                             rtol_mu=1e-16, tol_sub=1e-8, armijo_alpha=1e-4, line_search_params=None,
                             stagnation_window=3, stagnation_threshold=np.inf,
                             quadratic_output=False, quadratic_output_continuity_estimator=None,
                             quadratic_output_product_name=None):
    """Error aware trust-region method for a coercive RB model as surrogate.

    See :func:`trust_region`.

    Parameters
    ----------
    reductor
        The reductor used to generate the reduced order models and estimate the output error.
    primal_dual
        If `False`, only enrich with the primal solution. If `True`, additionally
        enrich with the dual solutions.
    quadratic_output
        Set to `True` if output is given by a quadratic functional.
    quadratic_output_continuity_estimator
        In case of a quadratic output functional, a |ParameterFunctional| giving an upper bound for
        the norm of the corresponding bilinear form.
    quadratic_output_product_name
        In case of a quadratic output functional, the name of the inner-product |Operator|
        of the ROM w.r.t. which the continuity constant for the output is estimated.
    """
    if not isinstance(reductor.fom, StationaryModel):
        raise NotImplementedError
    if primal_dual and quadratic_output:
        raise NotImplementedError
    if primal_dual:
        surrogate = PrimalDualTRSurrogate(reductor, initial_guess)
    elif quadratic_output:
        # special case for a quadratic output functional as we do not have a reductor yet which
        # assembles an appropriate error estimator
        surrogate = QuadraticOutputTRSurrogate(reductor, initial_guess,
                                               continuity_estimator_output=quadratic_output_continuity_estimator,
                                               product_name=quadratic_output_product_name)
    else:
        surrogate = BasicTRSurrogate(reductor, initial_guess)

    mu, data = trust_region(reductor.fom, surrogate, parameter_space=parameter_space, initial_guess=initial_guess,
                            beta=beta, radius=radius, shrink_factor=shrink_factor, miniter=miniter,
                            maxiter=maxiter, miniter_subproblem=miniter_subproblem,
                            maxiter_subproblem=maxiter_subproblem, tol_criticality=tol_criticality,
                            radius_tol=radius_tol, rtol_output=rtol_output, rtol_mu=rtol_mu, tol_sub=tol_sub,
                            armijo_alpha=armijo_alpha, line_search_params=line_search_params,
                            stagnation_window=stagnation_window, stagnation_threshold=stagnation_threshold)

    # factor 2 assumes adjoint approach, which is the default for the available coercive RB models.
    data['rom_evaluations'] = surrogate.rom_output_evaluations + 2 * surrogate.rom_output_d_mu_evaluations
    # every estimation included one rom evaluation for all available TRSurrogates
    data['rom_evaluations'] += surrogate.rom_output_estimations

    # Note: this evaluation count assumes caching of the primal solution, see above.
    # for every gradient, one also requires the dual solution (with the adjoint approach)
    # which is only available if primal_dual is `True`.
    fom_evaluations_outer = 0 if primal_dual else data['iterations']
    data['fom_evaluations'] = surrogate.fom_evaluations + fom_evaluations_outer

    return mu, data

class TRSurrogate(BasicObject):
    """Base class for :func:`trust_region` surrogates.

    Not to be used directly.
    """

    def __init__(self, reductor, initial_guess, name=None):
        self.__auto_init(locals())
        self.parameters = reductor.fom.parameters
        self.dim_output = reductor.fom.dim_output

        self.fom_evaluations = 0
        self.rom_output_evaluations = 0
        self.rom_output_d_mu_evaluations = 0
        self.rom_output_estimations = 0
        self.enrichments = 0

        # generate a first rom based on the initial guess
        if isinstance(initial_guess, Mu):
            initial_guess = initial_guess.to_numpy()
        assert isinstance(initial_guess, np.ndarray)
        self.extend(initial_guess)
        self.accept()

        # so far, we only support 1-dimensional outputs
        assert self.rom.dim_output == 1

        # initialize placeholders for extension
        self.new_reductor = None
        self.new_rom = None

    def output(self, mu):
        self.rom_output_evaluations += 1
        return self.rom.output(mu)

    def output_d_mu(self, mu):
        self.rom_output_d_mu_evaluations += 1
        return self.rom.output_d_mu(mu)

    def estimate_output_error(self, mu):
        self.rom_output_estimations += 1
        return self.rom.estimate_output_error(mu)

    @abstractmethod
    def extend(self, mu):
        pass

    def new_output(self, mu):
        assert self.new_rom is not None, 'No new ROM found. Did you forget to call surrogate.extend()?'
        assert self.new_rom.dim_output == 1
        self.rom_output_evaluations += 1
        return self.new_rom.output(mu)[0, 0]

    def accept(self):
        """Accept the new ROM.

        This function is intended to be called after :meth:`~TRSurrogate.extend` was called.
        """
        assert self.new_rom is not None, 'No new ROM found. Did you forget to call surrogate.extend()?'
        self.rom = self.new_rom
        self.reductor = self.new_reductor
        self.new_rom = None
        self.new_reductor = None
        self.enrichments += 1

    def reject(self):
        """Reject the new ROM.

        This function is intended to be called after :func:`extend` was called.
        """
        self.new_rom = None
        self.new_reductor = None

    def rb_size(self):
        return len(self.reductor.bases['RB'])


class BasicTRSurrogate(TRSurrogate):
    """Surrogate for :func:`trust_region` only enriching with the primal solution.

    Parameters
    ----------
    reductor
        The reductor used to generate the reduced order models and estimate the output error.
    initial_guess
        The |parameter values| containing an initial guess for the optimal parameter value.
    """

    def extend(self, mu):
        """Extend the current ROM for new |parameter values|.

        Parameters
        ----------
        mu
            The `Mu` instance for which an extension is computed.
        """
        with self.logger.block('Extending the basis...'):
            U_h_mu = self.reductor.fom.solve(mu)
            self.fom_evaluations += 1
            self.new_reductor = deepcopy(self.reductor)
            try:
                self.new_reductor.extend_basis(U_h_mu)
            except ExtensionError:
                self.new_reductor = self.reductor
            self.new_rom = self.new_reductor.reduce()


class PrimalDualTRSurrogate(TRSurrogate):
    """Surrogate for :func:`trust_region` enriching with both the primal and dual solutions.

    Parameters
    ----------
    reductor
        The reductor used to generate the reduced order models and estimate the output error.
    initial_guess
        The |parameter values| containing an initial guess for the optimal parameter value.
    """

    def __init__(self, reductor, initial_guess, name=None):
        if not isinstance(reductor.fom, StationaryModel):
            raise NotImplementedError
        super().__init__(reductor, initial_guess, name)

    def extend(self, mu):
        """Extend the current ROM for new |parameter values| with primal and dual solutions.

        Parameters
        ----------
        mu
            The `Mu` instance for which an extension is computed.
        """
        with self.logger.block('Extending the basis with primal and dual...'):
            fom = self.reductor.fom
            U_h_mu = fom.solve(mu)
            jacobian = fom.output_functional.jacobian(U_h_mu, self.reductor.fom.parameters.parse(mu))
            dual_solutions = fom.solution_space.empty()
            for d in range(fom.dim_output):
                dual_problem = fom.with_(operator=fom.operator.H, rhs=jacobian.H.as_range_array(mu)[d])
                P_h_mu = dual_problem.solve(mu)
                dual_solutions.append(P_h_mu)

            self.fom_evaluations += 1 + fom.dim_output
            self.new_reductor = deepcopy(self.reductor)
            try:
                self.new_reductor.extend_basis(U_h_mu)
            except ExtensionError:
                self.new_reductor = self.reductor
            # it can happen that primal is successful but duals are not.
            try:
                self.new_reductor.extend_basis(dual_solutions)
            except ExtensionError:
                pass
            self.new_rom = self.new_reductor.reduce()


class QuadraticOutputTRSurrogate(BasicTRSurrogate):
    """Surrogate for :func:`trust_region` with only the primal enrichment naive output estimate.

    .. note::
        This specialized TRSurrogate should soon be made obsolete when the quadratic
        output estimation is included into a reductor.

    Parameters
    ----------
    reductor
        The reductor used to generate the reduced order models and estimate the output error.
    initial_guess
        The |parameter values| containing an initial guess for the optimal parameter value.
    continuity_estimator_output
        Estimation for the continuity constant of the quadratic output functional.
    product_name
        Name of the inner-product |Operator| of the ROM w.r.t. which the continuity
        constant for the output is estimated.
    """

    def __init__(self, reductor, initial_guess, continuity_estimator_output, product_name=None):
        self.__auto_init(locals())
        super().__init__(reductor, initial_guess)

    def estimate_output_error(self, mu):
        self.rom_output_estimations += 1
        U, pr_err = self.rom.solve(mu, return_error_estimate=True)
        cont_est = self.continuity_estimator_output
        cont = cont_est.evaluate(self.parameters.parse(mu)) if hasattr(cont_est, 'evaluate') else cont_est
        U_norm = U.norm(self.rom.products[self.product_name] if self.product_name else None)
        return cont * (pr_err * (2 * U_norm + pr_err))
