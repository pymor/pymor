# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number
import time

import numpy as np

from pymor.core.logger import getLogger
from pymor.discretizations.basic import StationaryDiscretization
from pymor.parallel.dummy import dummy_pool
from pymor.reductors.basic import reduce_to_subbasis


def reduction_error_analysis(reduced_discretization, discretization=None, reconstructor=None,
                             test_mus=10, basis_sizes=0, random_seed=None,
                             estimator=True, condition=False, error_norms=tuple(), error_norm_names=None,
                             estimator_norm_index=0, custom=tuple(),
                             plot=False, plot_custom_logarithmic=True,
                             pool=dummy_pool):
    """Analyze the model reduction error.

    The maximum model reduction error is estimated by solving the reduced
    |Discretization| for given random |Parameters|.

    Parameters
    ----------
    reduced_discretization
        The reduced |Discretization|.
    discretization
        The high-dimensional |Discretization|. Must be specified if
        `error_norms` are given.
    reconstructor
        The reconstructor for `reduced_discretization`. Must be specified
        if `error_norms` are given.
    test_mus
        Either a list of |Parameters| to compute the errors for, or
        the number of parameters which are sampled randomly from
        `parameter_space` (if given) or `reduced_discretization.parameter_space`.
    basis_sizes
        Either a list of reduced basis dimensions to consider, or
        the number of dimensions (which are then selected equidistantly,
        always including the maximum reduced space dimension).
        The dimensions are input for `~pymor.reductors.basic.reduce_to_subbasis`
        to quickly obtain smaller reduced |Discretizations| from
        `rb_discretization`.
    random_seed
        If `test_mus` is a number, use this value as random seed
        for drawing the |Parameters|.
    estimator
        If `True` evalute the error estimator of `reduced_discretization`
        on the test |Parameters|.
    condition
        If `True`, compute the condition of the reduced system matrix
        for the given test |Parameters|. (Can only be specified if
        `rb_discretization` is an instance of |StationaryDiscretization|
        and `rb_discretization.operator` is linear.
    error_norms
        List of norms in which to compute the model reduction error.
    error_norm_names
        Names of the norms given by `error_norms`. If `None`, the
        `name` attributes of the given norms are used.
    estimator_norm_index
        When `estimator` is `True` and `error_norms` are specified,
        this is the index of the norm in `error_norms` w.r.t. which
        to compute the effectivity of the estimator.
    custom
        List of custom functions which are evaluated for each test |Parameter|
        and basis size. The function must have the signature ::

            def custom_value(reduced_discretization, discretization=d,
                             reconstructor, mu, dim):
                pass

    plot
        If `True`, generate a plot of the computed quantities w.r.t.
        the basis size.
    plot_custom_logarithmic
        If `True`, use a logarithmic y-axis to plot the computed custom
        values.
    pool
        If not `None`, the |WorkerPool| to use for parallelization.

    Returns
    -------
    Dict with the following fields:

        :mus:                    The test |Parameters| which have been considered.
        :basis_sizes:            The reduced basis dimensions which have been considered.
        :norms:                  |Array| of the norms of the high-dimensional solutions
                                 w.r.t. all given test |Parameters|, reduced basis
                                 dimensions and norms in `error_norms`.
                                 (Only present when `error_norms` has been specified.)
        :max_norms:              Maxima of `norms` over the given test |Parameters|.
        :max_norm_mus:           |Parameters| corresponding to `max_norms`.
        :errors:                 |Array| of the norms of the model reduction errors
                                 w.r.t. all given test |Parameters|, reduced basis
                                 dimensions and norms in `error_norms`.
                                 (Only present when `error_norms` has been specified.)
        :max_errors:             Maxima of `errors` over the given test |Parameters|.
        :max_error_mus:          |Parameters| corresponding to `max_errors`.
        :rel_errors:             `errors` divided by `norms`.
                                 (Only present when `error_norms` has been specified.)
        :max_rel_errors:         Maxima of `rel_errors` over the given test |Parameters|.
        :max_rel_error_mus:      |Parameters| corresponding to `max_rel_errors`.
        :error_norm_names:       Names of the the given `error_norms`.
                                 (Only present when `error_norms` has been specified.)
        :estimates:              |Array| of the model reduction error estimates
                                 w.r.t. all given test |Parameters| and reduced basis
                                 dimensions.
                                 (Only present when `estimator` is `True`.)
        :max_estimate:           Maxima of `estimates` over the given test |Parameters|.
        :max_estimate_mus:       |Parameters| corresponding to `max_estimates`.
        :effectivities:          `errors` divided by `estimates`.
                                 (Only present when `estimator` is `True` and `error_norms`
                                 has been specified.)
        :min_effectivities:      Minima of `effectivities` over the given test |Parameters|.
        :min_effectivity_mus:    |Parameters| corresponding to `min_effectivities`.
        :max_effectivities:      Maxima of `effectivities` over the given test |Parameters|.
        :max_effectivity_mus:    |Parameters| corresponding to `max_effectivities`.
        :errors:                 |Array| of the reduced system matrix conditions
                                 w.r.t. all given test |Parameters| and reduced basis
                                 dimensions.
                                 (Only present when `conditions` is `True`.)
        :max_conditions:         Maxima of `conditions` over the given test |Parameters|.
        :max_condition_mus:      |Parameters| corresponding to `max_conditions`.
        :custom_values:          |Array| of custom function evaluations
                                 w.r.t. all given test |Parameters|, reduced basis
                                 dimensions and functions in `custom`.
                                 (Only present when `custom` has been specified.)
        :max_custom_values:      Maxima of `custom_values` over the given test |Parameters|.
        :max_custom_values_mus:  |Parameters| corresponding to `max_custom_values`.
        :time:                   Time (in seconds) needed for the error analysis.
        :summary:                String containing a summary of all computed quantities for
                                 the largest (last) considered basis size.
        :figure:                 The figure containing the generated plots.
                                 (Only present when `plot` is `True`.)
    """

    assert not error_norms or (discretization and reconstructor)
    assert error_norm_names is None or len(error_norm_names) == len(error_norms)
    assert not condition \
        or isinstance(reduced_discretization, StationaryDiscretization) and reduced_discretization.operator.linear

    logger = getLogger('pymor.algorithms.error')
    if pool is None or pool is dummy_pool:
        pool = dummy_pool
    else:
        logger.info('Using pool of {} workers for error analysis'.format(len(pool)))

    tic = time.time()

    d, rd, rc = discretization, reduced_discretization, reconstructor

    if isinstance(test_mus, Number):
        test_mus = reduced_discretization.parameter_space.sample_randomly(test_mus, seed=random_seed)
    if isinstance(basis_sizes, Number):
        if basis_sizes == 1:
            basis_sizes = [rd.solution_space.dim]
        else:
            if basis_sizes == 0:
                basis_sizes = rd.solution_space.dim + 1
            basis_sizes = min(rd.solution_space.dim + 1, basis_sizes)
            basis_sizes = np.linspace(0, rd.solution_space.dim, basis_sizes).astype(int)
    if error_norm_names is None:
        error_norm_names = tuple(norm.name for norm in error_norms)

    norms, estimates, errors, conditions, custom_values = \
        list(zip(*pool.map(_compute_errors, test_mus, d=d, rd=rd, rc=rc, estimator=estimator,
                      error_norms=error_norms, condition=condition, custom=custom, basis_sizes=basis_sizes)))
    print()

    result = {}
    result['mus'] = test_mus = np.array(test_mus)
    result['basis_sizes'] = basis_sizes

    summary = []
    summary.append(('number of samples', str(len(test_mus))))

    if error_norms:
        result['norms'] = norms = np.array(norms)
        result['max_norms'] = max_norms = np.max(norms, axis=0)
        result['max_norm_mus'] = max_norm_mus = test_mus[np.argmax(norms, axis=0)]
        result['errors'] = errors = np.array(errors)
        result['max_errors'] = max_errors = np.max(errors, axis=0)
        result['max_error_mus'] = max_error_mus = test_mus[np.argmax(errors, axis=0)]
        result['rel_errors'] = rel_errors = errors / norms[:, :, np.newaxis]
        result['max_rel_errors'] = np.max(rel_errors, axis=0)
        result['max_rel_error_mus'] = test_mus[np.argmax(rel_errors, axis=0)]
        for name, norm, norm_mu, error, error_mu in zip(error_norm_names,
                                                        max_norms, max_norm_mus,
                                                        max_errors[:, -1], max_error_mus[:, -1]):
            summary.append(('maximum {}-norm'.format(name),
                            '{:.7e} (mu = {})'.format(norm, error_mu)))
            summary.append(('maximum {}-error'.format(name),
                            '{:.7e} (mu = {})'.format(error, error_mu)))
        result['error_norm_names'] = error_norm_names

    if estimator:
        result['estimates'] = estimates = np.array(estimates)
        result['max_estimates'] = max_estimates = np.max(estimates, axis=0)
        result['max_estimate_mus'] = max_estimate_mus = test_mus[np.argmax(estimates, axis=0)]
        summary.append(('maximum estimated error',
                        '{:.7e} (mu = {})'.format(max_estimates[-1], max_estimate_mus[-1])))

    if estimator and error_norms:
        result['effectivities'] = effectivities = errors[:, estimator_norm_index, :] / estimates
        result['max_effectivities'] = max_effectivities = np.max(effectivities, axis=0)
        result['max_effectivity_mus'] = max_effectivity_mus = test_mus[np.argmax(effectivities, axis=0)]
        result['min_effectivities'] = min_effectivities = np.min(effectivities, axis=0)
        result['min_effectivity_mus'] = min_effectivity_mus = test_mus[np.argmin(effectivities, axis=0)]
        summary.append(('minimum estimator effectivity',
                        '{:.7e} (mu = {})'.format(min_effectivities[-1], min_effectivity_mus[-1])))
        summary.append(('maximum estimator effectivity',
                        '{:.7e} (mu = {})'.format(max_effectivities[-1], max_effectivity_mus[-1])))

    if condition:
        result['conditions'] = conditions = np.array(conditions)
        result['max_conditions'] = max_conditions = np.max(conditions, axis=0)
        result['max_condition_mus'] = max_condition_mus = test_mus[np.argmax(conditions, axis=0)]
        summary.append(('maximum system matrix condition',
                        '{:.7e} (mu = {})'.format(max_conditions[-1], max_condition_mus[-1])))

    if custom:
        result['custom_values'] = custom_values = np.array(custom_values)
        result['max_custom_values'] = max_custom_values = np.max(custom_values, axis=0)
        result['max_custom_values_mus'] = max_custom_values_mus = test_mus[np.argmax(custom_values, axis=0)]
        for i, (value, mu) in enumerate(zip(max_custom_values[:, -1], max_custom_values_mus[:, -1])):
            summary.append(('maximum custom value {}'.format(i),
                            '{:.7e} (mu = {})'.format(value, mu)))

    toc = time.time()
    result['time'] = toc - tic
    summary.append(('elapsed time', str(toc - tic)))

    summary_fields, summary_values = list(zip(*summary))
    summary_field_width = np.max(list(map(len, summary_fields))) + 2
    summary_lines = ['    {:{}} {}'.format(field + ':', summary_field_width, value)
                     for field, value in zip(summary_fields, summary_values)]
    summary = 'Stochastic error estimation:\n' + '\n'.join(summary_lines)
    result['summary'] = summary

    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        num_plots = (int(bool(error_norms) or estimator) + int(bool(error_norms) and estimator) +
                     int(condition) + int(bool(custom)))
        current_plot = 1

        if bool(error_norms) or estimator:
            ax = fig.add_subplot(1, num_plots, current_plot)
            legend = []
            if error_norms:
                for name, errors in zip(error_norm_names, max_errors):
                    ax.semilogy(basis_sizes, errors)
                    legend.append(name)
            if estimator:
                ax.semilogy(basis_sizes, max_estimates)
                legend.append('estimator')
            ax.legend(legend)
            ax.set_title('maximum errors')
            current_plot += 1

        if bool(error_norms) and estimator:
            ax = fig.add_subplot(1, num_plots, current_plot)
            ax.semilogy(basis_sizes, min_effectivities)
            ax.semilogy(basis_sizes, max_effectivities)
            ax.legend(('min', 'max'))
            ax.set_title('estimator effectivities')
            current_plot += 1

        if condition:
            ax = fig.add_subplot(1, num_plots, current_plot)
            ax.semilogy(basis_sizes, max_conditions)
            ax.set_title('maximum condition')
            current_plot += 1

        if custom:
            ax = fig.add_subplot(1, num_plots, current_plot)
            legend = []
            for i, values in enumerate(custom_values):
                if plot_custom_logarithmic:
                    ax.semilogy(basis_sizes, values)
                else:
                    ax.plot(basis_sizes, values)
                legend.append('value ' + str(i))
            ax.legend(legend)
            ax.set_title('maximum custom values')
            current_plot += 1

        result['figure'] = fig

    return result


def _compute_errors(mu, d, rd, rc, estimator, error_norms, condition, custom, basis_sizes):
    import numpy as np
    import sys

    print('.', end='')
    sys.stdout.flush()

    estimates = np.empty(len(basis_sizes)) if estimator else None
    norms = np.empty(len(error_norms))
    errors = np.empty((len(error_norms), len(basis_sizes)))
    conditions = np.empty(len(basis_sizes)) if condition else None
    custom_values = np.empty((len(custom), len(basis_sizes)))

    if d:
        logging_disabled = d.logging_disabled
        d.disable_logging()
        U = d.solve(mu)
        d.disable_logging(logging_disabled)
        for i_norm, norm in enumerate(error_norms):
            n = norm(U)
            n = n[0] if hasattr(n, '__len__') else n
            norms[i_norm] = n

    for i_N, N in enumerate(basis_sizes):
        rrd, rrc = reduce_to_subbasis(rd, N, reconstructor=rc)[:2]
        u = rrd.solve(mu)
        if estimator:
            e = rrd.estimate(u, mu)
            e = e[0] if hasattr(e, '__len__') else e
            estimates[i_N] = e
        if d and rc:
            URB = rrc.reconstruct(u)
            for i_norm, norm in enumerate(error_norms):
                e = norm(U - URB)
                e = e[0] if hasattr(e, '__len__') else e
                errors[i_norm, i_N] = e
        if condition:
            conditions[i_N] = np.linalg.cond(rrd.operator.assemble(mu)._matrix) if N > 0 else 0.
        for i_custom, cust in enumerate(custom):
            c = cust(reduced_discretization=rrd,
                     discretization=d,
                     reconstructor=rrc,
                     mu=mu,
                     dim=N)
            c = c[0] if hasattr(c, '__len__') else c
            custom_values[i_custom, i_N] = c

    return norms, estimates, errors, conditions, custom_values
