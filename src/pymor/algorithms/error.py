# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import cycle
from numbers import Number
import time

import numpy as np

from pymor.core.logger import getLogger
from pymor.models.basic import StationaryModel
from pymor.parallel.dummy import dummy_pool


def reduction_error_analysis(rom, fom, reductor, test_mus,
                             basis_sizes=0,
                             error_estimator=True, condition=False, error_norms=(),
                             error_norm_names=None,
                             error_estimator_norm_index=0, custom=(), custom_names=None,
                             plot=False, plot_custom_logarithmic=True,
                             pool=dummy_pool):
    """Analyze the model reduction error.

    The maximum model reduction error is estimated by solving the reduced
    |Model| for given random |Parameters|.

    Parameters
    ----------
    rom
        The reduced |Model|.
    fom
        The high-dimensional |Model|.
    reductor
        The reductor which has created `rom`.
    test_mus
        List of |Parameters| to compute the errors for.
    basis_sizes
        Either a list of reduced basis dimensions to consider, or
        the number of dimensions (which are then selected equidistantly,
        always including the maximum reduced space dimension).
        The dimensions are input for the `dim`-Parameter of
        `reductor.reduce()`.
    error_estimator
        If `True` evaluate the error estimator of `rom`
        on the test |Parameters|.
    condition
        If `True`, compute the condition of the reduced system matrix
        for the given test |Parameters| (can only be specified if
        `rom` is an instance of |StationaryModel|
        and `rom.operator` is linear).
    error_norms
        List of norms in which to compute the model reduction error.
    error_norm_names
        Names of the norms given by `error_norms`. If `None`, the
        `name` attributes of the given norms are used.
    error_estimator_norm_index
        When `error_estimator` is `True` and `error_norms` are specified,
        this is the index of the norm in `error_norms` w.r.t. which
        to compute the effectivity of the error estimator.
    custom
        List of custom functions which are evaluated for each test
        |parameter values| and basis size. The functions must have
        the signature ::

            def custom_value(rom, fom, reductor, mu, dim):
                pass
    custom_names
        List of names to be used for plotting custom values.

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

        :mus:                       The test |Parameters| which have been considered.

        :basis_sizes:               The reduced basis dimensions which have been considered.

        :norms:                     |Array| of the norms of the high-dimensional solutions
                                    w.r.t. all given test |Parameters| and norms in `error_norms`.
                                    (Only present when `error_norms` has been specified.)

        :max_norms:                 Maxima of `norms` over the given test |Parameters|.

        :max_norm_mus:              |Parameters| corresponding to `max_norms`.

        :errors:                    |Array| of the norms of the model reduction errors
                                    w.r.t. all given test |Parameters|, reduced basis
                                    dimensions and norms in `error_norms`.
                                    (Only present when `error_norms` has been specified.)

        :max_errors:                Maxima of `errors` over the given test |Parameters|.

        :max_error_mus:             |Parameters| corresponding to `max_errors`.

        :rel_errors:                `errors` divided by `norms`.
                                    (Only present when `error_norms` has been specified.)

        :max_rel_errors:            Maxima of `rel_errors` over the given test |Parameters|.

        :max_rel_error_mus:         |Parameters| corresponding to `max_rel_errors`.

        :error_norm_names:          Names of the given `error_norms`.
                                    (Only present when `error_norms` has been specified.)

        :error_estimates:           |Array| of the model reduction error estimates
                                    w.r.t. all given test |Parameters| and reduced basis
                                    dimensions.
                                    (Only present when `error_estimator` is `True`.)

        :max_error_estimate:        Maxima of `error_estimates` over the given test |Parameters|.

        :max_error_estimate_mus:    |Parameters| corresponding to `max_error_estimates`.

        :effectivities:             `errors` divided by `error_estimates`.
                                    (Only present when `error_estimator` is `True` and `error_norms`
                                    has been specified.)

        :min_effectivities:         Minima of `effectivities` over the given test |Parameters|.

        :min_effectivity_mus:       |Parameters| corresponding to `min_effectivities`.

        :max_effectivities:         Maxima of `effectivities` over the given test |Parameters|.

        :max_effectivity_mus:       |Parameters| corresponding to `max_effectivities`.

        :errors:                    |Array| of the reduced system matrix conditions
                                    w.r.t. all given test |Parameters| and reduced basis
                                    dimensions.
                                    (Only present when `conditions` is `True`.)

        :max_conditions:            Maxima of `conditions` over the given test |Parameters|.

        :max_condition_mus:         |Parameters| corresponding to `max_conditions`.

        :custom_values:             |Array| of custom function evaluations
                                    w.r.t. all given test |Parameters|, reduced basis
                                    dimensions and functions in `custom`.
                                    (Only present when `custom` has been specified.)

        :max_custom_values:         Maxima of `custom_values` over the given test |Parameters|.

        :max_custom_values_mus:     |Parameters| corresponding to `max_custom_values`.

        :time:                      Time (in seconds) needed for the error analysis.

        :summary:                   String containing a summary of all computed quantities for
                                    the largest (last) considered basis size.
    """
    assert not error_norms or (fom and reductor)
    assert error_norm_names is None or len(error_norm_names) == len(error_norms)
    assert custom_names is None or (custom and len(custom_names) == len(custom))
    assert not condition \
        or isinstance(rom, StationaryModel) and rom.operator.linear

    logger = getLogger('pymor.algorithms.error')
    if pool is None or pool is dummy_pool:
        pool = dummy_pool
    else:
        logger.info(f'Using pool of {len(pool)} workers for error analysis')

    tic = time.perf_counter()

    if isinstance(basis_sizes, Number):
        if basis_sizes == 1:
            basis_sizes = [rom.solution_space.dim]
        else:
            if basis_sizes == 0:
                basis_sizes = rom.solution_space.dim + 1
            basis_sizes = min(rom.solution_space.dim + 1, basis_sizes)
            basis_sizes = np.linspace(0, rom.solution_space.dim, basis_sizes).astype(int)
    elif isinstance(basis_sizes, (list, tuple)):
        assert all([isinstance(sz, Number) for sz in basis_sizes])
        basis_sizes = np.array(basis_sizes)
    if error_norm_names is None:
        error_norm_names = tuple(norm.name for norm in error_norms)

    norms, error_estimates, errors, conditions, custom_values = \
        list(zip(*pool.map(_compute_errors, test_mus, fom=fom, reductor=reductor, error_estimator=error_estimator,
                           error_norms=error_norms, condition=condition, custom=custom, basis_sizes=basis_sizes)))
    print()

    result = {}
    result['mus'] = test_mus = np.array(test_mus)
    result['basis_sizes'] = basis_sizes

    summary = [('number of samples', str(len(test_mus)))]

    if error_norms:
        result['error_norm_names'] = error_norm_names
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
            summary.append((f'maximum {name}-norm',
                            f'{norm:.7e} (mu = {error_mu})'))
            summary.append((f'maximum {name}-error',
                            f'{error:.7e} (mu = {error_mu})'))
        result['error_norm_names'] = error_norm_names

    if error_estimator:
        result['error_estimates'] = error_estimates = np.array(error_estimates)
        result['max_error_estimates'] = max_error_estimates = np.max(error_estimates, axis=0)
        result['max_error_estimate_mus'] = max_error_estimate_mus = test_mus[np.argmax(error_estimates, axis=0)]
        summary.append(('maximum estimated error',
                        f'{max_error_estimates[-1]:.7e} (mu = {max_error_estimate_mus[-1]})'))

    if error_estimator and error_norms:
        result['effectivities'] = effectivities = errors[:, error_estimator_norm_index, :] / error_estimates
        result['max_effectivities'] = max_effectivities = np.max(effectivities, axis=0)
        result['max_effectivity_mus'] = max_effectivity_mus = test_mus[np.argmax(effectivities, axis=0)]
        result['min_effectivities'] = min_effectivities = np.min(effectivities, axis=0)
        result['min_effectivity_mus'] = min_effectivity_mus = test_mus[np.argmin(effectivities, axis=0)]
        summary.append(('minimum error estimator effectivity',
                        f'{min_effectivities[-1]:.7e} (mu = {min_effectivity_mus[-1]})'))
        summary.append(('maximum error estimator effectivity',
                        f'{max_effectivities[-1]:.7e} (mu = {max_effectivity_mus[-1]})'))

    if condition:
        result['conditions'] = conditions = np.array(conditions)
        result['max_conditions'] = max_conditions = np.max(conditions, axis=0)
        result['max_condition_mus'] = max_condition_mus = test_mus[np.argmax(conditions, axis=0)]
        summary.append(('maximum system matrix condition',
                        f'{max_conditions[-1]:.7e} (mu = {max_condition_mus[-1]})'))

    if custom:
        result['custom_names'] = custom_names
        result['custom_values'] = custom_values = np.array(custom_values)
        result['max_custom_values'] = max_custom_values = np.max(custom_values, axis=0)
        result['max_custom_values_mus'] = max_custom_values_mus = test_mus[np.argmax(custom_values, axis=0)]
        for i, (value, mu) in enumerate(zip(max_custom_values[:, -1], max_custom_values_mus[:, -1])):
            summary.append((f'maximum {custom_names[i]}',
                            f'{value:.7e} (mu = {mu})'))

    toc = time.perf_counter()
    result['time'] = toc - tic
    summary.append(('elapsed time', str(toc - tic)))

    summary_fields, summary_values = list(zip(*summary))
    summary_field_width = np.max(list(map(len, summary_fields))) + 2
    summary_lines = [f'    {field+":":{summary_field_width}} {value}'
                     for field, value in zip(summary_fields, summary_values)]
    summary = 'Stochastic error estimation:\n' + '\n'.join(summary_lines)
    result['summary'] = summary

    if plot:
        plot_reduction_error_analysis(result, None, plot_custom_logarithmic=plot_custom_logarithmic)

    return result


def plot_reduction_error_analysis(result, max_basis_size=None, plot_effectivities=True, plot_condition=True,
                                  plot_custom_logarithmic=True, plot_custom_with_errors=False):
    """Plots the results from :meth:`reduction_error_analysis`.

    Parameters
    ----------
    result
        Dictionary with entries as returned by :meth:`reduction_error_analysis`.
    max_basis_size
        Only plot results up to this basis size.
    plot_effectivities
        If `True`, plot the effectivities of the a posteriori error estimate.
    plot_condition
        If `True`, plot the condition of the reduced system matrix.
    plot_custom_logarithmic
        If `True`, use a logarithmic y-axis to plot the computed custom
        values.
    plot_custom_with_errors
        It `True`, plot errors and custom values in a single plot (otherwise in separate ones).
    """
    error_norms = 'norms' in result
    error_estimator = 'error_estimates' in result
    condition = 'conditions' in result
    custom = 'custom_values' in result

    basis_sizes = result['basis_sizes']
    if error_norms:
        error_norm_names = result['error_norm_names']
        max_errors = result['max_errors']
        errors = result['errors']
    if error_estimator:
        max_estimates = result['max_error_estimates']
    if error_norms and error_estimator:
        min_effectivities = result['min_effectivities']
        max_effectivities = result['max_effectivities']
    if condition:
        max_conditions = result['max_conditions']
    if custom:
        custom_values = result['custom_values']
        custom_names = result['custom_names']

    max_basis_size = max_basis_size if max_basis_size else len(basis_sizes)

    import matplotlib.pyplot as plt
    from matplotlib.colors import TABLEAU_COLORS as COLORS
    colors = cycle(COLORS.keys())

    fig = plt.figure()
    num_plots = (int(error_norms or error_estimator) + int(error_norms and error_estimator and plot_effectivities)
                 + int(condition and plot_condition) + int(custom and not plot_custom_with_errors))
    current_plot = 1

    if error_norms or error_estimator:
        ax = fig.add_subplot(1, num_plots, current_plot)
        legend = []
        if error_norms:
            for name, errors in zip(error_norm_names, max_errors):
                ax.semilogy(basis_sizes[:max_basis_size], errors[:max_basis_size], color=next(colors))
                legend.append(name)
        if error_estimator:
            ax.semilogy(basis_sizes[:max_basis_size], max_estimates[:max_basis_size], color=next(colors))
            legend.append('error estimator')
        if custom and plot_custom_with_errors:
            axwithyright = ax.twinx()
            max_custom_values = np.max(custom_values, axis=0)
            axwithyright_legend = []
            for i, values in enumerate(max_custom_values):
                values = values.reshape(basis_sizes.shape)
                color = next(colors)
                if plot_custom_logarithmic:
                    axwithyright.semilogy(basis_sizes[:max_basis_size], values[:max_basis_size], color=color)
                else:
                    axwithyright.plot(basis_sizes[:max_basis_size], values[:max_basis_size], color=color)
                axwithyright_legend.append(custom_names[i])
            if len(axwithyright_legend) == 1:
                axwithyright.tick_params(axis='y', labelcolor=color)
                axwithyright.set_ylabel(axwithyright_legend[0], color=color)
            else:
                axwithyright.tick_params(axis='y', labelcolor='gray')
                axwithyright.set_ylabel('custom values (bottom left)', color='gray')
                axwithyright.legend(axwithyright_legend, loc=3)
            ax.set_ylabel('error/estimator (top right)')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        ax.legend(legend, loc=1)
        ax.set_xlabel('ROM size')
        ax.set_title('maximum errors')
        current_plot += 1

    if error_norms and error_estimator:
        ax = fig.add_subplot(1, num_plots, current_plot)
        ax.semilogy(basis_sizes[:max_basis_size], min_effectivities[:max_basis_size])
        ax.semilogy(basis_sizes[:max_basis_size], max_effectivities[:max_basis_size])
        ax.legend(('min', 'max'))
        ax.set_title('error estimator effectivities')
        current_plot += 1

    if condition and plot_condition:
        ax = fig.add_subplot(1, num_plots, current_plot)
        ax.semilogy(basis_sizes[:max_basis_size], max_conditions[:max_basis_size])
        ax.set_title('maximum condition')
        current_plot += 1

    if custom and not plot_custom_with_errors:
        ax = fig.add_subplot(1, num_plots, current_plot)
        legend = []
        max_custom_values = np.max(custom_values, axis=0)
        for i, values in enumerate(max_custom_values):
            values = values.reshape(basis_sizes.shape)
            if plot_custom_logarithmic:
                ax.semilogy(basis_sizes[:max_basis_size], values[:max_basis_size])
            else:
                ax.plot(basis_sizes[:max_basis_size], values[:max_basis_size])
            legend.append(custom_names[i])
        ax.legend(legend)
        ax.set_title('maximum custom values')
        current_plot += 1

    plt.show()


def _compute_errors(mu, fom, reductor, error_estimator, error_norms, condition, custom, basis_sizes):
    import sys

    print('.', end='')
    sys.stdout.flush()

    error_estimates = np.empty(len(basis_sizes)) if error_estimator else None
    norms = np.empty(len(error_norms))
    errors = np.empty((len(error_norms), len(basis_sizes)))
    conditions = np.empty(len(basis_sizes)) if condition else None
    custom_values = np.empty((len(custom), len(basis_sizes)))

    if fom:
        logging_disabled = fom.logging_disabled
        fom.disable_logging()
        U = fom.solve(mu)
        fom.disable_logging(logging_disabled)
        for i_norm, norm in enumerate(error_norms):
            n = norm(U)
            n = n[0] if hasattr(n, '__len__') else n
            norms[i_norm] = n

    for i_N, N in enumerate(basis_sizes):
        rom = reductor.reduce(dims={k: N for k in reductor.bases})
        result = rom.compute(solution=True, solution_error_estimate=error_estimator, mu=mu)
        u = result['solution']
        if error_estimator:
            e = result['solution_error_estimate']
            e = e[0] if hasattr(e, '__len__') else e
            error_estimates[i_N] = e
        if fom and reductor:
            URB = reductor.reconstruct(u)
            for i_norm, norm in enumerate(error_norms):
                e = norm(U - URB)
                e = e[0] if hasattr(e, '__len__') else e
                errors[i_norm, i_N] = e
        if condition:
            conditions[i_N] = np.linalg.cond(rom.operator.assemble(mu).matrix) if N > 0 else 0.
        for i_custom, cust in enumerate(custom):
            c = cust(rom=rom,
                     fom=fom,
                     reductor=reductor,
                     mu=mu,
                     dim=N)
            c = c[0] if hasattr(c, '__len__') else c
            custom_values[i_custom, i_N] = c

    return norms, error_estimates, errors, conditions, custom_values
