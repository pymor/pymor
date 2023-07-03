#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


from time import perf_counter

import numpy as np
from typer import Argument, run

from pymor.algorithms.bfgs import error_aware_bfgs
from pymor.algorithms.greedy import rb_greedy
from pymor.algorithms.tr import coercive_rb_trust_region
from pymor.parameters.functionals import MaxThetaParameterFunctional, MinThetaParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor
from pymordemos.linear_optimization import create_fom


def main(
    output_number: int = Argument(..., help='Selects type of output functional [0, 1], '
                                         + 'where 0 stands for linear and 1 for a quadratic output'),
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    training_samples: int = Argument(..., help='Number of samples used for training the reduced basis.')
):
    """Error aware trust-region method for PDE-constrained parameter optimization problems.

    This demo compares three different approaches for solving PDE-constrained parameter
    optimization problems with linear and quadratic output functionals.
    As optimization method we compare:

    1. The reference method: Projected BFGS, only based on the FOM.
    2. The classical RB method: First standard Greedy for building a ROM, then projected BFGS
       with ROM.
    3. The error aware trust-region method proposed in :cite:`KMOSV21`.

    The methods are compared in terms of computational time, iterations,
    optimization error, and FOM/ROM evaluations.
    """
    assert output_number in [0, 1]
    if output_number == 0:
        fom, mu_bar = create_fom(grid_intervals, output_type='l2')
    else:
        fom, mu_bar = create_fom(grid_intervals, output_type='quadratic')

    parameter_space = fom.parameters.space(0, np.pi)
    initial_guess = fom.parameters.parse([0.25, 2.5])

    ####################
    # FOM optimization #
    ####################

    tic = perf_counter()
    reference_mu, reference_data = error_aware_bfgs(fom, parameter_space)
    toc = perf_counter()
    reference_data['time'] = toc - tic
    # assumes adjoint approach for gradient computation
    reference_data['fom_evaluations'] = sum(reference_data['line_search_iterations']) \
                                        + 2 * (reference_data['iterations'] + 1)

    #########################
    # ROM optimization BFGS #
    #########################

    coercivity_estimator = MinThetaParameterFunctional(fom.operator.coefficients, mu_bar)

    training_set = parameter_space.sample_uniformly(training_samples)

    greedy_reductor = CoerciveRBReductor(fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)
    greedy_data = rb_greedy(fom, greedy_reductor, training_set, atol=1e-4)
    greedy_rom = greedy_data['rom']

    tic = perf_counter()
    greedy_bfgs_mu, greedy_bfgs_data = error_aware_bfgs(greedy_rom, parameter_space)
    toc = perf_counter()
    greedy_bfgs_data['time'] = toc - tic + greedy_data['time']
    greedy_bfgs_data['basis_size'] = greedy_rom.solution_space.dim
    greedy_bfgs_data['fom_evaluations'] = greedy_rom.solution_space.dim

    ################################
    # ROM optimization adaptive TR #
    ################################

    reductor = CoerciveRBReductor(fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)
    if output_number == 1:
        output = reductor.fom.output_functional
        # case for quadratic outputs. Will be obsolete, when an appropriate reductor with error
        # estimators for quadratic functionals is available
        args = {
            'quadratic_output': True,
            'quadratic_output_continuity_estimator': MaxThetaParameterFunctional(output.coefficients, mu_bar,
                                                                                 gamma_mu_bar=1.),
            'quadratic_output_product_name': 'energy'
        }
    else:
        args = {}

    tic = perf_counter()
    tr_mu, tr_data = coercive_rb_trust_region(reductor, parameter_space=parameter_space,
                                              initial_guess=initial_guess, **args)
    toc = perf_counter()
    tr_data['time'] = toc - tic
    tr_data['basis_size'] = tr_data['rom'].solution_space.dim

    #############
    # Reporting #
    #############

    reference_output = fom.output(reference_mu)[0, 0]
    bfgs_output = fom.output(greedy_bfgs_mu)[0, 0]
    tr_output = fom.output(tr_mu)[0, 0]

    report(reference_mu, reference_output, reference_mu, reference_output, reference_data,
           parameter_space.parameters.parse, descriptor=' of optimization with FOM model')
    report(greedy_bfgs_mu, bfgs_output, reference_mu, reference_output, greedy_bfgs_data,
           parameter_space.parameters.parse, descriptor=' of optimization with fixed ROM model and BFGS method')
    report(tr_mu, tr_output, reference_mu, reference_output, tr_data,
           parameter_space.parameters.parse, descriptor=' of optimization with adaptive ROM model and TR method')


def report(mu, output, reference_mu, reference_output, data, parse, descriptor=None):
    print('')
    print('Report{}:'.format(descriptor or ''))
    print('  mu_min:        {}'.format(parse(mu)))
    print('  J(mu_min):     {}'.format(output))
    print('  abs parameter error w.r.t. reference solution: {:.2e}'.format(np.linalg.norm(mu - reference_mu)))
    print('  abs output error w.r.t. reference solution:    {:.2e}'.format(np.linalg.norm(output - reference_output)))
    print('  num iterations:            {}'.format(data['iterations']))
    print('  num fom evaluations:       {}'.format(data['fom_evaluations']), end='')
    print('  (offline/online splitting for estimators not counted)') if 'basis_size' in data else print('')
    if 'subproblem_data' in data:
        print('  num rom evaluations:       {}'.format(data['rom_evaluations']))
        print('  num enrichments:           {}'.format(data['enrichments']))
        subproblem_data = data['subproblem_data']
        print('  total BFGS iterations:     {}'.format(sum([subproblem_data[i]['iterations']
            for i in range(len(subproblem_data))])))
        if 'line_search_iterations' in subproblem_data[0]:
            print('  num line search calls:     {}'.format(
                sum(np.concatenate([subproblem_data[i]['line_search_iterations']
                                    for i in range(len(subproblem_data))]))))
    if 'line_search_iterations' in data:
        print('  num line search calls:     {}'.format(sum(data['line_search_iterations'])))
    if 'basis_size' in data:
        print('  RB size:                   {}'.format(data['basis_size']))
    print('  time:                      {:.5f} seconds'.format(data['time']))
    print('')


if __name__ == '__main__':
    run(main)
