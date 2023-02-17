#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


from time import perf_counter

import numpy as np
from typer import Option, run

from pymor.algorithms.bfgs import error_aware_bfgs
from pymor.algorithms.greedy import rb_greedy
from pymor.algorithms.tr import trust_region, DualTRSurrogate
from pymor.basic import *
from pymor.parameters.functionals import MinThetaParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor
from pymordemos.linear_optimization import create_fom


def main(
    grid_intervals: int = Option(10, help='Grid interval count.'),
    training_samples: int = Option(25, help='Number of samples used for training the reduced basis.')
):
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

    #########################
    # ROM optimization BFGS #
    #########################

    coercivity_estimator = MinThetaParameterFunctional(fom.operator.coefficients, mu_bar)

    training_set = parameter_space.sample_uniformly(training_samples)

    greedy_reductor = CoerciveRBReductor(fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)
    greedy_data = rb_greedy(fom, greedy_reductor, training_set, atol=1e-2)
    greedy_rom = greedy_data['rom']

    tic = perf_counter()
    bfgs_mu, bfgs_data = error_aware_bfgs(greedy_rom, parameter_space)
    toc = perf_counter()
    bfgs_data['time'] = toc - tic

    ################################
    # ROM optimization adaptive TR #
    ################################

    reductor = CoerciveRBReductor(fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)
    surrogate = DualTRSurrogate(reductor, initial_guess)

    tic = perf_counter()
    tr_mu, tr_data = trust_region(surrogate, parameter_space=parameter_space, initial_guess=initial_guess)
    toc = perf_counter()
    tr_data['time'] = toc - tic

    #############
    # Reporting #
    #############

    reference_output = fom.output(reference_mu)[0, 0]
    bfgs_output = fom.output(bfgs_mu)[0, 0]
    tr_output = fom.output(tr_mu)[0, 0]

    report(reference_mu, reference_output, reference_mu, reference_output, reference_data,
           parameter_space.parameters.parse, descriptor=' of optimization with FOM model')
    report(bfgs_mu, bfgs_output, reference_mu, reference_output, bfgs_data,
           parameter_space.parameters.parse, descriptor=' of optimization with fixed ROM model and BFGS method')
    report(tr_mu, tr_output, reference_mu, reference_output, tr_data,
           parameter_space.parameters.parse, descriptor=' of optimization with adaptive ROM model and TR method')


def report(mu, output, reference_mu, reference_output, data, parse, descriptor=None):
    print('')
    print('Report{}:'.format(descriptor or ''))
    print('  mu_min:    {}'.format(parse(mu)))
    print('  J(mu_min): {}'.format(output))
    print('  abs parameter error w.r.t. reference solution: {:.2e}'.format(np.linalg.norm(mu - reference_mu)))
    print('  abs output error w.r.t. reference solution:    {:.2e}'.format(np.linalg.norm(output - reference_output)))
    print('  num iterations:        {}'.format(data['iterations']))
    if 'subproblem_data' in data:
        print('  num fom evaluations:   {}'.format(data['fom_evaluations']))
        print('  num rom evaluations:   {}'.format(data['rom_evaluations']))
        print('  num enrichments:       {}'.format(data['enrichments']))
        subproblem_data = data['subproblem_data']
        print('  num BFGS calls:        {}'.format(sum([subproblem_data[i]['iterations']
            for i in range(len(subproblem_data))])))
        if 'line_search_iterations' in subproblem_data[0]:
            print('  num line search calls: {}'.format(sum(np.concatenate([subproblem_data[i]['line_search_iterations']
            for i in range(len(subproblem_data))]))))
    if 'line_search_iterations' in data:
        print('  num line search calls: {}'.format(sum(data['line_search_iterations'])))
    print('  time:                  {:.5f} seconds'.format(data['time']))
    print('')


if __name__ == '__main__':
    run(main)
