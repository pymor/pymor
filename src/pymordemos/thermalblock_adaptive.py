#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import sys

from typer import Argument, Option, run

from pymor.algorithms.adaptivegreedy import rb_adaptive_greedy
from pymor.algorithms.error import plot_reduction_error_analysis, reduction_error_analysis
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.core.pickle import dump
from pymor.discretizers.builtin import discretize_stationary_cg
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parallel.default import new_parallel_pool
from pymor.reductors.coercive import CoerciveRBReductor, SimpleCoerciveRBReductor
from pymor.tools.typer import Choices


def main(
    rbsize: int = Argument(..., help='Size of the reduced basis.'),

    cache_region: Choices('none memory disk persistent') = Option(
        'none',
        help='Name of cache region to use for caching solution snapshots.'
    ),
    error_estimator: bool = Option(True, help='Use error estimator for basis generation.'),
    gamma: float = Option(0.2, help='Weight factor for age penalty term in refinement indicators.'),
    grid: int = Option(100, help='Use grid with 2*NI*NI elements.'),
    ipython_engines: int = Option(
        0,
        help='If positive, the number of IPython cluster engines to use for parallel greedy search. '
             'If zero, no parallelization is performed.'
    ),
    ipython_profile: str = Option(None, help='IPython profile to use for parallelization.'),
    list_vector_array: bool = Option(
        False,
        help='Solve using ListVectorArray[NumpyVector] instead of NumpyVectorArray.'
    ),
    pickle: str = Option(
        None,
        help='Pickle reduced discretization, as well as reductor and high-dimensional model to files with this prefix.'
    ),
    plot_err: bool = Option(False, help='Plot error.'),
    plot_solutions: bool = Option(False, help='Plot some example solutions.'),
    plot_error_sequence: bool = Option(False, help='Plot reduction error vs. basis size.'),
    product: Choices('euclidean h1') = Option(
        'h1',
        help='Product  w.r.t. which to orthonormalize and calculate Riesz representatives.'
    ),
    reductor: Choices('traditional residual_basis') = Option(
        'residual_basis',
        help='Reductor (error estimator) to choose (traditional, residual_basis).'
    ),
    rho: float = Option(1.1, help='Maximum allowed ratio between error on validation set and on training set.'),
    test: int = Option(10, help='Use COUNT snapshots for stochastic error estimation.'),
    theta: float = Option(0., help='Ratio of elements to refine.'),
    validation_mus: int = Option(0, help='Size of validation set.'),
    visualize_refinement: bool = Option(True, help='Visualize the training set refinement indicators.'),
):
    """Modified thermalblock demo using adaptive greedy basis generation algorithm."""
    problem = thermal_block_problem(num_blocks=(2, 2))
    functionals = [ExpressionParameterFunctional('diffusion[0]', {'diffusion': 2}),
                   ExpressionParameterFunctional('diffusion[1]**2', {'diffusion': 2}),
                   ExpressionParameterFunctional('diffusion[0]', {'diffusion': 2}),
                   ExpressionParameterFunctional('diffusion[1]', {'diffusion': 2})]
    problem = problem.with_(
        diffusion=problem.diffusion.with_(coefficients=functionals),
    )

    print('Discretize ...')
    fom, _ = discretize_stationary_cg(problem, diameter=1. / grid)

    if list_vector_array:
        from pymor.discretizers.builtin.list import convert_to_numpy_list_vector_array
        fom = convert_to_numpy_list_vector_array(fom)

    if cache_region != 'none':
        # building a cache_id is only needed for persistent CacheRegions
        cache_id = f"pymordemos.thermalblock_adaptive {grid}"
        fom.enable_caching(cache_region.value, cache_id)

    if plot_solutions:
        print('Showing some solutions')
        Us = ()
        legend = ()
        for mu in problem.parameter_space.sample_randomly(2):
            print(f"Solving for diffusion = \n{mu['diffusion']} ... ")
            sys.stdout.flush()
            Us = Us + (fom.solve(mu),)
            legend = legend + (str(mu['diffusion']),)
        fom.visualize(Us, legend=legend, title='Detailed Solutions for different parameters', block=True)

    print('RB generation ...')

    product_op = fom.h1_0_semi_product if product == 'h1' else None
    coercivity_estimator = ExpressionParameterFunctional('min([diffusion[0], diffusion[1]**2])',
                                                         fom.parameters)
    reductors = {'residual_basis': CoerciveRBReductor(fom, product=product_op,
                                                      coercivity_estimator=coercivity_estimator),
                 'traditional': SimpleCoerciveRBReductor(fom, product=product_op,
                                                         coercivity_estimator=coercivity_estimator)}
    reductor = reductors[reductor]

    pool = new_parallel_pool(ipython_num_engines=ipython_engines, ipython_profile=ipython_profile)
    greedy_data = rb_adaptive_greedy(
        fom, reductor, problem.parameter_space,
        validation_mus=validation_mus,
        rho=rho,
        gamma=gamma,
        theta=theta,
        use_error_estimator=error_estimator,
        error_norm=fom.h1_0_semi_norm,
        max_extensions=rbsize,
        visualize=visualize_refinement
    )

    rom = greedy_data['rom']

    if pickle:
        print(f"\nWriting reduced model to file {pickle}_reduced ...")
        with open(pickle + '_reduced', 'wb') as f:
            dump((rom, problem.parameter_space), f)
        print(f"Writing detailed model and reductor to file {pickle}_detailed ...")
        with open(pickle + '_detailed', 'wb') as f:
            dump((fom, reductor), f)

    print('\nSearching for maximum error on random snapshots ...')

    results = reduction_error_analysis(rom,
                                       fom=fom,
                                       reductor=reductor,
                                       error_estimator=True,
                                       error_norms=(fom.h1_0_semi_norm,),
                                       condition=True,
                                       test_mus=problem.parameter_space.sample_randomly(test),
                                       basis_sizes=25 if plot_error_sequence else 1,
                                       pool=pool)

    real_rb_size = rom.solution_space.dim

    print('''
*** RESULTS ***

Problem:
   number of blocks:                   2x2
   h:                                  sqrt(2)/{grid}

Greedy basis generation:
   error estimator enabled:            {error_estimator}
   product:                            {product}
   prescribed basis size:              {rbsize}
   actual basis size:                  {real_rb_size}
   elapsed time:                       {greedy_data[time]}
'''.format(**locals()))
    print(results['summary'])

    sys.stdout.flush()

    if plot_error_sequence:
        plot_reduction_error_analysis(results)
    if plot_err:
        mumax = results['max_error_mus'][0, -1]
        U = fom.solve(mumax)
        URB = reductor.reconstruct(rom.solve(mumax))
        fom.visualize((U, URB, U - URB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                      title='Maximum Error Solution', separate_colorbars=True, block=True)


if __name__ == '__main__':
    run(main)
