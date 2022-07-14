#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import sys
import time

from typer import Argument, Option, run

from pymor.algorithms.error import plot_reduction_error_analysis, reduction_error_analysis
from pymor.core.pickle import dump
from pymor.parallel.default import new_parallel_pool
from pymor.tools.typer import Choices


def main(
    xblocks: int = Argument(..., help='Number of blocks in x direction.'),
    yblocks: int = Argument(..., help='Number of blocks in y direction.'),
    snapshots: int = Argument(
        ...,
        help='naive: ignored\n\n'
             'greedy/pod: Number of training_set parameters per block '
             '(in total SNAPSHOTS^(XBLOCKS * YBLOCKS) parameters).\n\n'
             'adaptive_greedy: size of validation set.\n\n'
    ),
    rbsize: int = Argument(..., help='Size of the reduced basis.'),

    adaptive_greedy_gamma: float = Option(0.2, help='See pymor.algorithms.adaptivegreedy.'),
    adaptive_greedy_rho: float = Option(1.1, help='See pymor.algorithms.adaptivegreedy.'),
    adaptive_greedy_theta: float = Option(0., help='See pymor.algorithms.adaptivegreedy.'),
    alg: Choices('naive greedy adaptive_greedy pod') = Option('greedy', help='The model reduction algorithm to use.'),
    cache_region: Choices('none memory disk persistent') = Option(
        'none',
        help='Name of cache region to use for caching solution snapshots.'
    ),
    extension_alg: Choices('trivial gram_schmidt') = Option(
        'gram_schmidt',
        help='Basis extension algorithm to be used.'
    ),
    fenics: bool = Option(False, help='Use FEniCS model.'),
    greedy_with_error_estimator: bool = Option(True, help='Use error estimator for basis generation.'),
    grid: int = Option(100, help='Use grid with 4*NI*NI elements'),
    ipython_engines: int = Option(
        None,
        help='If positive, the number of IPython cluster engines to use for '
             'parallel greedy search. If zero, no parallelization is performed.'
    ),
    ipython_profile: str = Option(None, help='IPython profile to use for parallelization.'),
    list_vector_array: bool = Option(
        False,
        help='Solve using ListVectorArray[NumpyVector] instead of NumpyVectorArray.'
    ),
    order: int = Option(1, help='Polynomial order of the Lagrange finite elements to use in FEniCS.'),
    pickle: str = Option(
        None,
        help='Pickle reduced model, as well as reductor and high-dimensional model '
             'to files with this prefix.'
    ),
    product: Choices('euclidean h1') = Option(
        'h1',
        help='Product w.r.t. which to orthonormalize and calculate Riesz representatives.'
    ),
    plot_err: bool = Option(False, help='Plot error'),
    plot_error_sequence: bool = Option(False, help='Plot reduction error vs. basis size.'),
    plot_solutions: bool = Option(False, help='Plot some example solutions.'),
    reductor: Choices('traditional residual_basis') = Option(
        'residual_basis',
        help='Reductor (error estimator) to choose.'
    ),
    test: int = Option(10, help='Use COUNT snapshots for stochastic error estimation.'),
):
    """Thermalblock demo."""
    if fenics and cache_region != 'none':
        raise ValueError('Caching of high-dimensional solutions is not supported for FEniCS model.')
    if not fenics and order != 1:
        raise ValueError('Higher-order finite elements only supported for FEniCS model.')

    pool = new_parallel_pool(ipython_num_engines=ipython_engines, ipython_profile=ipython_profile)

    if fenics:
        fom, fom_summary = discretize_fenics(xblocks, yblocks, grid, order)
    else:
        fom, fom_summary = discretize_pymor(xblocks, yblocks, grid, list_vector_array)

    parameter_space = fom.parameters.space(0.1, 1.)

    if cache_region != 'none':
        # building a cache_id is only needed for persistent CacheRegions
        cache_id = (f"pymordemos.thermalblock {fenics} {xblocks} {yblocks}"
                    f"{grid} {order}")
        fom.enable_caching(cache_region.value, cache_id)

    if plot_solutions:
        print('Showing some solutions')
        Us = ()
        legend = ()
        for mu in parameter_space.sample_randomly(2):
            print(f"Solving for diffusion = \n{mu['diffusion']} ... ")
            sys.stdout.flush()
            Us = Us + (fom.solve(mu),)
            legend = legend + (str(mu['diffusion']),)
        fom.visualize(Us, legend=legend, title='Detailed Solutions for different parameters',
                      separate_colorbars=False, block=True)

    print('RB generation ...')

    # define estimator for coercivity constant
    from pymor.parameters.functionals import ExpressionParameterFunctional
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', fom.parameters)

    # inner product for computation of Riesz representatives
    product = fom.h1_0_semi_product if product == 'h1' else None

    if reductor == 'residual_basis':
        from pymor.reductors.coercive import CoerciveRBReductor
        reductor = CoerciveRBReductor(fom, product=product, coercivity_estimator=coercivity_estimator,
                                      check_orthonormality=False)
    elif reductor == 'traditional':
        from pymor.reductors.coercive import SimpleCoerciveRBReductor
        reductor = SimpleCoerciveRBReductor(fom, product=product, coercivity_estimator=coercivity_estimator,
                                            check_orthonormality=False)
    else:
        assert False  # this should never happen

    if alg == 'naive':
        rom, red_summary = reduce_naive(fom=fom, reductor=reductor, parameter_space=parameter_space,
                                        basis_size=rbsize)
    elif alg == 'greedy':
        parallel = greedy_with_error_estimator or not fenics  # cannot pickle FEniCS model
        rom, red_summary = reduce_greedy(fom=fom, reductor=reductor, parameter_space=parameter_space,
                                         snapshots_per_block=snapshots,
                                         extension_alg_name=extension_alg.value,
                                         max_extensions=rbsize,
                                         use_error_estimator=greedy_with_error_estimator,
                                         pool=pool if parallel else None)
    elif alg == 'adaptive_greedy':
        parallel = greedy_with_error_estimator or not fenics  # cannot pickle FEniCS model
        rom, red_summary = reduce_adaptive_greedy(fom=fom, reductor=reductor, parameter_space=parameter_space,
                                                  validation_mus=snapshots,
                                                  extension_alg_name=extension_alg.value,
                                                  max_extensions=rbsize,
                                                  use_error_estimator=greedy_with_error_estimator,
                                                  rho=adaptive_greedy_rho,
                                                  gamma=adaptive_greedy_gamma,
                                                  theta=adaptive_greedy_theta,
                                                  pool=pool if parallel else None)
    elif alg == 'pod':
        rom, red_summary = reduce_pod(fom=fom, reductor=reductor, parameter_space=parameter_space,
                                      snapshots_per_block=snapshots,
                                      basis_size=rbsize)
    else:
        assert False  # this should never happen

    if pickle:
        print(f"\nWriting reduced model to file {pickle}_reduced ...")
        with open(pickle + '_reduced', 'wb') as f:
            dump((rom, parameter_space), f)
        if not fenics:  # FEniCS data structures do not support serialization
            print(f"Writing detailed model and reductor to file {pickle}_detailed ...")
            with open(pickle + '_detailed', 'wb') as f:
                dump((fom, reductor), f)

    print('\nSearching for maximum error on random snapshots ...')

    results = reduction_error_analysis(rom,
                                       fom=fom,
                                       reductor=reductor,
                                       error_estimator=True,
                                       error_norms=(fom.h1_0_semi_norm, fom.l2_norm),
                                       condition=True,
                                       test_mus=parameter_space.sample_randomly(test, seed=999),
                                       basis_sizes=0 if plot_error_sequence else 1,
                                       pool=None if fenics else pool  # cannot pickle FEniCS model
                                       )

    print('\n*** RESULTS ***\n')
    print(fom_summary)
    print(red_summary)
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

    global test_results
    test_results = results


def discretize_pymor(xblocks, yblocks, grid_num_intervals, use_list_vector_array):
    from pymor.analyticalproblems.thermalblock import thermal_block_problem
    from pymor.discretizers.builtin import discretize_stationary_cg
    from pymor.discretizers.builtin.list import convert_to_numpy_list_vector_array

    print('Discretize ...')
    # setup analytical problem
    problem = thermal_block_problem(num_blocks=(xblocks, yblocks))

    # discretize using continuous finite elements
    fom, _ = discretize_stationary_cg(problem, diameter=1. / grid_num_intervals)

    if use_list_vector_array:
        fom = convert_to_numpy_list_vector_array(fom)

    summary = f'''pyMOR model:
   number of blocks: {xblocks}x{yblocks}
   grid intervals:   {grid_num_intervals}
   ListVectorArray:  {use_list_vector_array}
'''

    return fom, summary


def discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order):
    from pymor.tools import mpi

    if mpi.parallel:
        from pymor.models.mpi import mpi_wrap_model
        fom = mpi_wrap_model(lambda: _discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order),
                             use_with=True, pickle_local_spaces=False)
    else:
        fom = _discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order)

    summary = f'''FEniCS model:
   number of blocks:      {xblocks}x{yblocks}
   grid intervals:        {grid_num_intervals}
   finite element order:  {element_order}
'''

    return fom, summary


def _discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order):

    from pymor.analyticalproblems.thermalblock import thermal_block_problem
    from pymor.discretizers.fenics import discretize_stationary_cg

    print('Discretize ...')
    # setup analytical problem
    problem = thermal_block_problem(num_blocks=(xblocks, yblocks))

    # discretize using continuous finite elements
    fom, _ = discretize_stationary_cg(problem, diameter=1. / grid_num_intervals, degree=element_order)

    return fom


def reduce_naive(fom, reductor, parameter_space, basis_size):

    tic = time.perf_counter()

    training_set = parameter_space.sample_randomly(basis_size)

    for mu in training_set:
        reductor.extend_basis(fom.solve(mu), method='trivial')

    rom = reductor.reduce()

    elapsed_time = time.perf_counter() - tic

    summary = f'''Naive basis generation:
   basis size set: {basis_size}
   elapsed time:   {elapsed_time}
'''

    return rom, summary


def reduce_greedy(fom, reductor, parameter_space, snapshots_per_block,
                  extension_alg_name, max_extensions, use_error_estimator, pool):

    from pymor.algorithms.greedy import rb_greedy

    # run greedy
    training_set = parameter_space.sample_uniformly(snapshots_per_block)
    greedy_data = rb_greedy(fom, reductor, training_set,
                            use_error_estimator=use_error_estimator, error_norm=fom.h1_0_semi_norm,
                            extension_params={'method': extension_alg_name}, max_extensions=max_extensions,
                            pool=pool)
    rom = greedy_data['rom']

    # generate summary
    real_rb_size = rom.solution_space.dim
    training_set_size = len(training_set)
    summary = f'''Greedy basis generation:
   size of training set:   {training_set_size}
   error estimator used:   {use_error_estimator}
   extension method:       {extension_alg_name}
   prescribed basis size:  {max_extensions}
   actual basis size:      {real_rb_size}
   elapsed time:           {greedy_data["time"]}
'''

    return rom, summary


def reduce_adaptive_greedy(fom, reductor, parameter_space, validation_mus,
                           extension_alg_name, max_extensions, use_error_estimator,
                           rho, gamma, theta, pool):

    from pymor.algorithms.adaptivegreedy import rb_adaptive_greedy

    # run greedy
    greedy_data = rb_adaptive_greedy(fom, reductor, parameter_space, validation_mus=-validation_mus,
                                     use_error_estimator=use_error_estimator, error_norm=fom.h1_0_semi_norm,
                                     extension_params={'method': extension_alg_name}, max_extensions=max_extensions,
                                     rho=rho, gamma=gamma, theta=theta, pool=pool)
    rom = greedy_data['rom']

    # generate summary
    real_rb_size = rom.solution_space.dim
    # the validation set consists of `validation_mus` random parameters plus the centers of the
    # adaptive sample set cells
    validation_mus += 1
    summary = f'''Adaptive greedy basis generation:
   initial size of validation set:  {validation_mus}
   error estimator used:            {use_error_estimator}
   extension method:                {extension_alg_name}
   prescribed basis size:           {max_extensions}
   actual basis size:               {real_rb_size}
   elapsed time:                    {greedy_data["time"]}
'''

    return rom, summary


def reduce_pod(fom, reductor, parameter_space, snapshots_per_block, basis_size):
    from pymor.algorithms.pod import pod

    tic = time.perf_counter()

    training_set = parameter_space.sample_uniformly(snapshots_per_block)

    print('Solving on training set ...')
    snapshots = fom.operator.source.empty(reserve=len(training_set))
    for mu in training_set:
        snapshots.append(fom.solve(mu))

    print('Performing POD ...')
    basis, singular_values = pod(snapshots, modes=basis_size, product=reductor.products['RB'])

    print('Reducing ...')
    reductor.extend_basis(basis, method='trivial')
    rom = reductor.reduce()

    elapsed_time = time.perf_counter() - tic

    # generate summary
    real_rb_size = rom.solution_space.dim
    training_set_size = len(training_set)
    summary = f'''POD basis generation:
   size of training set:   {training_set_size}
   prescribed basis size:  {basis_size}
   actual basis size:      {real_rb_size}
   elapsed time:           {elapsed_time}
'''

    return rom, summary


if __name__ == '__main__':
    run(main)
