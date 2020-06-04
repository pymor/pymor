#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Modified thermalblock demo using adaptive greedy basis generation algorithm.

Usage:
  thermalblock_adaptive.py [options] RBSIZE
  thermalblock_adaptive.py -h | --help


Arguments:
  RBSIZE     Size of the reduced basis


Options:
  -h, --help                 Show this message.
  --without-estimator        Do not use error estimator for basis generation.
  --extension-alg=ALG        Basis extension algorithm (trivial, gram_schmidt)
                             to be used [default: gram_schmidt].
  --grid=NI                  Use grid with 2*NI*NI elements [default: 100].
  --pickle=PREFIX            Pickle reduced discretizaion, as well as reductor and high-dimensional
                             model to files with this prefix.
  -p, --plot-err             Plot error.
  --plot-solutions           Plot some example solutions.
  --plot-error-sequence      Plot reduction error vs. basis size.
  --product=PROD             Product (euclidean, h1) w.r.t. which to orthonormalize
                             and calculate Riesz representatives [default: h1].
  --reductor=RED             Reductor (error estimator) to choose (traditional, residual_basis)
                             [default: residual_basis]
  --test=COUNT               Use COUNT snapshots for stochastic error estimation
                             [default: 10].
  --ipython-engines=COUNT    If positive, the number of IPython cluster engines to use for
                             parallel greedy search. If zero, no parallelization is performed.
                             [default: 0]
  --ipython-profile=PROFILE  IPython profile to use for parallelization.
  --cache-region=REGION      Name of cache region to use for caching solution snapshots
                             (NONE, MEMORY, DISK, PERSISTENT)
                             [default: NONE]
  --list-vector-array        Solve using ListVectorArray[NumpyVector] instead of NumpyVectorArray.
  --no-visualize-refinement  Do not visualize the training set refinement indicators.
  --validation-mus=VALUE     Size of validation set. [default: 0]
  --rho=VALUE                Maximum allowed ratio between error on validation set and on
                             training set [default: 1.1].
  --gamma=VALUE              Weight factor for age penalty term in refinement indicators
                             [default: 0.2].
  --theta=VALUE              Ratio of elements to refine [default: 0.].
"""

import sys

from docopt import docopt

from pymor.algorithms.adaptivegreedy import rb_adaptive_greedy
from pymor.algorithms.error import reduction_error_analysis
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.core.pickle import dump
from pymor.discretizers.builtin import discretize_stationary_cg
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parallel.default import new_parallel_pool
from pymor.reductors.coercive import CoerciveRBReductor, SimpleCoerciveRBReductor


def thermalblock_demo(args):
    args['--grid'] = int(args['--grid'])
    args['RBSIZE'] = int(args['RBSIZE'])
    args['--test'] = int(args['--test'])
    args['--ipython-engines'] = int(args['--ipython-engines'])
    args['--extension-alg'] = args['--extension-alg'].lower()
    assert args['--extension-alg'] in {'trivial', 'gram_schmidt'}
    args['--product'] = args['--product'].lower()
    assert args['--product'] in {'trivial', 'h1'}
    args['--reductor'] = args['--reductor'].lower()
    assert args['--reductor'] in {'traditional', 'residual_basis'}
    args['--cache-region'] = args['--cache-region'].lower()
    args['--validation-mus'] = int(args['--validation-mus'])
    args['--rho'] = float(args['--rho'])
    args['--gamma'] = float(args['--gamma'])
    args['--theta'] = float(args['--theta'])

    problem = thermal_block_problem(num_blocks=(2, 2))
    functionals = [ExpressionParameterFunctional('diffusion[0]', {'diffusion': 2}),
                   ExpressionParameterFunctional('diffusion[1]**2', {'diffusion': 2}),
                   ExpressionParameterFunctional('diffusion[0]', {'diffusion': 2}),
                   ExpressionParameterFunctional('diffusion[1]', {'diffusion': 2})]
    problem = problem.with_(
        diffusion=problem.diffusion.with_(coefficients=functionals),
    )

    print('Discretize ...')
    fom, _ = discretize_stationary_cg(problem, diameter=1. / args['--grid'])

    if args['--list-vector-array']:
        from pymor.discretizers.builtin.list import convert_to_numpy_list_vector_array
        fom = convert_to_numpy_list_vector_array(fom)

    if args['--cache-region'] != 'none':
        # building a cache_id is only needed for persistent CacheRegions
        cache_id = f"pymordemos.thermalblock_adaptive {args['--grid']}"
        fom.enable_caching(args['--cache-region'], cache_id)

    if args['--plot-solutions']:
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

    product = fom.h1_0_semi_product if args['--product'] == 'h1' else None
    coercivity_estimator = ExpressionParameterFunctional('min([diffusion[0], diffusion[1]**2])',
                                                         fom.parameters)
    reductors = {'residual_basis': CoerciveRBReductor(fom, product=product,
                                                      coercivity_estimator=coercivity_estimator),
                 'traditional': SimpleCoerciveRBReductor(fom, product=product,
                                                         coercivity_estimator=coercivity_estimator)}
    reductor = reductors[args['--reductor']]

    pool = new_parallel_pool(ipython_num_engines=args['--ipython-engines'], ipython_profile=args['--ipython-profile'])
    greedy_data = rb_adaptive_greedy(
        fom, reductor, problem.parameter_space,
        validation_mus=args['--validation-mus'],
        rho=args['--rho'],
        gamma=args['--gamma'],
        theta=args['--theta'],
        use_estimator=not args['--without-estimator'],
        error_norm=fom.h1_0_semi_norm,
        max_extensions=args['RBSIZE'],
        visualize=not args['--no-visualize-refinement']
    )

    rom = greedy_data['rom']

    if args['--pickle']:
        print(f"\nWriting reduced model to file {args['--pickle']}_reduced ...")
        with open(args['--pickle'] + '_reduced', 'wb') as f:
            dump(rom, f)
        print(f"Writing detailed model and reductor to file {args['--pickle']}_detailed ...")
        with open(args['--pickle'] + '_detailed', 'wb') as f:
            dump((fom, reductor), f)

    print('\nSearching for maximum error on random snapshots ...')

    results = reduction_error_analysis(rom,
                                       fom=fom,
                                       reductor=reductor,
                                       estimator=True,
                                       error_norms=(fom.h1_0_semi_norm,),
                                       condition=True,
                                       test_mus=problem.parameter_space.sample_randomly(args['--test']),
                                       basis_sizes=25 if args['--plot-error-sequence'] else 1,
                                       plot=True,
                                       pool=pool)

    real_rb_size = rom.solution_space.dim

    print('''
*** RESULTS ***

Problem:
   number of blocks:                   2x2
   h:                                  sqrt(2)/{args[--grid]}

Greedy basis generation:
   estimator disabled:                 {args[--without-estimator]}
   extension method:                   {args[--extension-alg]}
   product:                            {args[--product]}
   prescribed basis size:              {args[RBSIZE]}
   actual basis size:                  {real_rb_size}
   elapsed time:                       {greedy_data[time]}
'''.format(**locals()))
    print(results['summary'])

    sys.stdout.flush()

    if args['--plot-error-sequence']:
        from matplotlib import pyplot as plt
        plt.show(results['figure'])
    if args['--plot-err']:
        mumax = results['max_error_mus'][0, -1]
        U = fom.solve(mumax)
        URB = reductor.reconstruct(rom.solve(mumax))
        fom.visualize((U, URB, U - URB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                    title='Maximum Error Solution', separate_colorbars=True, block=True)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    thermalblock_demo(args)
