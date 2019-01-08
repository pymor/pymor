#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
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
                             discretization to files with this prefix.

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

from pymor.algorithms.adaptivegreedy import adaptive_greedy
from pymor.algorithms.error import reduction_error_analysis
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.core.pickle import dump
from pymor.discretizers.cg import discretize_stationary_cg
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace
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
    functionals = [ExpressionParameterFunctional('diffusion[0]', {'diffusion': (2,)}),
                   ExpressionParameterFunctional('diffusion[1]**2', {'diffusion': (2,)}),
                   ExpressionParameterFunctional('diffusion[0]', {'diffusion': (2,)}),
                   ExpressionParameterFunctional('diffusion[1]', {'diffusion': (2,)})]
    problem = problem.with_(
        diffusion=problem.diffusion.with_(coefficients=functionals),
        parameter_space=CubicParameterSpace({'diffusion': (2,)}, 0.1, 1.)
    )

    print('Discretize ...')
    d, _ = discretize_stationary_cg(problem, diameter=1. / args['--grid'])

    if args['--list-vector-array']:
        from pymor.playground.discretizers.numpylistvectorarray import convert_to_numpy_list_vector_array
        d = convert_to_numpy_list_vector_array(d)

    if args['--cache-region'] != 'none':
        d.enable_caching(args['--cache-region'])

    if args['--plot-solutions']:
        print('Showing some solutions')
        Us = ()
        legend = ()
        for mu in d.parameter_space.sample_randomly(2):
            print('Solving for diffusion = \n{} ... '.format(mu['diffusion']))
            sys.stdout.flush()
            Us = Us + (d.solve(mu),)
            legend = legend + (str(mu['diffusion']),)
        d.visualize(Us, legend=legend, title='Detailed Solutions for different parameters', block=True)

    print('RB generation ...')

    product = d.h1_0_semi_product if args['--product'] == 'h1' else None
    coercivity_estimator = ExpressionParameterFunctional('min([diffusion[0], diffusion[1]**2])',
                                                         d.parameter_type)
    reductors = {'residual_basis': CoerciveRBReductor(d, product=product,
                                                      coercivity_estimator=coercivity_estimator),
                 'traditional': SimpleCoerciveRBReductor(d, product=product,
                                                         coercivity_estimator=coercivity_estimator)}
    reductor = reductors[args['--reductor']]

    pool = new_parallel_pool(ipython_num_engines=args['--ipython-engines'], ipython_profile=args['--ipython-profile'])
    greedy_data = adaptive_greedy(
        d, reductor,
        validation_mus=args['--validation-mus'],
        rho=args['--rho'],
        gamma=args['--gamma'],
        theta=args['--theta'],
        use_estimator=not args['--without-estimator'],
        error_norm=d.h1_0_semi_norm,
        max_extensions=args['RBSIZE'],
        visualize=not args['--no-visualize-refinement']
    )

    rd = greedy_data['rd']

    if args['--pickle']:
        print('\nWriting reduced discretization to file {} ...'.format(args['--pickle'] + '_reduced'))
        with open(args['--pickle'] + '_reduced', 'wb') as f:
            dump(rd, f)
        print('Writing detailed discretization and reductor to file {} ...'.format(args['--pickle'] + '_detailed'))
        with open(args['--pickle'] + '_detailed', 'wb') as f:
            dump((d, reductor), f)

    print('\nSearching for maximum error on random snapshots ...')

    results = reduction_error_analysis(rd,
                                       d=d,
                                       reductor=reductor,
                                       estimator=True,
                                       error_norms=(d.h1_0_semi_norm,),
                                       condition=True,
                                       test_mus=args['--test'],
                                       basis_sizes=25 if args['--plot-error-sequence'] else 1,
                                       plot=True,
                                       pool=pool)

    real_rb_size = rd.solution_space.dim

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
        U = d.solve(mumax)
        URB = reductor.reconstruct(rd.solve(mumax))
        d.visualize((U, URB, U - URB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                    title='Maximum Error Solution', separate_colorbars=True, block=True)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    thermalblock_demo(args)
