#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Modified thermalblock demo using adaptive greedy basis generation algorithm.

Usage:
  thermalblock_adaptive.py [options] RBSIZE
  thermalblock_adaptive.py -h | --help


Arguments:
  RBSIZE     Size of the reduced basis


Options:
  -h, --help                 Show this message.

  --estimator-norm=NORM      Norm (trivial, h1) in which to calculate the residual
                             [default: h1].

  --without-estimator        Do not use error estimator for basis generation.

  --extension-alg=ALG        Basis extension algorithm (trivial, gram_schmidt, h1_gram_schmidt)
                             to be used [default: h1_gram_schmidt].

  --grid=NI                  Use grid with 2*NI*NI elements [default: 100].

  --pickle=PREFIX            Pickle reduced discretizaion, as well as reconstructor and high-dimensional
                             discretization to files with this prefix.

  -p, --plot-err             Plot error.

  --plot-solutions           Plot some example solutions.

  --plot-error-sequence      Plot reduction error vs. basis size.

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

  --visualize-refinement     Visualize the training set refinement indicators.

  --validation-mus           Size of validation set. [default: 0]

  --rho=VALUE                Maximum allowed ratio between error on validation set and on
                             training set [default: 1.1].

  --gamma=VALUE              Weight factor for age penalty term in refinement indicators
                             [default: 0.2].

  --theta=VALUE              Ratio of elements to refine [default: 0.].
"""

import sys
from functools import partial

from docopt import docopt

from pymor.algorithms.basisextension import trivial_basis_extension, gram_schmidt_basis_extension
from pymor.algorithms.adaptivegreedy import adaptive_greedy
from pymor.algorithms.error import reduction_error_analysis
from pymor.analyticalproblems.thermalblock import ThermalBlockProblem
from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.core.pickle import dump
from pymor.discretizers.elliptic import discretize_elliptic_cg
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace
from pymor.parallel.default import new_parallel_pool
from pymor.reductors.coercive import reduce_coercive, reduce_coercive_simple


def thermalblock_demo(args):
    args['--grid'] = int(args['--grid'])
    args['RBSIZE'] = int(args['RBSIZE'])
    args['--test'] = int(args['--test'])
    args['--ipython-engines'] = int(args['--ipython-engines'])
    args['--estimator-norm'] = args['--estimator-norm'].lower()
    assert args['--estimator-norm'] in {'trivial', 'h1'}
    args['--extension-alg'] = args['--extension-alg'].lower()
    assert args['--extension-alg'] in {'trivial', 'gram_schmidt', 'h1_gram_schmidt'}
    args['--reductor'] = args['--reductor'].lower()
    assert args['--reductor'] in {'traditional', 'residual_basis'}
    args['--cache-region'] = args['--cache-region'].lower()
    args['--validation-mus'] = int(args['--validation-mus'])
    args['--rho'] = float(args['--rho'])
    args['--gamma'] = float(args['--gamma'])
    args['--theta'] = float(args['--theta'])

    print('Solving on TriaGrid(({0},{0}))'.format(args['--grid']))

    print('Setup Problem ...')
    problem = ThermalBlockProblem(num_blocks=(2, 2))
    functionals = [ExpressionParameterFunctional('diffusion[0]', {'diffusion': (2,)}),
                   ExpressionParameterFunctional('diffusion[1]**2', {'diffusion': (2,)}),
                   ExpressionParameterFunctional('diffusion[0]', {'diffusion': (2,)}),
                   ExpressionParameterFunctional('diffusion[1]', {'diffusion': (2,)})]
    problem = EllipticProblem(domain=problem.domain,
                              diffusion_functions=problem.diffusion_functions,
                              diffusion_functionals=functionals,
                              rhs=problem.rhs,
                              parameter_space=CubicParameterSpace({'diffusion': (2,)}, 0.1, 1.))

    print('Discretize ...')
    discretization, _ = discretize_elliptic_cg(problem, diameter=1. / args['--grid'])

    if args['--list-vector-array']:
        from pymor.playground.discretizers.numpylistvectorarray import convert_to_numpy_list_vector_array
        discretization = convert_to_numpy_list_vector_array(discretization)

    if args['--cache-region'] != 'none':
        discretization.enable_caching(args['--cache-region'])

    print('The parameter type is {}'.format(discretization.parameter_type))

    if args['--plot-solutions']:
        print('Showing some solutions')
        Us = tuple()
        legend = tuple()
        for mu in discretization.parameter_space.sample_randomly(2):
            print('Solving for diffusion = \n{} ... '.format(mu['diffusion']))
            sys.stdout.flush()
            Us = Us + (discretization.solve(mu),)
            legend = legend + (str(mu['diffusion']),)
        discretization.visualize(Us, legend=legend, title='Detailed Solutions for different parameters', block=True)

    print('RB generation ...')

    error_product = discretization.h1_0_semi_product if args['--estimator-norm'] == 'h1' else None
    coercivity_estimator=ExpressionParameterFunctional('min([diffusion[0], diffusion[1]**2])', discretization.parameter_type)
    reductors = {'residual_basis': partial(reduce_coercive, error_product=error_product,
                                   coercivity_estimator=coercivity_estimator),
                 'traditional': partial(reduce_coercive_simple, error_product=error_product,
                                        coercivity_estimator=coercivity_estimator)}
    reductor = reductors[args['--reductor']]
    extension_algorithms = {'trivial': trivial_basis_extension,
                            'gram_schmidt': gram_schmidt_basis_extension,
                            'h1_gram_schmidt': partial(gram_schmidt_basis_extension, product=discretization.h1_0_semi_product)}
    extension_algorithm = extension_algorithms[args['--extension-alg']]

    pool = new_parallel_pool(ipython_num_engines=args['--ipython-engines'], ipython_profile=args['--ipython-profile'])
    greedy_data = adaptive_greedy(discretization, reductor,
                                  validation_mus=args['--validation-mus'], rho=args['--rho'], gamma=args['--gamma'],
                                  theta=args['--theta'],
                                  use_estimator=not args['--without-estimator'], error_norm=discretization.h1_0_semi_norm,
                                  extension_algorithm=extension_algorithm, max_extensions=args['RBSIZE'],
                                  visualize=args['--visualize-refinement'])

    rb_discretization, reconstructor = greedy_data['reduced_discretization'], greedy_data['reconstructor']

    if args['--pickle']:
        print('\nWriting reduced discretization to file {} ...'.format(args['--pickle'] + '_reduced'))
        with open(args['--pickle'] + '_reduced', 'wb') as f:
            dump(rb_discretization, f)
        print('Writing detailed discretization and reconstructor to file {} ...'.format(args['--pickle'] + '_detailed'))
        with open(args['--pickle'] + '_detailed', 'wb') as f:
            dump((discretization, reconstructor), f)

    print('\nSearching for maximum error on random snapshots ...')

    results = reduction_error_analysis(rb_discretization,
                                       discretization=discretization,
                                       reconstructor=reconstructor,
                                       estimator=True,
                                       error_norms=(discretization.h1_0_semi_norm,),
                                       condition=True,
                                       test_mus=args['--test'],
                                       basis_sizes=25 if args['--plot-error-sequence'] else 1,
                                       plot=True,
                                       pool=pool)

    real_rb_size = rb_discretization.solution_space.dim

    print('''
*** RESULTS ***

Problem:
   number of blocks:                   2x2
   h:                                  sqrt(2)/{args[--grid]}

Greedy basis generation:
   estimator disabled:                 {args[--without-estimator]}
   estimator norm:                     {args[--estimator-norm]}
   extension method:                   {args[--extension-alg]}
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
        U = discretization.solve(mumax)
        URB = reconstructor.reconstruct(rb_discretization.solve(mumax))
        discretization.visualize((U, URB, U - URB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                                 title='Maximum Error Solution', separate_colorbars=True, block=True)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    thermalblock_demo(args)
