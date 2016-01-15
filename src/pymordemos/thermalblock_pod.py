#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Laier <m_laie01@uni-muenster.de>

"""Thermalblock with POD demo.

Usage:
  thermalblock_pod.py [options] XBLOCKS YBLOCKS SNAPSHOTS RBSIZE


Arguments:
  XBLOCKS    Number of blocks in x direction.

  YBLOCKS    Number of blocks in y direction.

  SNAPSHOTS  Number of snapshots for basis generation per component.
             In total SNAPSHOTS^(XBLOCKS * YBLOCKS).

  RBSIZE     Size of the reduced basis


Options:
  --grid=NI              Use grid with 2*NI*NI elements [default: 100].

  -h, --help             Show this message.

  -p, --plot-err         Plot error.

  --plot-solutions       Plot some example solutions.

  --plot-error-sequence  Plot reduction error vs. basis size.

  --pod-norm=NORM        Norm (trivial, h1) w.r.t. which to calculate the POD
                         [default: h1].

  --test=COUNT           Use COUNT snapshots for stochastic error estimation
                         [default: 10].
"""

from __future__ import absolute_import, division, print_function

import sys
import time

import numpy as np
from docopt import docopt

from pymor.algorithms.pod import pod
from pymor.algorithms.error import reduction_error_analysis
from pymor.analyticalproblems.thermalblock import ThermalBlockProblem
from pymor.discretizers.elliptic import discretize_elliptic_cg
from pymor.reductors.basic import reduce_generic_rb


def thermalblock_demo(args):
    args['XBLOCKS'] = int(args['XBLOCKS'])
    args['YBLOCKS'] = int(args['YBLOCKS'])
    args['--grid'] = int(args['--grid'])
    args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
    args['RBSIZE'] = int(args['RBSIZE'])
    args['--test'] = int(args['--test'])
    args['--pod-norm'] = args['--pod-norm'].lower()
    assert args['--pod-norm'] in {'trivial', 'h1'}

    print('Solving on TriaGrid(({0},{0}))'.format(args['--grid']))

    print('Setup Problem ...')
    problem = ThermalBlockProblem(num_blocks=(args['XBLOCKS'], args['YBLOCKS']))

    print('Discretize ...')
    discretization, _ = discretize_elliptic_cg(problem, diameter=1. / args['--grid'])

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

    tic = time.time()

    print('Solving on training set ...')
    S_train = discretization.parameter_space.sample_uniformly(args['SNAPSHOTS'])
    snapshots = discretization.operator.source.empty(reserve=len(S_train))
    for mu in S_train:
        snapshots.append(discretization.solve(mu))

    print('Performing POD ...')
    pod_product = discretization.h1_0_semi_product if args['--pod-norm'] == 'h1' else None
    rb = pod(snapshots, modes=args['RBSIZE'], product=pod_product)[0]

    print('Reducing ...')
    reductor = reduce_generic_rb
    rb_discretization, reconstructor, _ = reductor(discretization, rb)

    toc = time.time()
    t_offline = toc - tic

    print('\nSearching for maximum error on random snapshots ...')

    results = reduction_error_analysis(rb_discretization,
                                       discretization=discretization,
                                       reconstructor=reconstructor,
                                       estimator=False,
                                       error_norms=(discretization.h1_0_semi_norm,),
                                       condition=True,
                                       test_mus=args['--test'],
                                       basis_sizes=25 if args['--plot-error-sequence'] else 1,
                                       plot=True)

    real_rb_size = rb_discretization.solution_space.dim

    print('''
*** RESULTS ***

Problem:
   number of blocks:                   {args[XBLOCKS]}x{args[YBLOCKS]}
   h:                                  sqrt(2)/{args[--grid]}

POD basis generation:
   number of snapshots:                {args[SNAPSHOTS]}^({args[XBLOCKS]}x{args[YBLOCKS]})
   pod norm:                           {args[--pod-norm]}
   prescribed basis size:              {args[RBSIZE]}
   actual basis size:                  {real_rb_size}
   elapsed time:                       {t_offline}
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
