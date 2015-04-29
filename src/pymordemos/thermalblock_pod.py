#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Laier <m_laie01@uni-muenster.de>

"""Thermalblock with POD demo.

Usage:
  thermalblock_pod.py [-hp] [--grid=NI] [--help] [--plot-solutions] [--pod-norm=NORM]
                  [--test=COUNT] XBLOCKS YBLOCKS SNAPSHOTS RBSIZE


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
    S_train = list(discretization.parameter_space.sample_uniformly(args['SNAPSHOTS']))
    snapshots = discretization.operator.source.empty(reserve=len(S_train))
    for mu in S_train:
        snapshots.append(discretization.solve(mu))

    print('Performing POD ...')
    pod_product = discretization.h1_product if args['--pod-norm'] == 'h1' else None
    rb = pod(snapshots, modes=args['RBSIZE'], product=pod_product)[0]

    print('Reducing ...')
    reductor = reduce_generic_rb
    rb_discretization, reconstructor, _ = reductor(discretization, rb)

    toc = time.time()
    t_offline = toc - tic

    print('\nSearching for maximum error on random snapshots ...')

    tic = time.time()
    h1_err_max = -1
    cond_max = -1
    for mu in discretization.parameter_space.sample_randomly(args['--test']):
        print('Solving RB-Scheme for mu = {} ... '.format(mu), end='')
        URB = reconstructor.reconstruct(rb_discretization.solve(mu))
        U = discretization.solve(mu)
        h1_err = discretization.h1_norm(U - URB)[0]
        cond = np.linalg.cond(rb_discretization.operator.assemble(mu)._matrix)
        if h1_err > h1_err_max:
            h1_err_max = h1_err
            mumax = mu
        if cond > cond_max:
            cond_max = cond
            cond_max_mu = mu
        print('H1-error = {}, condition = {}'.format(h1_err, cond))
    toc = time.time()
    t_est = toc - tic
    real_rb_size = len(rb)

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

    Stochastic error estimation:
       number of samples:                  {args[--test]}
       maximal H1-error:                   {h1_err_max}  (mu = {mumax})
       maximal condition of system matrix: {cond_max}  (mu = {cond_max_mu})
       elapsed time:                       {t_est}
    '''.format(**locals()))

    sys.stdout.flush()
    if args['--plot-err']:
        discretization.visualize((U, URB, U - URB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                                 title='Maximum Error Solution', separate_colorbars=True, block=True)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    thermalblock_demo(args)
