#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Dune Thermalblock demo.

Usage:
  thermalblock.py [-ehp] [--estimator-norm=NORM] [--extension-alg=ALG] [--help]
                  [--plot-solutions] [--test=COUNT] SNAPSHOTS RBSIZE


Arguments:
  SNAPSHOTS  Number of snapshots for basis generation per component.
             In total SNAPSHOTS^(XBLOCKS * YBLOCKS).

  RBSIZE     Size of the reduced basis


Options:
  -e, --with-estimator   Use error estimator.

  --estimator-norm=NORM  Norm (trivial, h1) in which to calculate the residual
                         [default: trivial].

  --extension-alg=ALG    Basis extension algorithm (trivial, gram_schmidt)
                         to be used [default: gram_schmidt].

  -h, --help             Show this message.

  -p, --plot-err         Plot error.

  --plot-solutions       Plot some example solutions.

  --test=COUNT           Use COUNT snapshots for stochastic error estimation
                         [default: 10].
'''

from __future__ import absolute_import, division, print_function

import sys
import math as m
import time
from functools import partial

import numpy as np
from docopt import docopt

import pymor.core as core
core.logger.MAX_HIERACHY_LEVEL = 2
from pymor.playground.discretizations.dune import DuneLinearEllipticCGDiscretization
from pymor.playground.la.dunevectorarray import DuneVectorArray
from pymor.reductors.linear import reduce_stationary_affine_linear
from pymor.algorithms import greedy, trivial_basis_extension, gram_schmidt_basis_extension
core.getLogger('pymor.algorithms').setLevel('INFO')
core.getLogger('pymor.discretizations').setLevel('INFO')


def dune_thermalblock_demo(args):
    args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
    args['RBSIZE'] = int(args['RBSIZE'])
    args['--test'] = int(args['--test'])
    args['--estimator-norm'] = args['--estimator-norm'].lower()
    assert args['--estimator-norm'] in {'trivial', 'h1'}
    args['--extension-alg'] = args['--extension-alg'].lower()
    assert args['--extension-alg'] in {'trivial', 'gram_schmidt'}

    print('Discretize ...')
    discretization = DuneLinearEllipticCGDiscretization()

    print('The parameter type is {}'.format(discretization.parameter_type))

    if args['--plot-solutions']:
        print('Showing some solutions')
        for mu in discretization.parameter_space.sample_randomly(2):
            print('Solving for diffusion = \n{} ... '.format(mu['diffusion']))
            sys.stdout.flush()
            U = discretization.solve(mu)
            discretization.visualize(U)


    print('RB generation ...')

    error_product = discretization.h1_product if args['--estimator-norm'] == 'h1' else None
    reductor = partial(reduce_stationary_affine_linear, error_product=error_product)
    extension_algorithms = {'trivial': trivial_basis_extension,
                            'gram_schmidt': partial(gram_schmidt_basis_extension, product=discretization.h1_product)}
    extension_algorithm = extension_algorithms[args['--extension-alg']]
    greedy_data = greedy(discretization, reductor, discretization.parameter_space.sample_uniformly(args['SNAPSHOTS']),
                         initial_data = DuneVectorArray.empty(dim=discretization.solution_dim),
                         use_estimator=args['--with-estimator'], error_norm=discretization.h1_norm,
                         extension_algorithm=extension_algorithm, max_extensions=args['RBSIZE'])
    rb_discretization, reconstructor = greedy_data['reduced_discretization'], greedy_data['reconstructor']


    print('\nSearching for maximum error on random snapshots ...')

    tic = time.time()
    h1_err_max = -1
    cond_max = -1
    for mu in discretization.parameter_space.sample_randomly(args['--test']):
        print('Solving RB-Scheme for mu = {} ... '.format(mu), end='')
        URB = reconstructor.reconstruct(rb_discretization.solve(mu))
        U = discretization.solve(mu)
        h1_err = discretization.h1_norm(U - URB)
        cond = np.linalg.cond(rb_discretization.operator.assemble(mu)._matrix)
        if h1_err > h1_err_max:
            h1_err_max = h1_err
            Umax = U
            URBmax = URB
            mumax = mu
        if cond > cond_max:
            cond_max = cond
            cond_max_mu = mu
        print('H1-error = {}, condition = {}'.format(h1_err, cond))
    toc = time.time()
    t_est = toc - tic
    real_rb_size = len(greedy_data['data'])

    print('''
    *** RESULTS ***

    Greedy basis generation:
       used estimator:                     {args[--with-estimator]}
       estimator norm:                     {args[--estimator-norm]}
       extension method:                   {args[--extension-alg]}
       prescribed basis size:              {args[RBSIZE]}
       actual basis size:                  {real_rb_size}
       elapsed time:                       {greedy_data[time]}

    Stochastic error estimation:
       number of samples:                  {args[--test]}
       maximal H1-error:                   {h1_err_max}  (mu = {mumax})
       maximal condition of system matrix: {cond_max}  (mu = {cond_max_mu})
       elapsed time:                       {t_est}
    '''.format(**locals()))

    sys.stdout.flush()
    if args['--plot-err']:
        discretization.visualize(Umax - URBmax)
        # mu = {'diffusion': [1, 0.1, 1, 1]}
        # U = discretization.solve(mu=mu)
        # u = U._vectors[0]
        # ops = [op.dune_op for op in discretization.operator.operators]
        # op_aff = discretization.operator.operator_affine_part.dune_op
        # opsu = [o.apply(u) for o in ops]
        # for u, m in zip(opsu, mu['diffusion']):
        #     u.scale(m)
        # opaffu = op_aff.apply(u)
        # uu = opaffu
        # for u in opsu:
        #     uu = uu.add(u)
        # uu.scale(-1)

        # ops = discretization.operator.operators
        # op_aff = discretization.operator.operator_affine_part
        # opsu = [o.apply(U) for o in ops]
        # opsu2 = [u * m for u, m in zip(opsu, mu['diffusion'])]
        # opaffu = op_aff.apply(U)
        # #uu2 = opaffu + sum(opsu2)

        # uu2 = discretization.operator.apply(U, mu=mu)
        # err = uu.add(uu2._vectors[0])


        # #err = uu.add(discretization.rhs.dune_vec)
        # #err = uu.add(discretization.operator.apply(U, mu=mu)._vectors[0])
        # discretization.example.visualize(err, 'blubb', 'error')

        #err = discretization.operator.apply(U, mu=mu) - discretization.rhs.as_vector()
        #discretization.visualize(err)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    dune_thermalblock_demo(args)
