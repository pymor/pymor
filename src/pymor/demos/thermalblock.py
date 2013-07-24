#!/usr/bin/env python
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Thermalblock demo.

Usage:
  thermalblock.py [-ehp] [--estimator-norm=NORM] [--extension-alg=ALG] [--grid=NI] [--help] [--gridform=FORM]
                  [--plot-solutions] [--reductor=RED] [--test=COUNT] XBLOCKS YBLOCKS SNAPSHOTS RBSIZE


Arguments:
  XBLOCKS    Number of blocks in x direction.

  YBLOCKS    Number of blocks in y direction.

  SNAPSHOTS  Number of snapshots for basis generation per component.
             In total SNAPSHOTS^(XBLOCKS * YBLOCKS).

  RBSIZE     Size of the reduced basis


Options:
  -e, --with-estimator   Use error estimator.

  --estimator-norm=NORM  Norm (trivial, h1) in which to calculate the residual
                         [default: trivial].

  --extension-alg=ALG    Basis extension algorithm (trivial, gram_schmidt, h1_gram_schmidt,
                         numpy_trivial, numpy_gram_schmidt) to be used [default: h1_gram_schmidt].
  
  --grid=NI              Use grid with 2*NI*NI elements [default: 100].

  --gridform=FORM        Grid (TriaGrid, RectGrid) [default: TriaGrid].

  -h, --help             Show this message.

  -p, --plot-err         Plot error.

  --plot-solutions       Plot some example solutions.

  --reductor=RED         Reduction algorithm (default, numpy_default) [default: default].

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
from pymor.analyticalproblems import ThermalBlockProblem
from pymor.discretizers import discretize_elliptic_cg
from pymor.reductors.linear import reduce_stationary_affine_linear, numpy_reduce_stationary_affine_linear
from pymor.algorithms import greedy, trivial_basis_extension, gram_schmidt_basis_extension
from pymor.algorithms.basisextension import numpy_trivial_basis_extension, numpy_gram_schmidt_basis_extension
from pymor.grids import RectGrid, TriaGrid, OnedGrid
from pymor.grids import AllDirichletBoundaryInfo

core.getLogger('pymor.algorithms').setLevel('INFO')
core.getLogger('pymor.discretizations').setLevel('INFO')


def thermalblock_demo(args):
    args['XBLOCKS'] = int(args['XBLOCKS'])
    args['YBLOCKS'] = int(args['YBLOCKS'])
    args['--grid'] = int(args['--grid'])
    args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
    args['RBSIZE'] = int(args['RBSIZE'])
    args['--test'] = int(args['--test'])
    args['--estimator-norm'] = args['--estimator-norm'].lower()
    assert args['--estimator-norm'] in {'trivial', 'h1'}
    args['--extension-alg'] = args['--extension-alg'].lower()
    assert args['--extension-alg'] in {'trivial', 'gram_schmidt', 'h1_gram_schmidt', 'numpy_trivial',
                                       'numpy_gram_schmidt'}
    args['--reductor'] = args['--reductor'].lower()
    assert args['--reductor'] in {'default', 'numpy_default'}

    #args['--gridform'] = args['--gridform'].lower()
    assert args['--gridform'] in {'TriaGrid', 'RectGrid'} 

    #print('Solving on TriaGrid(({0},{0}))'.format(args['--grid']))
    print('Solving on {0}'.format(args['--gridform']))
    print('with grid mesh {0} x {0}'.format(args['--grid']))

    print('Setup Problem ...')
    problem = ThermalBlockProblem(num_blocks=(args['XBLOCKS'], args['YBLOCKS']))

    print('Discretize ...')
    gitter2 = {'TriaGrid': TriaGrid((args['--grid'], args['--grid'])), 'RectGrid': RectGrid((args['--grid'], args['--grid']))}
    gitter = gitter2[args['--gridform']]
    discretization, _ = discretize_elliptic_cg(problem, diameter=m.sqrt(2) / args['--grid'], grid = gitter, boundary_info=AllDirichletBoundaryInfo(gitter))
 
    print(discretization.parameter_info())

    if args['--plot-solutions']:
        print('Showing some solutions')
        for mu in discretization.parameter_space.sample_randomly(2):
            print('Solving for diffusion = \n{} ... '.format(mu['diffusion']))
            sys.stdout.flush()
            U = discretization.solve([[1,1],[1,1]])
            discretization.visualize(U)


    print('RB generation ...')

    error_product = discretization.h1_product if args['--estimator-norm'] == 'h1' else None
    reductors = {'default': partial(reduce_stationary_affine_linear, error_product=error_product),
                 'numpy_default': partial(numpy_reduce_stationary_affine_linear, error_product=error_product)}
    reductor = reductors[args['--reductor']]
    extension_algorithms = {'trivial': trivial_basis_extension,
                            'gram_schmidt': gram_schmidt_basis_extension,
                            'h1_gram_schmidt': partial(gram_schmidt_basis_extension, product=discretization.h1_product),
                            'numpy_trivial': numpy_trivial_basis_extension,
                            'numpy_gram_schmidt': numpy_gram_schmidt_basis_extension}
    extension_algorithm = extension_algorithms[args['--extension-alg']]
    greedy_data = greedy(discretization, reductor, discretization.parameter_space.sample_uniformly(args['SNAPSHOTS']),
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
        h1_err = discretization.h1_norm(U - URB)[0]
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

    Problem:
       number of blocks:                   {args[XBLOCKS]}x{args[YBLOCKS]}
       h:                                  sqrt(2)/{args[--grid]}

    Greedy basis generation:
       number of snapshots:                {args[SNAPSHOTS]}^({args[XBLOCKS]}x{args[YBLOCKS]})
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
        discretization.visualize(U - URB)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    thermalblock_demo(args)
