#!/usr/bin/env python
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Thermalblock demo.

Usage:
  burgers.py [-hp] [--grid=NI] [--nt=COUNT] [--plot-solutions] [--plot-ei-err] [--test=COUNT]
             EXP_MIN EXP_MAX EI_SNAPSHOTS EISIZE SNAPSHOTS RBSIZE


Arguments:
  EXP_MIN       Minimal exponent

  EXP_MAX       Maximal exponent

  EI_SNAPSHOTS  Number of snapshots for empirical interpolation.

  EISIZE        Number of interpolation DOFs.

  SNAPSHOTS     Number of snapshots for basis generation.

  RBSIZE        Size of the reduced basis


Options:
  --grid=NI              Use grid with (2*NI)*NI elements [default: 60].

  --nt=COUNT             Number of time steps [default: 100].

  -h, --help             Show this message.

  -p, --plot-err         Plot error.

  --plot-ei-err          Plot empirical interpolation error.

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
from pymor.analyticalproblems.burgers import BurgersProblem
from pymor.discretizers.advection import discretize_instationary_advection_fv
from pymor.reductors import reduce_generic_rb
from pymor.algorithms import greedy
from pymor.algorithms.basisextension import pod_basis_extension
from pymor.algorithms.ei import interpolate_operators
from pymor.la import NumpyVectorArray


core.getLogger('pymor.algorithms').setLevel('INFO')
core.getLogger('pymor.discretizations').setLevel('INFO')


def burgers_demo(args):
    args['--nt'] = int(args['--nt'])
    args['--grid'] = int(args['--grid'])
    args['EXP_MIN'] = int(args['EXP_MIN'])
    args['EXP_MAX'] = int(args['EXP_MAX'])
    args['EI_SNAPSHOTS'] = int(args['EI_SNAPSHOTS'])
    args['EISIZE'] = int(args['EISIZE'])
    args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
    args['RBSIZE'] = int(args['RBSIZE'])
    args['--test'] = int(args['--test'])


    print('Setup Problem ...')
    problem = BurgersProblem(parameter_range=(args['EXP_MIN'], args['EXP_MAX']))

    print('Discretize ...')
    discretization, _ = discretize_instationary_advection_fv(problem, diameter=m.sqrt(2) / args['--grid'],
                                                             nt=args['--nt'])
    print(discretization.operator.grid)

    print(discretization.parameter_info())

    if args['--plot-solutions']:
        print('Showing some solutions')
        for mu in discretization.parameter_space.sample_randomly(2):
            print('Solving for exponent = \n{} ... '.format(mu['exponent']))
            sys.stdout.flush()
            U = discretization.solve(mu)
            discretization.visualize(U)


    ei_discretization, ei_data = interpolate_operators(discretization, 'operator',
                                                       discretization.parameter_space.sample_uniformly(args['EI_SNAPSHOTS']),
                                                       error_norm=discretization.l2_norm,
                                                       target_error=1e-10,
                                                       max_interpolation_dofs=args['EISIZE'])

    if args['--plot-ei-err']:
        print('Showing some EI errors')
        for mu in discretization.parameter_space.sample_randomly(2):
            print('Solving for exponent = \n{} ... '.format(mu['exponent']))
            sys.stdout.flush()
            U = discretization.solve(mu)
            U_EI = ei_discretization.solve(mu)
            ERR = U - U_EI
            print('Error: {}'.format(np.max(discretization.l2_norm(ERR))))
            discretization.visualize(ERR)

        print('Showing interpolation DOFs ...')
        U = np.zeros(U.dim)
        dofs = ei_discretization.operator.interpolation_dofs
        U[dofs] = np.arange(1, len(dofs) + 1)
        U[ei_discretization.operator.source_dofs] += int(len(dofs)/2)
        discretization.visualize(NumpyVectorArray(U))


    print('RB generation ...')
    import cProfile, pstats, io

    reductor = reduce_generic_rb
    extension_algorithm = partial(pod_basis_extension)
    # pr = cProfile.Profile()
    # pr.enable()
    greedy_data = greedy(ei_discretization, reductor, discretization.parameter_space.sample_uniformly(args['SNAPSHOTS']),
                         use_estimator=False, error_norm=lambda U: np.max(discretization.l2_norm(U)),
                         extension_algorithm=extension_algorithm, max_extensions=args['RBSIZE'])
    # pr.disable()
    # pr.dump_stats('greedy.profile')
    rb_discretization, reconstructor = greedy_data['reduced_discretization'], greedy_data['reconstructor']


    print('\nSearching for maximum error on random snapshots ...')

    tic = time.time()
    l2_err_max = -1
    cond_max = -1
    for mu in discretization.parameter_space.sample_randomly(args['--test']):
        print('Solving RB-Scheme for mu = {} ... '.format(mu), end='')
        URB = reconstructor.reconstruct(rb_discretization.solve(mu))
        U = discretization.solve(mu)
        l2_err = np.max(discretization.l2_norm(U - URB))
        if l2_err > l2_err_max:
            l2_err_max = l2_err
            Umax = U
            URBmax = URB
            mumax = mu
        print('L2-error = {}'.format(l2_err))
    toc = time.time()
    t_est = toc - tic
    real_rb_size = len(greedy_data['data'])

    print('''
    *** RESULTS ***

    Problem:
       parameter range:                    ({args[EXP_MIN]}, {args[EXP_MAX]})
       h:                                  sqrt(2)/{args[--grid]}
       nt:                                 {args[--nt]}

    Greedy basis generation:
       number of snapshots:                {args[SNAPSHOTS]}
       prescribed basis size:              {args[RBSIZE]}
       actual basis size:                  {real_rb_size}
       elapsed time:                       {greedy_data[time]}

    Stochastic error estimation:
       number of samples:                  {args[--test]}
       maximal L2-error:                   {l2_err_max}  (mu = {mumax})
       elapsed time:                       {t_est}
    '''.format(**locals()))

    sys.stdout.flush()
    if args['--plot-err']:
        discretization.visualize(U - URB)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    burgers_demo(args)
