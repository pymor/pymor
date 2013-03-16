#!/usr/bin/env python
# vim: set filetype=python:
'''Thermalblock demo.

Usage:
  thermalblock.py [-ehp] [--estimator-norm=NORM] [--extension-alg=ALG] [--grid=NI]
                  [--plot-solutions] [--test=COUNT] XBLOCKS YBLOCKS SNAPSHOTS RBSIZE


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

  --extension-alg=ALG    Basis extension algorithm (trivial, gram_schmidt) to be used
                         [default: gram_schmidt].

  --grid=NI              Use grid with 2*NI*NI elements [default: 100].

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

import numpy as np
from docopt import docopt

from pymor.analyticalproblems import ThermalBlockProblem
from pymor.discretizers import discretize_elliptic_cg
from pymor.reductors.linear import StationaryAffineLinearReductor
from pymor.algorithms import greedy, trivial_basis_extension, gram_schmidt_basis_extension
from pymor.functions import GenericFunction

# set log level
# from pymor.core import getLogger; getLogger('pymor').setLevel('INFO')
from pymor.core import getLogger
getLogger('pymor.algorithms').setLevel('INFO')
getLogger('pymor.discretizations').setLevel('INFO')

# parse arguments
args = docopt(__doc__)

args['XBLOCKS']          = int(args['XBLOCKS'])
args['YBLOCKS']          = int(args['YBLOCKS'])
args['--grid']           = int(args['--grid'])
args['SNAPSHOTS']        = int(args['SNAPSHOTS'])
args['RBSIZE']           = int(args['RBSIZE'])
args['--test']           = int(args['--test'])
args['--estimator-norm'] = args['--estimator-norm'].lower()
assert args['--estimator-norm'] in {'trivial', 'h1'}
args['--extension-alg']  = args['--extension-alg'].lower()
assert args['--extension-alg'] in {'trivial', 'gram_schmidt'}


print('Solving on TriaGrid(({0},{0}))'.format(args['--grid']))

print('Setup Problem ...')
problem = ThermalBlockProblem(num_blocks=(args['XBLOCKS'], args['YBLOCKS']))

print('Discretize ...')
discretization, _ = discretize_elliptic_cg(problem, diameter=m.sqrt(2) / args['--grid'])

print(discretization.parameter_info())

if args['--plot-solutions']:
    print('Showing some solutions')
    for mu in discretization.parameter_space.sample_randomly(2):
        print('Solving for diffusion = \n{} ... '.format(mu['diffusion']))
        sys.stdout.flush()
        U = discretization.solve(mu)
        discretization.visualize(U)


print('RB generation ...')

error_product = discretization.h1_product if args['--estimator-norm'] == 'h1' else None
reductor = StationaryAffineLinearReductor(discretization, error_product=error_product)
extension_algorithm = gram_schmidt_basis_extension if args['--extension-alg'] == 'gram_schmidt' else trivial_basis_extension
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
    h1_err = discretization.h1_norm(U - URB)
    cond = np.linalg.cond(rb_discretization.operator.matrix(mu))
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
   actual basis size:                  {greedy_data[data].shape[0]}
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
