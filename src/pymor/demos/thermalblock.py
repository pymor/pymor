#!/usr/bin/env python
# vim: set filetype=python:

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import math as m
import time

import numpy as np

from pymor.analyticalproblems import ThermalBlockProblem
from pymor.discretizers import PoissonCGDiscretizer
from pymor.reductors.linear import StationaryAffineLinearReductor
from pymor.algorithms import GreedyRB

# set log level
# from pymor.core import getLogger; getLogger('pymor').setLevel('INFO')
from pymor.core import getLogger; getLogger('pymor.algorithms').setLevel('INFO')
from pymor.core import getLogger; getLogger('pymor.discretizations').setLevel('INFO')

# parse arguments
if len(sys.argv) < 11:
    sys.exit('Usage: {} X Y N SNAP RB_SIZE USE_ESTIMATOR ESTIMATOR_NORM EXTENSION_ALG TEST_SIZE PLOT'.format(sys.argv[0]))

nx = int(sys.argv[1])
ny = int(sys.argv[2])
n = int(sys.argv[3])
snap_size = int(sys.argv[4])
rb_size = int(sys.argv[5])
use_estimator = bool(int(sys.argv[6]))
estimator_norm = sys.argv[7].lower()
assert estimator_norm in {'trivial', 'h1'}
ext_alg = sys.argv[8]
assert ext_alg in {'trivial', 'gram_schmidt'}
test_size = int(sys.argv[9])
plot = bool(int(sys.argv[10]))



print('Solving on TriaGrid(({0},{0}))'.format(n))

print('Setup Problem ...')
problem = ThermalBlockProblem(num_blocks=(nx, ny))

print('Discretize ...')
discretizer = PoissonCGDiscretizer(problem)
discretization = discretizer.discretize(diameter=m.sqrt(2) / n)

print(discretization.parameter_info())

if plot > 1:
    print('Showing some solutions')
    for mu in discretization.parameter_space.sample_randomly(2):
        print('Solving for diffusion = \n{} ... '.format(mu['diffusion']))
        sys.stdout.flush()
        U = discretization.solve(mu)
        discretization.visualize(U)


print('RB generation ...')

tic = time.time()
error_product = discretization.h1_product if estimator_norm == 'h1' else None
reductor = StationaryAffineLinearReductor(discretization, error_product=error_product)
greedy = GreedyRB(discretization, reductor, use_estimator=use_estimator, error_norm=discretization.h1_norm, extension_algorithm=ext_alg)
RB = greedy.run(discretization.parameter_space.sample_uniformly(snap_size), Nmax=rb_size)
rb_discretization, reconstructor = greedy.rb_discretization, greedy.reconstructor

print('\nSearching for maximum error on random snapshots ...')

toc = time.time()
h1_err_max = -1
cond_max = -1
for mu in discretization.parameter_space.sample_randomly(test_size):
    print('Solving RB-Scheme for mu = {} ... '.format(mu), end='')
    URB = reconstructor.reconstruct(rb_discretization.solve(mu))
    U = discretization.solve(mu)
    h1_err = discretization.h1_norm(U-URB)
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
tac = time.time()

t_offline = toc - tic
t_est = tac - toc

print('''
*** RESULTS ***

Problem:
   number of blocks:                   {nx}x{ny}
   h:                                  sqrt(2)/{n}

Greedy basis generation:
   number of snapshots:                {snap_size}^({nx}x{ny})
   used estimator:                     {use_estimator}
   estimator norm:                     {estimator_norm}
   extension method:                   {ext_alg}
   prescribed basis size:              {rb_size}
   actual basis size:                  {RB.shape[0]}
   elapsed time:                       {t_offline}

Stochastic error estimation:
   number of samples:                  {test_size}
   maximal H1-error:                   {h1_err_max}  (mu = {mumax})
   maximal condition of system matrix: {cond_max}  (mu = {cond_max_mu})
   elapsed time:                       {t_est}
'''.format(**locals()))

sys.stdout.flush()
if plot:
    discretization.visualize(U-URB)
