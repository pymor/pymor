#!/usr/bin/env python
# vim: set filetype=python:

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import math as m

import numpy as np

from pymor.analyticalproblems import ThermalBlockProblem
from pymor.discretizers import PoissonCGDiscretizer
from pymor.reductors.linear import StationaryAffineLinearReductor
from pymor.algorithms import GreedyRB

# set log level
# from pymor.core import getLogger; getLogger('pymor').setLevel('INFO')
from pymor.core import getLogger; getLogger('pymor.algorithms').setLevel('INFO')
from pymor.core import getLogger; getLogger('pymor.discretizations').setLevel('INFO')

if len(sys.argv) < 8:
    sys.exit('Usage: {} X Y N SNAP RB EXT_ALG PLOT'.format(sys.argv[0]))

nx = int(sys.argv[1])
ny = int(sys.argv[2])
n = int(sys.argv[3])
snap_size = int(sys.argv[4])
rb_size = int(sys.argv[5])
ext_alg = sys.argv[6]
plot = bool(int(sys.argv[7]))

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

reductor = StationaryAffineLinearReductor(discretization, error_product=discretization.h1_product)
greedy = GreedyRB(discretization, reductor, error_norm=discretization.h1_norm, extension_algorithm=ext_alg)
RB = greedy.run(discretization.parameter_space.sample_uniformly(snap_size), Nmax=rb_size)
rb_discretization, reconstructor = greedy.rb_discretization, greedy.reconstructor

print('\nSearching for maximum error on random snapshots ...')
h1_err_max = -1
cond_max = -1
for mu in discretization.parameter_space.sample_randomly(20):
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

print('')
print('Basis size: {}'.format(RB.shape[0]))
print('')
print('Maximal H1-error: {} for mu = {}'.format(h1_err_max, mu))
print('')
print('Maximal condition of system matrix: {} for mu = {}'.format(cond_max, cond_max_mu))
sys.stdout.flush()
if plot:
    discretization.visualize(U-URB)
