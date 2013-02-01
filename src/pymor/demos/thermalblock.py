#!/usr/bin/env python
# vim: set filetype=python:

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import math as m

import numpy as np

from pymor.analyticalproblems import ThermalBlockProblem
from pymor.discretizers import PoissonCGDiscretizer
from pymor.reductors import GenericRBReductor

# set log level
# from pymor.core import getLogger; getLogger('pymor').setLevel('INFO')

if len(sys.argv) < 5:
    sys.exit('Usage: {} X Y N RB PLOT'.format(sys.argv[0]))

nx = int(sys.argv[1])
ny = int(sys.argv[2])
n = int(sys.argv[3])
rbsize = int(sys.argv[4])
plot = bool(int(sys.argv[5]))

print('Solving on TriaGrid(({0},{0}))'.format(n))

print('Setup Problem ...')
problem = ThermalBlockProblem(num_blocks=(nx, ny))

print('Discretize ...')
discretizer = PoissonCGDiscretizer(problem)
discretization = discretizer.discretize(diameter=m.sqrt(2) / n)

print(discretization.parameter_info())

if print:
    print('Showing some solutions')
    for mu in discretization.parameter_space.sample_randomly(2):
        print('Solving for diffusion = \n{} ... '.format(mu['diffusion']))
        sys.stdout.flush()
        U = discretization.solve(mu)
        discretization.visualize(U)


mu_snap = tuple(discretization.parameter_space.sample_uniformly(rbsize))
RB = np.empty((len(mu_snap), discretization.operator.range_dim))
for i, mu in enumerate(mu_snap):
    print('Solving for mu = {} ...'.format(mu))
    RB[i] = discretization.solve(mu)

print('Projecting operators ...')

reductor = GenericRBReductor(discretization)
rb_discretization = reductor.reduce(RB)

l2_err_max = -1
for mu in discretization.parameter_space.sample_randomly(10):
    print('Solving RB-Scheme for mu = {} ... '.format(mu), end='')
    URB = np.dot(rb_discretization.solve(mu), RB)
    U = discretization.solve(mu)
    l2_err = np.sqrt(np.sum((U-URB)**2)) / np.sqrt(np.sum(U**2))
    if l2_err > l2_err_max:
        l2_err_max = l2_err
        Umax = U
        URBmax = URB
        mumax = mu
    print('rel L2-error = {}'.format(l2_err))

print('')
print('Maximal relative L2-error: {} for mu = {}'.format(l2_err_max, mu))
sys.stdout.flush()
if plot:
    discretization.visualize(U-URB)
