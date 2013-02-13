#!/usr/bin/env python
# Proof of concept for solving the poisson equation in 2D using linear finite elements
# and our grid interface

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import math as m

import numpy as np

from pymor.domaindescriptions import RectDomain
from pymor.analyticalproblems import PoissonProblem
from pymor.discretizers import PoissonCGDiscretizer
from pymor.functions import GenericFunction
from pymor.parameters import CubicParameterSpace, ProjectionParameterFunctional, GenericParameterFunctional
from pymor.reductors import GenericRBReductor
from pymor.algorithms import GreedyRB

# set log level
from pymor.core import getLogger; getLogger('pymor.algorithms').setLevel('INFO')
from pymor.core import getLogger; getLogger('pymor.discretizations').setLevel('INFO')

if len(sys.argv) < 6:
    sys.exit('Usage: {} PROBLEM-NUMBER N SNAP RB PLOT'.format(sys.argv[0]))

rhs0 = GenericFunction(lambda X: np.ones(X.shape[:-1]) * 10, dim_domain=2)
rhs1 = GenericFunction(lambda X: (X[..., 0] - 0.5) ** 2 * 1000, dim_domain=2)

nrhs = int(sys.argv[1])
assert 0 <= nrhs <= 1, ValueError('Invalid rhs number.')
rhs = eval('rhs{}'.format(nrhs))

n = int(sys.argv[2])
snapshot_size = int(sys.argv[3])
rbsize = int(sys.argv[4])
plot = bool(int(sys.argv[5]))

d0 = GenericFunction(lambda X: 1 - X[..., 0], dim_domain=2)
d1 = GenericFunction(lambda X: X[..., 0], dim_domain=2)

parameter_space = CubicParameterSpace({'diffusionl':1}, 0.1, 1)
f0 = ProjectionParameterFunctional(parameter_space, 'diffusionl')
f1 = GenericParameterFunctional(parameter_space, lambda mu:1)

print('Solving on TriaGrid(({0},{0}))'.format(n))

print('Setup Problem ...')
problem = PoissonProblem(domain=RectDomain(), rhs=rhs, diffusion_functions=(d0, d1), diffusion_functionals=(f0, f1))

print('Discretize ...')
discretizer = PoissonCGDiscretizer(problem)
discretization = discretizer.discretize(diameter=m.sqrt(2) / n)

print(discretization.parameter_info())


print('RB generation ...')

reductor = GenericRBReductor(discretization)
greedy = GreedyRB(discretization, reductor, error_norm=discretization.h1_norm)
greedy.run(parameter_space.sample_uniformly(snapshot_size), Nmax=rbsize)
rb_discretization, reconstructor = greedy.rb_discretization, greedy.reconstructor

print('\nSearching for maximum error on random snapshots ...')
h1_err_max = -1
for mu in parameter_space.sample_randomly(10):
    print('Solving RB-Scheme for mu = {} ... '.format(mu), end='')
    URB = reconstructor.reconstruct(rb_discretization.solve(mu))
    U = discretization.solve(mu)
    h1_err = discretization.h1_norm(U - URB)
    if h1_err > h1_err_max:
        h1_err_max = h1_err
        Umax = U
        URBmax = URB
        mumax = mu
    print('H1-error = {}'.format(h1_err))

print('')
print('Maximal relative H1-error: {} for mu = {}'.format(h1_err_max, mu))
if plot:
    discretization.visualize(U-URB)
