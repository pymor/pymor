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

# set log level
from pymor.core import getLogger; getLogger('pymor').setLevel('INFO')

if len(sys.argv) < 5:
    sys.exit('Usage: {} PROBLEM-NUMBER N RB PLOT'.format(sys.argv[0]))

rhs0 = GenericFunction(lambda X: np.ones(X.shape[:-1]) * 10, dim_domain=2)
rhs1 = GenericFunction(lambda X: (X[..., 0] - 0.5) ** 2 * 1000, dim_domain=2)

nrhs = int(sys.argv[1])
assert 0 <= nrhs <= 1, ValueError('Invalid rhs number.')
rhs = eval('rhs{}'.format(nrhs))

n = int(sys.argv[2])
rbsize = int(sys.argv[3])
plot = bool(int(sys.argv[4]))

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

RB = np.empty((rbsize, discretization.operator.range_dim))
mu_snap = tuple(parameter_space.sample_uniformly(rbsize))
for i, mu in enumerate(mu_snap):
    print('Solving for mu = {} ...'.format(mu))
    RB[i] = discretization.solve(mu)

print('Projecting operators ...')

reductor = GenericRBReductor(discretization)
rb_discretization, reconstructor = reductor.reduce(RB)

l2_err_max = -1
for mu in parameter_space.sample_randomly(10):
    print('Solving RB-Scheme for mu = {} ... '.format(mu), end='')
    URB = reconstructor.reconstruct(rb_discretization.solve(mu))
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
if plot:
    discretization.visualize(U-URB)
