#!/usr/bin/env python
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''
Proof of concept for solving the poisson equation in 1D using linear finite elements and our grid interface
'''

from __future__ import absolute_import, division, print_function

import sys
import math as m

import numpy as np

from pymor.domaindescriptions import RectDomain
from pymor.analyticalproblems import EllipticProblem
from pymor.discretizers import discretize_elliptic_cg
from pymor.functions import GenericFunction
from pymor.parameters import CubicParameterSpace, ProjectionParameterFunctional, GenericParameterFunctional

if len(sys.argv) < 4:
    sys.exit('Usage: {} PROBLEM-NUMBER N PLOT'.format(sys.argv[0]))

rhs0 = GenericFunction(lambda X: np.ones(X.shape[:-1]) * 10, dim_domain=2)
rhs1 = GenericFunction(lambda X: (X[..., 0] - 0.5) ** 2 * 1000, dim_domain=2)

nrhs = int(sys.argv[1])
assert 0 <= nrhs <= 1, ValueError('Invalid rhs number.')
rhs = eval('rhs{}'.format(nrhs))

n = int(sys.argv[2])
plot = bool(int(sys.argv[3]))

d0 = GenericFunction(lambda X: 1 - X[..., 0], dim_domain=2)
d1 = GenericFunction(lambda X: X[..., 0], dim_domain=2)

parameter_space = CubicParameterSpace({'diffusionl': 1}, 0.1, 1)
f0 = ProjectionParameterFunctional(parameter_space, 'diffusionl')
f1 = GenericParameterFunctional(parameter_space, lambda mu: 1)

print('Solving on TriaGrid(({0},{0}))'.format(n))

print('Setup Problem ...')
problem = EllipticProblem(domain=RectDomain(), rhs=rhs, diffusion_functions=(d0, d1), diffusion_functionals=(f0, f1))

print('Discretize ...')
discretization, _ = discretize_elliptic_cg(problem, diameter=m.sqrt(2) / n)

print(discretization.parameter_info())

for mu in parameter_space.sample_uniformly(4):
    print('Solving for mu = {} ...'.format(mu))
    U = discretization.solve(mu)

    if plot:
        print('Plot ...')
        discretization.visualize(U)

    print('')
