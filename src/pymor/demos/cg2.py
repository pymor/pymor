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
from pymor.parameters import ProjectionFunctional, GenericFunctional
from collections import OrderedDict

if len(sys.argv) < 4:
    sys.exit('Usage: {} PROBLEM-NUMBER N PLOT'.format(sys.argv[0]))

rhs0 = GenericFunction(lambda X: np.ones(X.shape[0]) * 10, dim_domain=2)
rhs1 = GenericFunction(lambda X: (X[:, 0] - 0.5) ** 2 * 1000, dim_domain=2)

nrhs = int(sys.argv[1])
assert 0 <= nrhs <= 1, ValueError('Invalid rhs number.')
rhs = eval('rhs{}'.format(nrhs))

n = int(sys.argv[2])
plot = bool(int(sys.argv[3]))

d0 = GenericFunction(lambda X: 1 - X[:, 0], dim_domain=2)
d1 = GenericFunction(lambda X: X[:, 0], dim_domain=2)

pspace = OrderedDict((('diffusionl',1),))
f0 = ProjectionFunctional(pspace, 'diffusionl')
f1 = GenericFunctional(pspace, lambda mu:1)

print('Solving on TriaGrid(({0},{0}))'.format(n))

print('Setup Problem ...')
problem = PoissonProblem(domain=RectDomain(), rhs=rhs, diffusion_functions=(d0, d1), diffusion_functionals=(f0, f1))

print('Discretize ...')
discretizer = PoissonCGDiscretizer(problem)
discretization = discretizer.discretize(diameter=m.sqrt(2) / n)

print(discretization.parameter_info())

for d in [1, 0.5, 0.25, 0.125]:
    print('Solve ...')
    U = discretization.solve(d)

    if plot:
        print('Plot ...')
        discretization.visualize(U)

    print('')
