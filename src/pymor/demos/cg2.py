#!/usr/bin/env python
# Proof of concept for solving the poisson equation in 2D using linear finite elements
# and our grid interface

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import math as m

import numpy as np

from pymor.common import BoundaryType
from pymor.common.domaindescription import Rect as DRect
from pymor.analyticalproblem import Poisson
from pymor import discretizer

if len(sys.argv) < 2:
    sys.exit('Usage: %s PROBLEM-NUMBER'.format(sys.argv[0]))

rhs0 = lambda X: np.ones(X.shape[0]) * 10
rhs1 = lambda X: (X[:, 0] - 0.5) ** 2 * 1000

nrhs = int(sys.argv[1])
assert 0 <= nrhs <= 1, ValueError('Invalid rhs number.')
rhs = eval('rhs{}'.format(nrhs))

d1 = lambda X: X[:, 0]
d2 = lambda X: 1 - X[:, 0]

n = 256

print('Solving on Tria(({0},{0}))'.format(n))

print('Setup Problem ...')
aproblem = Poisson(domain=DRect(), rhs=rhs, diffusion_functions=(d1, d2))

print('Discretize ...')
discrt = discretizer.PoissonCG(diameter=m.sqrt(2) / n)
discretization = discrt.discretize(aproblem)

for d in [1, 0.5, 0.25, 0.125]:
    print('Solve ...')
    U = discretization.solve(np.array((1, d)))

    print('Plot ...')
    discretization.visualize(U)

    print('')
