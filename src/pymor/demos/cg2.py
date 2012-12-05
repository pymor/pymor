#!/usr/bin/env python
# Proof of concept for solving the poisson equation in 2D using linear finite elements
# and our grid interface

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import numpy as np
import matplotlib.pyplot as pl
from scipy.sparse.linalg import bicg

from pymor.grid.tria import Tria
from pymor.common.boundaryinfo import AllDirichletZero
from pymor.common.discreteoperator.cg import DiffusionOperatorP1D2, L2ProductFunctionalP1D2
from pymor.common.discreteoperator.affine import LinearAffinelyDecomposedDOP

if len(sys.argv) < 2:
    sys.exit('Usage: %s PROBLEM-NUMBER'.format(sys.argv[0]))

rhs0 = lambda X: np.ones(X.shape[0]) * 10
rhs1 = lambda X: (X[:, 0] - 0.5) ** 2 * 1000

nrhs = int(sys.argv[1])
assert 0 <= nrhs <= 1, ValueError('Invalid rhs number.')
rhs = eval('rhs{}'.format(nrhs))
n = 256

print('Solving on Tria(({0},{0}))'.format(n))

print('Setup grid ...')
g = Tria((n, n))
bi = AllDirichletZero(g)

d1 = lambda X: X[:, 0]
d2 = lambda X: 1 - X[:, 0]

print('Assemble operators ...')
F = L2ProductFunctionalP1D2(g, bi, rhs)
L0 = DiffusionOperatorP1D2(g, bi, diffusion_constant=0)
L1 = DiffusionOperatorP1D2(g, bi, diffusion_function=d1, dirichlet_clear_diag=True)
L2 = DiffusionOperatorP1D2(g, bi, diffusion_function=d2, dirichlet_clear_diag=True)
L = LinearAffinelyDecomposedDOP((L1, L2), L0)

RHS = F.matrix()
L.matrix(mu=np.array((0, 0)))

for d in [1, 0.5, 0.25, 0.125]:

    print('Assemble system matrix ...')
    A = L.matrix(mu=np.array((1, d)))

    print('Solve ...')
    U, info = bicg(A, RHS)

    print('Plot ...')
    pl.tripcolor(g.centers(2)[:, 0], g.centers(2)[:, 1], g.subentities(0, 2), U)
    pl.colorbar()
    pl.show()

    print('')
