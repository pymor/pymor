#!/usr/bin/env python
# Proof of concept for solving the poisson equation in 2D using linear finite elements
# and our grid interface

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import numpy as np
import matplotlib.pyplot as pl
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import bicg

from pymor.grid.tria import Tria
from pymor.common.boundaryinfo import AllDirichletZero
from pymor.common.discreteoperator.cg import DiffusionOperatorP1D2

if len(sys.argv) < 2:
    sys.exit('Usage: %s PROBLEM-NUMBER'.format(sys.argv[0]))

nrhs = int(sys.argv[1])


def rhs0(x):
    return np.ones(x.shape[0]) * 10


def rhs1(x):
    return (x[:, 0] - 0.5) ** 2 * 1000

if nrhs == 0:
    rhs = rhs0
elif nrhs == 1:
    rhs = rhs1
else:
    raise ValueError('Invalid problem number!')

for n in [32, 64, 128, 256]:
    print('Solving on Tria(({0},{0}))'.format(n))

    print('Generate grid ...')
    g = Tria((n, n))

    # the following calls are only here to seperate all
    # grid calculations (which are currentily terribly slow)
    # from the rest of the work
    print('Calculate grid data ...')
    g.subentities(0, 2)
    g.centers(2)
    g.jacobian_inverse_transposed(0)
    g.volumes(0)
    g.boundaries(2)
    g.boundary_mask(2)

    bi = AllDirichletZero(g)

    print('Assemble right hand side ...')
    F = rhs(g.centers(2)) * g.integration_element(0)[0]
    F[bi.dirichlet_boundaries(2)] = 0

    print('Assemble system matrix ...')
    L = DiffusionOperatorP1D2(g, bi)
    A = L.matrix()

    print('Solve ...')
    U, info = bicg(A, F)

    print('Plot ...')
    pl.tripcolor(g.centers(2)[:, 0], g.centers(2)[:, 1], g.subentities(0, 2), U)
    pl.colorbar()
    pl.show()

    print('')
