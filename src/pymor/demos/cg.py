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

    print('Calculate grid data ...')
    g.subentities(0, 2)
    g.centers(2)
    g.jacobian_inverse_transposed(0)
    g.volumes(0)

    print('Determine boundary ...')
    II = np.all((0 < g.centers(2)[:, 0],
                 g.centers(2)[:, 0] < 1,
                 0 < g.centers(2)[:, 1],
                 g.centers(2)[:, 1] < 1), axis=0)
    IB = np.where(1 - II)[0]

    print('Assemble right hand side ...')
    F = rhs(g.centers(2)) * g.integration_element(0)[0]

    # shape functions are given as
    # phi_0(x0, x1) = 1 - x0 - x1
    # phi_1(x0, x1) = x0
    # phi_2(x0, x1) = x1

    # gradients of shape functions
    SF_GRAD = np.array(([-1., -1.],
                        [1., 0.],
                        [0., 1.]))

    print('Calulate gradients of shape functions transformed by reference map ...')
    SF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), SF_GRAD)

    print('Calculate all local scalar products beween gradients ...')
    SF_INTS = np.einsum('epi,eqi,e->epq', SF_GRADS, SF_GRADS, g.volumes(0)).ravel()

    print('Determine global dofs ...')
    SF_I0 = np.repeat(g.subentities(0, 2), 3, axis=1).ravel()
    SF_I1 = np.tile(g.subentities(0, 2), [1, 3]).ravel()

    print('Boundary treatment ...')
    SF_INTS = SF_INTS * II[SF_I0] * II[SF_I1]
    F[IB] = 0
    SF_INTS = np.hstack((SF_INTS, np.ones(IB.size)))
    SF_I0 = np.hstack((SF_I0, IB))
    SF_I1 = np.hstack((SF_I1, IB))

    print('Assemble system matrix ...')
    A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(2), g.size(2)))
    A = csr_matrix(A)

    print('Solve ...')
    U, info = bicg(A, F)

    print('Plot ...')
    pl.tripcolor(g.centers(2)[:, 0], g.centers(2)[:, 1], g.subentities(0, 2), U)
    pl.colorbar()
    pl.show()

    print('')
