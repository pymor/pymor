from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

import pymor.core as core

from pymor.grid.referenceelements import triangle
from pymor.common.discreteoperator.interfaces import ILinearDiscreteOperator


class DiffusionOperatorP1D2(ILinearDiscreteOperator):
    '''Simple Diffusion Operator for linear finite elements in two dimensions on an triangular grid and
    constant diffusion coefficient of value 1. Add more functionality later ...
    '''

    def __init__(self, grid, boundary_info, clear_columns=True):
        assert grid.reference_element(0) == triangle, ValueError('A triangular grid is expected!')
        self.grid = grid
        self.boundary_info = boundary_info
        self.clear_columns = clear_columns

    def assemble(self, mu=np.array([])):
        assert mu.size == self.parameter_dim,\
         ValueError('Invalid parameter dimensions (was {}, expected {})'.format(mu.size, self.parameter_dim))

        g = self.grid
        bi = self.boundary_info

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
        if bi.has_dirichlet:
            SF_INTS = np.where(bi.dirichlet_mask(2)[SF_I0], 0, SF_INTS)
            if self.clear_columns:
                SF_INTS = np.where(bi.dirichlet_mask(2)[SF_I1], 0, SF_INTS)

        SF_INTS = np.hstack((SF_INTS, np.ones(bi.dirichlet_boundaries(2).size)))
        SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(2)))
        SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(2)))

        print('Assemble system matrix ...')
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(2), g.size(2)))
        A = csr_matrix(A)

        return A
