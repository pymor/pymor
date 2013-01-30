from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

import pymor.core as core

from pymor.grids.referenceelements import triangle, line
from pymor.discreteoperators.interfaces import LinearDiscreteOperatorInterface


class L2ProductFunctionalP1D2(LinearDiscreteOperatorInterface):
    '''Scalar product with an L2-function for linear finite elements in two dimensions on a triangular
    grid. The integral is simply calculated by evaluating the function on the codim-2 entities.
    '''

    def __init__(self, grid, boundary_info, function, dirichlet_data=None, name=None):
        assert grid.reference_element(0) == triangle, ValueError('A triangular grid is expected!')
        self.source_dim = grid.size(2)
        self.range_dim = 1
        self.grid = grid
        self.boundary_info = boundary_info
        self.function = function
        self.dirichlet_data = dirichlet_data
        self.name = name
        self.set_parameter_type(inherits={'function':function, 'dirichlet_data':dirichlet_data})

    def assemble(self, mu={}):
        g = self.grid
        bi = self.boundary_info

        F = np.squeeze(self.function(g.centers(2), mu=self.map_parameter(mu, 'function')))
        SE = g.superentities(2, 0)
        V = np.sum(np.where(SE == -1, 0, g.volumes(0)[SE]), axis=1)

        I = F * (1 / 3) * V

        if bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(2)
            if self.dirichlet_data is not None:
                I[DI] = self.dirichlet_data(g.centers(2)[DI], self.map_parameter(mu, 'dirichlet_data'))
            else:
                I[DI] = 0

        return I


class L2ProductFunctionalP1D1(LinearDiscreteOperatorInterface):
    '''Scalar product with an L2-function for linear finite elements in one dimension.
    The integral is caculated with a two point Gauss quadrature.
    '''

    def __init__(self, grid, boundary_info, function, dirichlet_data=None, name=None):
        assert grid.reference_element(0) == line
        assert function.dim_range == 1
        self.source_dim = grid.size(1)
        self.range_dim = 1
        self.grid = grid
        self.boundary_info = boundary_info
        self.function = function
        self.dirichlet_data = dirichlet_data
        self.name = name
        self.set_parameter_type(inherits={'function':function, 'dirichlet_data':dirichlet_data}) 

    def assemble(self, mu={}):
        g = self.grid
        bi = self.boundary_info

        # evaluate function at all quadrature points -> shape = (g.size(0), 2, 1)
        # the singleton dimension correspoints to the dimension of the range of the function
        F = self.function(g.quadrature_points(0, order=2), mu=self.map_parameter(mu, 'function'))

        # evaluate the shape functions (1-id) and id at the quadrature points on the reference
        # element -> shape = (number of shape functions == 2, 2, 1)
        q, w = line.quadrature(order=2)
        SF = np.array((1-q, q))

        # integrate the products of the function with the shape functions on each element
        # -> shape = (g.size(0), number of shape functions == 2)
        SF_INTS = np.einsum('eix,piy,e,i->ep', F, SF, g.integration_elements(0), w)

        # map local DOFs to global DOFS
        SE = g.subentities(0,1)
        I = np.zeros(g.size(1))
        I[SE[:, 0]] = SF_INTS[:, 0]
        I[SE[:, 1]] += SF_INTS[:, 1]

        # note that the same algorithm also works for a larger number of quadrature points

        if bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(1)
            if self.dirichlet_data is not None:
                I[DI] = self.dirichlet_data(g.centers(1)[DI], self.map_parameter(mu, 'dirichlet_data'))
            else:
                I[DI] = 0

        return I


class DiffusionOperatorP1D2(LinearDiscreteOperatorInterface):
    '''Simple Diffusion Operator for linear finite elements in two dimensions on an triangular grid. 
    Add more functionality later ...
    '''

    def __init__(self, grid, boundary_info, diffusion_function=None, diffusion_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False, name=None):
        assert grid.reference_element(0) == triangle, ValueError('A triangular grid is expected!')
        self.source_dim = self.range_dim = grid.size(2)
        self.grid = grid
        self.boundary_info = boundary_info
        self.diffusion_constant = diffusion_constant
        self.diffusion_function = diffusion_function
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.name = name
        if diffusion_function is not None:
            self.set_parameter_type(inherits={'diffusion':diffusion_function})

    def assemble(self, mu={}):
        self.map_parameter(mu)
        g = self.grid
        bi = self.boundary_info

        # gradients of shape functions
        SF_GRAD = np.array(([-1., -1.],
                            [1., 0.],
                            [0., 1.]))

        print('Calulate gradients of shape functions transformed by reference map ...')
        SF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), SF_GRAD)

        print('Calculate all local scalar products beween gradients ...')
        if self.diffusion_function is not None:
            D = self.diffusion_function(self.grid.centers(0), mu=self.map_parameter(mu, 'diffusion')).ravel()
            SF_INTS = np.einsum('epi,eqi,e,e->epq', SF_GRADS, SF_GRADS, g.volumes(0), D).ravel()
        else:
            SF_INTS = np.einsum('epi,eqi,e->epq', SF_GRADS, SF_GRADS, g.volumes(0)).ravel()

        if self.diffusion_constant is not None:
            SF_INTS *= self.diffusion_constant

        print('Determine global dofs ...')
        SF_I0 = np.repeat(g.subentities(0, 2), 3, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, 2), [1, 3]).ravel()

        print('Boundary treatment ...')
        if bi.has_dirichlet:
            SF_INTS = np.where(bi.dirichlet_mask(2)[SF_I0], 0, SF_INTS)
            if self.dirichlet_clear_columns:
                SF_INTS = np.where(bi.dirichlet_mask(2)[SF_I1], 0, SF_INTS)

        if not self.dirichlet_clear_diag:
            SF_INTS = np.hstack((SF_INTS, np.ones(bi.dirichlet_boundaries(2).size)))
            SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(2)))
            SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(2)))


        print('Assemble system matrix ...')
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(2), g.size(2)))
        A = csr_matrix(A)

        return A


# TODO make a DiffusionOperatorP1 ...
class DiffusionOperatorP1D1(LinearDiscreteOperatorInterface):
    '''Simple Diffusion Operator for linear finite elements in one dimension.'''

    def __init__(self, grid, boundary_info, diffusion_function=None, diffusion_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False, name=None):
        assert grid.reference_element(0) == line
        self.source_dim = self.range_dim = grid.size(1)
        self.grid = grid
        self.boundary_info = boundary_info
        self.diffusion_constant = diffusion_constant
        self.diffusion_function = diffusion_function
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.name = name
        if diffusion_function is not None:
            self.set_parameter_type(inherits={'diffusion':diffusion_function})

    def assemble(self, mu={}):
        self.map_parameter(mu)
        g = self.grid
        bi = self.boundary_info

        # gradients of shape functions
        SF_GRAD = np.array(([-1.],
                            [1.,]))

        print('Calulate gradients of shape functions transformed by reference map ...')
        SF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), SF_GRAD)

        print('Calculate all local scalar products beween gradients ...')
        if self.diffusion_function is not None:
            D = self.diffusion_function(self.grid.centers(0), mu=self.map_parameter(mu, 'diffusion')).ravel()
            SF_INTS = np.einsum('epi,eqi,e,e->epq', SF_GRADS, SF_GRADS, g.volumes(0), D).ravel()
        else:
            SF_INTS = np.einsum('epi,eqi,e->epq', SF_GRADS, SF_GRADS, g.volumes(0)).ravel()

        if self.diffusion_constant is not None:
            SF_INTS *= self.diffusion_constant

        print('Determine global dofs ...')
        SF_I0 = np.repeat(g.subentities(0, 1), 2, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, 1), [1, 2]).ravel()

        print('Boundary treatment ...')
        if bi.has_dirichlet:
            SF_INTS = np.where(bi.dirichlet_mask(1)[SF_I0], 0, SF_INTS)
            if self.dirichlet_clear_columns:
                SF_INTS = np.where(bi.dirichlet_mask(1)[SF_I1], 0, SF_INTS)

        if not self.dirichlet_clear_diag:
            SF_INTS = np.hstack((SF_INTS, np.ones(bi.dirichlet_boundaries(1).size)))
            SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(1)))
            SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(1)))


        print('Assemble system matrix ...')
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(1), g.size(1)))
        A = csr_matrix(A)

        return A
