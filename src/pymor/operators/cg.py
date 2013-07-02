# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from pymor.la import NumpyVectorArray
from pymor.grids.referenceelements import triangle, line
from pymor.operators.interfaces import LinearOperatorInterface
from pymor.operators.numpy import NumpyLinearOperator


class L2ProductFunctionalP1(LinearOperatorInterface):
    '''Scalar product with an L2-function for linear finite elements.

    It is moreover possible to specify a `BoundaryInfo` and a Dirichlet data function
    such that boundary DOFs are evaluated to the corresponding dirichlet values.
    This is useful for imposing boundary conditions on the solution.
    The integral is caculated by an order two Gauss quadrature.

    The current implementation works in one and two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        Grid over which to assemble the functional.
    function
        The `Function` with which to take the scalar product.
    boundary_info
        `BoundaryInfo` determining the Dirichlet boundaries or None.
    dirichlet_data
        The `Function` providing the Dirichlet boundary values. If None, zero boundary
        is assumed.
    name
        The name of the functional.
    '''

    type_source = type_range = NumpyVectorArray
    sparse = False

    def __init__(self, grid, function, boundary_info=None, dirichlet_data=None, name_map=None, name=None):
        assert grid.reference_element(0) in {line, triangle}
        assert function.dim_range == 1
        super(L2ProductFunctionalP1, self).__init__()
        self.dim_source = grid.size(grid.dim)
        self.dim_range = 1
        self.grid = grid
        self.boundary_info = boundary_info
        self.function = function
        self.dirichlet_data = dirichlet_data
        self.name = name
        self.build_parameter_type(inherits={'function': function, 'dirichlet_data': dirichlet_data}, name_map=name_map)
        self.lock()

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # evaluate function at all quadrature points -> shape = (g.size(0), number of quadrature points, 1)
        # the singleton dimension correspoints to the dimension of the range of the function
        F = self.function(g.quadrature_points(0, order=2), mu=self.map_parameter(mu, 'function'))

        # evaluate the shape functions at the quadrature points on the reference
        # element -> shape = (number of shape functions, number of quadrature points)
        q, w = g.reference_element.quadrature(order=2)
        if g.dim == 1:
            SF = np.squeeze(np.array((1 - q, q)))
        elif g.dim == 2:
            SF = np.array(((1 - np.sum(q, axis=-1)),
                          q[..., 0],
                          q[..., 1]))
        else:
            raise NotImplementedError

        # integrate the products of the function with the shape functions on each element
        # -> shape = (g.size(0), number of shape functions)
        SF_INTS = np.einsum('eix,pi,e,i->ep', F, SF, g.integration_elements(0), w).ravel()

        # map local DOFs to global DOFS
        # FIXME This implementation is horrible, find a better way!
        SF_I = g.subentities(0, g.dim).ravel()
        I = np.array(coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).todense()).ravel()

        # boundary treatment
        if bi is not None and bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(g.dim)
            if self.dirichlet_data is not None:
                I[DI] = self.dirichlet_data(g.centers(g.dim)[DI], self.map_parameter(mu, 'dirichlet_data'))
            else:
                I[DI] = 0

        return NumpyLinearOperator(I.reshape((1, -1)))


class L2ProductP1(LinearOperatorInterface):
    '''Operator representing the L2-product for linear finite functions.

    To evaluate the product use the apply2 method.

    The current implementation works in one and two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        The grid on which to assemble the product.
    boundary_info
        BoundaryInfo associating boundary types to boundary entities.
    dirichlet_clear_rows
        If True, set rows of the system matrix corresponding to Dirichlet boundary
        DOFs to zero. (Useful when used as mass matrix in time-stepping schemes.)
    dirichlet_clear_columns
        If True, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero (to obtain a symmetric matrix).
    dirichlet_clear_diag
        If True, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero (e.g. for affine decomposition). Otherwise, if either dirichlet_clear_rows or
        dirichlet_clear_columns is true, they are set to one.
    name
        The name of the product.
    '''

    type_source = type_range = NumpyVectorArray
    sparse = True

    def __init__(self, grid, boundary_info, dirichlet_clear_rows=True, dirichlet_clear_columns=False,
                 dirichlet_clear_diag=False, name=None):
        assert grid.reference_element in (line, triangle)
        super(L2ProductP1, self).__init__()
        self.dim_source = grid.size(grid.dim)
        self.dim_range = self.dim_source
        self.grid = grid
        self.boundary_info = boundary_info
        self.dirichlet_clear_rows = dirichlet_clear_rows
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.name = name
        self.lock()

    def _assemble(self, mu=None):
        assert mu is None
        g = self.grid
        bi = self.boundary_info

        # our shape functions
        if g.dim == 2:
            SF = [lambda X: 1 - X[..., 0] - X[..., 1],
                  lambda X: X[..., 0],
                  lambda X: X[..., 1]]
        elif g.dim == 1:
            SF = [lambda X: 1 - X[..., 0],
                  lambda X: X[..., 0]]
        else:
            raise NotImplementedError

        q, w = triangle.quadrature(order=2)

        # evaluate the shape functions on the quadrature points
        SFQ = np.array(tuple(f(q) for f in SF))

        self.logger.info('Integrate the products of the shape functions on each element')
        # -> shape = (g.size(0), number of shape functions ** 2)
        SF_INTS = np.einsum('iq,jq,q,e->eij', SFQ, SFQ, w, g.integration_elements(0)).ravel()

        self.logger.info('Determine global dofs ...')
        SF_I0 = np.repeat(g.subentities(0, g.dim), g.dim + 1, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, g.dim + 1]).ravel()

        self.logger.info('Boundary treatment ...')
        if bi.has_dirichlet:
            if self.dirichlet_clear_rows:
                SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, SF_INTS)
            if self.dirichlet_clear_columns:
                SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I1], 0, SF_INTS)
            if not self.dirichlet_clear_diag and (self.dirichlet_clear_rows or self.dirichlet_clear_columns):
                SF_INTS = np.hstack((SF_INTS, np.ones(bi.dirichlet_boundaries(g.dim).size)))
                SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(g.dim)))
                SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(g.dim)))

        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
        A = csr_matrix(A).copy()   # See DiffusionOperatorP1 for why copy() is necessary

        return NumpyLinearOperator(A)


class DiffusionOperatorP1(LinearOperatorInterface):
    '''Diffusion operator for linear finite elements.

    The operator is of the form ::

        (Lu)(x) = c ∇ ⋅ [ d(x) ∇ u(x) ]

    The current implementation works in one and two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        The grid on which to assemble the operator.
    boundary_info
        BoundaryInfo associating boundary types to boundary entities.
    diffusion_function
        The `Function` d(x).
    diffusion_constant
        The constant c.
    dirichlet_clear_columns
        If True, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero to obtain a symmetric system matrix. Otherwise, only the rows will
        be set to zero.
    dirichlet_clear_diag
        If True, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero (e.g. for affine decomposition). Otherwise they are set to one.
    name
        Name of the operator.
    '''

    type_source = type_range = NumpyVectorArray
    sparse = True

    def __init__(self, grid, boundary_info, diffusion_function=None, diffusion_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False, name_map=None, name=None):
        assert grid.reference_element(0) in {triangle, line}, ValueError('A simplicial grid is expected!')
        super(DiffusionOperatorP1, self).__init__()
        self.dim_source = self.dim_range = grid.size(grid.dim)
        self.grid = grid
        self.boundary_info = boundary_info
        self.diffusion_constant = diffusion_constant
        self.diffusion_function = diffusion_function
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.name = name
        if diffusion_function is not None:
            self.build_parameter_type(inherits={'diffusion': diffusion_function}, name_map=name_map)
        self.lock()

    def _assemble(self, mu=None):
        mu = self.parse_parameter(mu)
        g = self.grid
        bi = self.boundary_info

        # gradients of shape functions
        if g.dim == 2:
            SF_GRAD = np.array(([-1., -1.],
                                [1., 0.],
                                [0., 1.]))
        elif g.dim == 1:
            SF_GRAD = np.array(([-1.],
                                [1., ]))
        else:
            raise NotImplementedError

        self.logger.info('Calulate gradients of shape functions transformed by reference map ...')
        SF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), SF_GRAD)

        self.logger.info('Calculate all local scalar products beween gradients ...')
        if self.diffusion_function is not None:
            D = self.diffusion_function(self.grid.centers(0), mu=self.map_parameter(mu, 'diffusion')).ravel()
            SF_INTS = np.einsum('epi,eqi,e,e->epq', SF_GRADS, SF_GRADS, g.volumes(0), D).ravel()
        else:
            SF_INTS = np.einsum('epi,eqi,e->epq', SF_GRADS, SF_GRADS, g.volumes(0)).ravel()

        if self.diffusion_constant is not None:
            SF_INTS *= self.diffusion_constant

        self.logger.info('Determine global dofs ...')
        SF_I0 = np.repeat(g.subentities(0, g.dim), g.dim + 1, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, g.dim + 1]).ravel()

        self.logger.info('Boundary treatment ...')
        if bi.has_dirichlet:
            SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, SF_INTS)
            if self.dirichlet_clear_columns:
                SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I1], 0, SF_INTS)

            if not self.dirichlet_clear_diag:
                SF_INTS = np.hstack((SF_INTS, np.ones(bi.dirichlet_boundaries(g.dim).size)))
                SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(g.dim)))
                SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(g.dim)))

        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
        A = csr_matrix(A).copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        # from pymor.tools.memory import print_memory_usage
        # print_memory_usage('matrix: {0:5.1f}'.format((A.data.nbytes + A.indptr.nbytes + A.indices.nbytes)/1024**2))

        return NumpyLinearOperator(A)
