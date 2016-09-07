# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

""" This module provides some operators for continuous finite element discretizations."""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from pymor.functions.interfaces import FunctionInterface
from pymor.grids.referenceelements import triangle, line, square
from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class L2ProductFunctionalP1(NumpyMatrixBasedOperator):
    """Linear finite element |Functional| representing the inner product with an L2-|Function|.

    Boundary treatment can be performed by providing `boundary_info` and `dirichlet_data`,
    in which case the DOFs corresponding to Dirichlet boundaries are set to the values
    provided by `dirichlet_data`. Neumann boundaries are handled by providing a
    `neumann_data` function, Robin boundaries by providing a `robin_data` tuple.

    The current implementation works in one and two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    function
        The |Function| with which to take the inner product.
    boundary_info
        |BoundaryInfo| determining the Dirichlet and Neumann boundaries or `None`.
        If `None`, no boundary treatment is performed.
    dirichlet_data
        |Function| providing the Dirichlet boundary values. If `None`,
        constant-zero boundary is assumed.
    neumann_data
        |Function| providing the Neumann boundary values. If `None`,
        constant-zero is assumed.
    robin_data
        Tuple of two |Functions| providing the Robin parameter and boundary values, see
        :class:`RobinBoundaryOperator`.  If `None`, constant-zero for both functions is
        assumed.
    order
        Order of the Gauss quadrature to use for numerical integration.
    name
        The name of the functional.
    """

    sparse = False
    range = NumpyVectorSpace(1)

    def __init__(self, grid, function, boundary_info=None, dirichlet_data=None, neumann_data=None, robin_data=None,
                 order=2, solver_options=None, name=None):
        assert grid.reference_element(0) in {line, triangle}
        assert function.shape_range == ()
        self.source = NumpyVectorSpace(grid.size(grid.dim))
        self.grid = grid
        self.boundary_info = boundary_info
        self.function = function
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.robin_data = robin_data
        self.order = order
        self.solver_options = solver_options
        self.name = name
        self.build_parameter_type(inherits=(function, dirichlet_data, neumann_data))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # evaluate function at all quadrature points -> shape = (g.size(0), number of quadrature points)
        F = self.function(g.quadrature_points(0, order=self.order), mu=mu)

        # evaluate the shape functions at the quadrature points on the reference
        # element -> shape = (number of shape functions, number of quadrature points)
        q, w = g.reference_element.quadrature(order=self.order)
        if g.dim == 1:
            SF = np.array((1 - q[..., 0], q[..., 0]))
        elif g.dim == 2:
            SF = np.array(((1 - np.sum(q, axis=-1)),
                           q[..., 0],
                           q[..., 1]))
        else:
            raise NotImplementedError

        # integrate the products of the function with the shape functions on each element
        # -> shape = (g.size(0), number of shape functions)
        SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(0), w).ravel()

        # map local DOFs to global DOFs
        # FIXME This implementation is horrible, find a better way!
        SF_I = g.subentities(0, g.dim).ravel()
        I = coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).toarray().ravel()

        # neumann boundary treatment
        if bi is not None and bi.has_neumann and self.neumann_data is not None:
            NI = bi.neumann_boundaries(1)
            if g.dim == 1:
                I[NI] -= self.neumann_data(g.centers(1)[NI])
            else:
                F = -self.neumann_data(g.quadrature_points(1, order=self.order)[NI], mu=mu)
                q, w = line.quadrature(order=self.order)
                SF = np.squeeze(np.array([1 - q, q]))
                SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(1)[NI], w).ravel()
                SF_I = g.subentities(1, 2)[NI].ravel()
                I += coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).toarray().ravel()

        # robin boundary treatment
        if bi is not None and bi.has_robin and self.robin_data is not None:
            RI = bi.robin_boundaries(1)
            if g.dim == 1:
                xref = g.centers(1)[RI]
                I[RI] += (self.robin_data[0](xref) * self.robin_data[1](xref))
            else:
                xref = g.quadrature_points(1, order=self.order)[RI]
                F = (self.robin_data[0](xref, mu=mu) * self.robin_data[1](xref, mu=mu))
                q, w = line.quadrature(order=self.order)
                SF = np.squeeze(np.array([1 - q, q]))
                SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(1)[RI], w).ravel()
                SF_I = g.subentities(1, 2)[RI].ravel()
                I += coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).toarray().ravel()

        if bi is not None and bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(g.dim)
            if self.dirichlet_data is not None:
                I[DI] = self.dirichlet_data(g.centers(g.dim)[DI], mu=mu)
            else:
                I[DI] = 0

        return I.reshape((1, -1))


class L2ProductFunctionalQ1(NumpyMatrixBasedOperator):
    """Bilinear finite element |Functional| representing the inner product with an L2-|Function|.

    Boundary treatment can be performed by providing `boundary_info` and `dirichlet_data`,
    in which case the DOFs corresponding to Dirichlet boundaries are set to the values
    provided by `dirichlet_data`. Neumann boundaries are handled by providing a
    `neumann_data` function, Robin boundaries by providing a `robin_data` tuple.

    The current implementation works in two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    function
        The |Function| with which to take the inner product.
    boundary_info
        |BoundaryInfo| determining the Dirichlet boundaries or `None`.
        If `None`, no boundary treatment is performed.
    dirichlet_data
        |Function| providing the Dirichlet boundary values. If `None`,
        constant-zero boundary is assumed.
    neumann_data
        |Function| providing the Neumann boundary values. If `None`,
        constant-zero is assumed.
    robin_data
        Tuple of two |Functions| providing the Robin parameter and boundary values, see
        :class:`RobinBoundaryOperator`.  If `None`, constant-zero for both functions
        is assumed.
    order
        Order of the Gauss quadrature to use for numerical integration.
    name
        The name of the functional.
    """

    sparse = False
    range = NumpyVectorSpace(1)

    def __init__(self, grid, function, boundary_info=None, dirichlet_data=None, neumann_data=None, robin_data=None,
                 order=2, name=None):
        assert grid.reference_element(0) in {square}
        assert function.shape_range == ()
        self.source = NumpyVectorSpace(grid.size(grid.dim))
        self.grid = grid
        self.boundary_info = boundary_info
        self.function = function
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.robin_data = robin_data
        self.order = order
        self.name = name
        self.build_parameter_type(inherits=(function, dirichlet_data))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # evaluate function at all quadrature points -> shape = (g.size(0), number of quadrature points)
        F = self.function(g.quadrature_points(0, order=self.order), mu=mu)

        # evaluate the shape functions at the quadrature points on the reference
        # element -> shape = (number of shape functions, number of quadrature points)
        q, w = g.reference_element.quadrature(order=self.order)
        if g.dim == 2:
            SF = np.array(((1 - q[..., 0]) * (1 - q[..., 1]),
                           (1 - q[..., 1]) * (q[..., 0]),
                           (q[..., 0]) * (q[..., 1]),
                           (q[..., 1]) * (1 - q[..., 0])))
        else:
            raise NotImplementedError

        # integrate the products of the function with the shape functions on each element
        # -> shape = (g.size(0), number of shape functions)
        SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(0), w).ravel()

        # map local DOFs to global DOFs
        # FIXME This implementation is horrible, find a better way!
        SF_I = g.subentities(0, g.dim).ravel()
        I = coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).toarray().ravel()

        # neumann boundary treatment
        if bi is not None and bi.has_neumann and self.neumann_data is not None:
            NI = bi.neumann_boundaries(1)
            F = -self.neumann_data(g.quadrature_points(1, order=self.order)[NI], mu=mu)
            q, w = line.quadrature(order=self.order)
            SF = np.squeeze(np.array([1 - q, q]))
            SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(1)[NI], w).ravel()
            SF_I = g.subentities(1, 2)[NI].ravel()
            I += coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).toarray().ravel()

        if bi is not None and bi.has_robin and self.robin_data is not None:
            RI = bi.robin_boundaries(1)
            xref = g.quadrature_points(1, order=self.order)[RI]
            F = self.robin_data[0](xref, mu=mu) * self.robin_data[1](xref, mu=mu)
            q, w = line.quadrature(order=self.order)
            SF = np.squeeze(np.array([1 - q, q]))
            SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(1)[RI], w).ravel()
            SF_I = g.subentities(1, 2)[RI].ravel()
            I += coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).toarray().ravel()

        if bi is not None and bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(g.dim)
            if self.dirichlet_data is not None:
                I[DI] = self.dirichlet_data(g.centers(g.dim)[DI], mu=mu)
            else:
                I[DI] = 0

        return I.reshape((1, -1))


class L2ProductP1(NumpyMatrixBasedOperator):
    """|Operator| representing the L2-product between linear finite element functions.

    The current implementation works in one and two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the product.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    dirichlet_clear_rows
        If `True`, set the rows of the system matrix corresponding to Dirichlet boundary
        DOFs to zero.
    dirichlet_clear_columns
        If `True`, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero.
    dirichlet_clear_diag
        If `True`, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero. Otherwise, if either `dirichlet_clear_rows` or `dirichlet_clear_columns` is
        `True`, the diagonal entries are set to one.
    coefficient_function
        Coefficient |Function| for product with `shape_range == ()`.
        If `None`, constant one is assumed.
    name
        The name of the product.
    """

    sparse = True

    def __init__(self, grid, boundary_info, dirichlet_clear_rows=True, dirichlet_clear_columns=False,
                 dirichlet_clear_diag=False, coefficient_function=None, solver_options=None, name=None):
        assert grid.reference_element in (line, triangle)
        self.source = self.range = NumpyVectorSpace(grid.size(grid.dim))
        self.grid = grid
        self.boundary_info = boundary_info
        self.dirichlet_clear_rows = dirichlet_clear_rows
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.coefficient_function = coefficient_function
        self.solver_options = solver_options
        self.name = name

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # our shape functions
        if g.dim == 2:
            SF = [lambda X: 1 - X[..., 0] - X[..., 1],
                  lambda X: X[..., 0],
                  lambda X: X[..., 1]]
            q, w = triangle.quadrature(order=2)
        elif g.dim == 1:
            SF = [lambda X: 1 - X[..., 0],
                  lambda X: X[..., 0]]
            q, w = line.quadrature(order=2)
        else:
            raise NotImplementedError

        # evaluate the shape functions on the quadrature points
        SFQ = np.array(tuple(f(q) for f in SF))

        self.logger.info('Integrate the products of the shape functions on each element')
        # -> shape = (g.size(0), number of shape functions ** 2)
        if self.coefficient_function is not None:
            C = self.coefficient_function(self.grid.centers(0), mu=mu)
            SF_INTS = np.einsum('iq,jq,q,e,e->eij', SFQ, SFQ, w, g.integration_elements(0), C).ravel()
            del C
        else:
            SF_INTS = np.einsum('iq,jq,q,e->eij', SFQ, SFQ, w, g.integration_elements(0)).ravel()

        del SFQ

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
        del SF_INTS, SF_I0, SF_I1
        A = csc_matrix(A).copy()  # See DiffusionOperatorP1 for why copy() is necessary

        return A


class L2ProductQ1(NumpyMatrixBasedOperator):
    """|Operator| representing the L2-product between bilinear finite element functions.

    The current implementation works in two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the product.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    dirichlet_clear_rows
        If `True`, set the rows of the system matrix corresponding to Dirichlet boundary
        DOFs to zero.
    dirichlet_clear_columns
        If `True`, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero.
    dirichlet_clear_diag
        If `True`, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero. Otherwise, if either `dirichlet_clear_rows` or `dirichlet_clear_columns`
        is `True`, the diagonal entries are set to one.
    coefficient_function
        Coefficient |Function| for product with `shape_range == ()`.
        If `None`, constant one is assumed.
    name
        The name of the product.
    """

    sparse = True

    def __init__(self, grid, boundary_info, dirichlet_clear_rows=True, dirichlet_clear_columns=False,
                 dirichlet_clear_diag=False, coefficient_function=None, solver_options=None, name=None):
        assert grid.reference_element in {square}
        self.source = self.range = NumpyVectorSpace(grid.size(grid.dim))
        self.grid = grid
        self.boundary_info = boundary_info
        self.dirichlet_clear_rows = dirichlet_clear_rows
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.coefficient_function = coefficient_function
        self.solver_options = solver_options
        self.name = name

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # our shape functions
        if g.dim == 2:
            SF = [lambda X: (1 - X[..., 0]) * (1 - X[..., 1]),
                  lambda X: (1 - X[..., 1]) * (X[..., 0]),
                  lambda X: (X[..., 0]) * (X[..., 1]),
                  lambda X: (1 - X[..., 0]) * (X[..., 1])]
        else:
            raise NotImplementedError

        q, w = square.quadrature(order=2)

        # evaluate the shape functions on the quadrature points
        SFQ = np.array(tuple(f(q) for f in SF))

        self.logger.info('Integrate the products of the shape functions on each element')
        # -> shape = (g.size(0), number of shape functions ** 2)
        if self.coefficient_function is not None:
            C = self.coefficient_function(self.grid.quadrature_points(0, order=2), mu=mu)
            SF_INTS = np.einsum('iq,jq,q,e,eq->eij', SFQ, SFQ, w, g.integration_elements(0), C).ravel()
            del C
        else:
            SF_INTS = np.einsum('iq,jq,q,e->eij', SFQ, SFQ, w, g.integration_elements(0)).ravel()

        del SFQ

        self.logger.info('Determine global dofs ...')
        SF_I0 = np.repeat(g.subentities(0, g.dim), 4, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, 4]).ravel()

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
        del SF_INTS, SF_I0, SF_I1
        A = csc_matrix(A).copy()  # See DiffusionOperatorP1 for why copy() is necessary

        return A


class DiffusionOperatorP1(NumpyMatrixBasedOperator):
    """Diffusion |Operator| for linear finite elements.

    The operator is of the form ::

        (Lu)(x) = c ∇ ⋅ [ d(x) ∇ u(x) ]

    The function `d` can be scalar- or matrix-valued.
    The current implementation works in one and two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    diffusion_function
        The |Function| `d(x)` with `shape_range == ()` or
        `shape_range = (grid.dim_outer, grid.dim_outer)`. If `None`, constant one is
        assumed.
    diffusion_constant
        The constant `c`. If `None`, `c` is set to one.
    dirichlet_clear_columns
        If `True`, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero to obtain a symmetric system matrix. Otherwise, only the rows will
        be set to zero.
    dirichlet_clear_diag
        If `True`, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero. Otherwise they are set to one.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, diffusion_function=None, diffusion_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element(0) in {triangle, line}, 'A simplicial grid is expected!'
        assert diffusion_function is None \
            or (isinstance(diffusion_function, FunctionInterface) and
                diffusion_function.dim_domain == grid.dim_outer and
                diffusion_function.shape_range == () or diffusion_function.shape_range == (grid.dim_outer,) * 2)
        self.source = self.range = NumpyVectorSpace(grid.size(grid.dim))
        self.grid = grid
        self.boundary_info = boundary_info
        self.diffusion_constant = diffusion_constant
        self.diffusion_function = diffusion_function
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.solver_options = solver_options
        self.name = name
        if diffusion_function is not None:
            self.build_parameter_type(inherits=(diffusion_function,))

    def _assemble(self, mu=None):
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
        if self.diffusion_function is not None and self.diffusion_function.shape_range == ():
            D = self.diffusion_function(self.grid.centers(0), mu=mu)
            SF_INTS = np.einsum('epi,eqi,e,e->epq', SF_GRADS, SF_GRADS, g.volumes(0), D).ravel()
            del D
        elif self.diffusion_function is not None:
            D = self.diffusion_function(self.grid.centers(0), mu=mu)
            SF_INTS = np.einsum('epi,eqj,e,eij->epq', SF_GRADS, SF_GRADS, g.volumes(0), D).ravel()
            del D
        else:
            SF_INTS = np.einsum('epi,eqi,e->epq', SF_GRADS, SF_GRADS, g.volumes(0)).ravel()

        del SF_GRADS

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
        del SF_INTS, SF_I0, SF_I1
        A = csc_matrix(A).copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        return A


class DiffusionOperatorQ1(NumpyMatrixBasedOperator):
    """Diffusion |Operator| for bilinear finite elements.

    The operator is of the form ::

        (Lu)(x) = c ∇ ⋅ [ d(x) ∇ u(x) ]

    The function `d` can be scalar- or matrix-valued.
    The current implementation works in two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    diffusion_function
        The |Function| `d(x)` with `shape_range == ()` or
        `shape_range = (grid.dim_outer, grid.dim_outer)`. If `None`, constant one is
        assumed.
    diffusion_constant
        The constant `c`. If `None`, `c` is set to one.
    dirichlet_clear_columns
        If `True`, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero to obtain a symmetric system matrix. Otherwise, only the rows will
        be set to zero.
    dirichlet_clear_diag
        If `True`, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero. Otherwise they are set to one.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, diffusion_function=None, diffusion_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element(0) in {square}, 'A square grid is expected!'
        assert diffusion_function is None \
            or (isinstance(diffusion_function, FunctionInterface) and
                diffusion_function.dim_domain == grid.dim_outer and
                diffusion_function.shape_range == () or diffusion_function.shape_range == (grid.dim_outer,) * 2)
        self.source = self.range = NumpyVectorSpace(grid.size(grid.dim))
        self.grid = grid
        self.boundary_info = boundary_info
        self.diffusion_constant = diffusion_constant
        self.diffusion_function = diffusion_function
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.solver_options = solver_options
        self.name = name
        if diffusion_function is not None:
            self.build_parameter_type(inherits=(diffusion_function,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # gradients of shape functions
        if g.dim == 2:
            q, w = g.reference_element.quadrature(order=2)
            SF_GRAD = np.array(([q[..., 1] - 1., q[..., 0] - 1.],
                                [1. - q[..., 1], -q[..., 0]],
                                [q[..., 1], q[..., 0]],
                                [-q[..., 1], 1. - q[..., 0]]))
        else:
            raise NotImplementedError

        self.logger.info('Calulate gradients of shape functions transformed by reference map ...')
        SF_GRADS = np.einsum('eij,pjc->epic', g.jacobian_inverse_transposed(0), SF_GRAD)

        self.logger.info('Calculate all local scalar products beween gradients ...')
        if self.diffusion_function is not None and self.diffusion_function.shape_range == ():
            D = self.diffusion_function(self.grid.quadrature_points(0, order=2), mu=mu)
            SF_INTS = np.einsum('epic,eqic,c,e,ec->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0), D).ravel()
            del D
        elif self.diffusion_function is not None:
            D = self.diffusion_function(self.grid.quadrature_points(0, order=2), mu=mu)
            SF_INTS = np.einsum('epic,eqjc,c,e,ecij->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0), D).ravel()
            del D
        else:
            SF_INTS = np.einsum('epic,eqic,c,e->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0)).ravel()

        del SF_GRADS

        if self.diffusion_constant is not None:
            SF_INTS *= self.diffusion_constant

        self.logger.info('Determine global dofs ...')

        SF_I0 = np.repeat(g.subentities(0, g.dim), 4, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, 4]).ravel()

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
        del SF_INTS, SF_I0, SF_I1
        A = csc_matrix(A).copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        return A


class AdvectionOperatorP1(NumpyMatrixBasedOperator):
    """Linear advection |Operator| for linear finite elements.

    The operator is of the form ::

        (Lu)(x) = c ∇ ⋅ [ v(x) u(x) ]

    The function `v` is vector-valued.
    The current implementation works in one and two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    advection_function
        The |Function| `v(x)` with `shape_range = (grid.dim_outer, )`.
        If `None`, constant one is assumed.
    advection_constant
        The constant `c`. If `None`, `c` is set to one.
    dirichlet_clear_columns
        If `True`, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero to obtain a symmetric system matrix. Otherwise, only the rows will
        be set to zero.
    dirichlet_clear_diag
        If `True`, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero. Otherwise they are set to one.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, advection_function=None, advection_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element(0) in {triangle, line}, 'A simplicial grid is expected!'
        assert advection_function is None \
            or (isinstance(advection_function, FunctionInterface) and
                advection_function.dim_domain == grid.dim_outer and
                advection_function.shape_range == (grid.dim_outer,))
        self.source = self.range = NumpyVectorSpace(grid.size(grid.dim))
        self.grid = grid
        self.boundary_info = boundary_info
        self.advection_constant = advection_constant
        self.advection_function = advection_function
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.solver_options = solver_options
        self.name = name
        if advection_function is not None:
            self.build_parameter_type(inherits=(advection_function,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # gradients of shape functions
        if g.dim == 2:
            SF_GRAD = np.array(([-1., -1.],
                                [1., 0.],
                                [0., 1.]))
            SF = [lambda X: 1 - X[..., 0] - X[..., 1],
                  lambda X: X[..., 0],
                  lambda X: X[..., 1]]
            # SF_GRAD(function, component)
        else:
            raise NotImplementedError

        q, w = g.reference_element.quadrature(order=2)

        self.logger.info('Calulate gradients of shape functions transformed by reference map ...')
        SF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), SF_GRAD)
        # SF_GRADS(element, function, component)

        SFQ = np.array(tuple(f(q) for f in SF))
        # SFQ(function, quadraturepoint)

        self.logger.info('Calculate all local scalar products beween gradients ...')
        D = self.advection_function(self.grid.quadrature_points(0, order=2), mu=mu)
        SF_INTS = - np.einsum('pc,eqi,c,e,eci->eqp', SFQ, SF_GRADS, w, g.integration_elements(0), D).ravel()
        del D
        del SF_GRADS

        if self.advection_constant is not None:
            SF_INTS *= self.advection_constant

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
        del SF_INTS, SF_I0, SF_I1
        A = csc_matrix(A).copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        return A


class AdvectionOperatorQ1(NumpyMatrixBasedOperator):
    """Linear advection |Operator| for bilinear finite elements.

    The operator is of the form ::

        (Lu)(x) = c ∇ ⋅ [ v(x) u(x) ]

    The function `v` has to be vector-valued.
    The current implementation works in two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    advection_function
        The |Function| `v(x)` with `shape_range = (grid.dim_outer, )`.
        If `None`, constant one is assumed.
    advection_constant
        The constant `c`. If `None`, `c` is set to one.
    dirichlet_clear_columns
        If `True`, set columns of the system matrix corresponding to Dirichlet boundary
        DOFs to zero to obtain a symmetric system matrix. Otherwise, only the rows will
        be set to zero.
    dirichlet_clear_diag
        If `True`, also set diagonal entries corresponding to Dirichlet boundary DOFs to
        zero. Otherwise they are set to one.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, advection_function=None, advection_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element(0) in {square}, 'A square grid is expected!'
        assert advection_function is None \
            or (isinstance(advection_function, FunctionInterface) and
                advection_function.dim_domain == grid.dim_outer and
                advection_function.shape_range == (grid.dim_outer,))
        self.source = self.range = NumpyVectorSpace(grid.size(grid.dim))
        self.grid = grid
        self.boundary_info = boundary_info
        self.advection_constant = advection_constant
        self.advection_function = advection_function
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.solver_options = solver_options
        self.name = name
        if advection_function is not None:
            self.build_parameter_type(inherits=(advection_function,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # gradients of shape functions
        if g.dim == 2:
            q, w = g.reference_element.quadrature(order=2)
            SF_GRAD = np.array(([q[..., 1] - 1., q[..., 0] - 1.],
                                [1. - q[..., 1], -q[..., 0]],
                                [q[..., 1], q[..., 0]],
                                [-q[..., 1], 1. - q[..., 0]]))
            # SF_GRAD(function, component, quadraturepoint)
            SF = [lambda X: (1 - X[..., 0]) * (1 - X[..., 1]),
                  lambda X: (1 - X[..., 1]) * (X[..., 0]),
                  lambda X: (X[..., 0]) * (X[..., 1]),
                  lambda X: (1 - X[..., 0]) * (X[..., 1])]
        else:
            raise NotImplementedError

        self.logger.info('Calulate gradients of shape functions transformed by reference map ...')
        SF_GRADS = np.einsum('eij,pjc->epic', g.jacobian_inverse_transposed(0), SF_GRAD)
        # SF_GRADS(element,function,component,quadraturepoint)

        SFQ = np.array(tuple(f(q) for f in SF))
        # SFQ(function, quadraturepoint)

        self.logger.info('Calculate all local scalar products beween gradients ...')

        D = self.advection_function(self.grid.quadrature_points(0, order=2), mu=mu)
        SF_INTS = - np.einsum('pc,eqic,c,e,eci->eqp', SFQ, SF_GRADS, w, g.integration_elements(0), D).ravel()
        del D
        del SF_GRADS

        if self.advection_constant is not None:
            SF_INTS *= self.advection_constant

        self.logger.info('Determine global dofs ...')

        SF_I0 = np.repeat(g.subentities(0, g.dim), 4, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, 4]).ravel()

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
        del SF_INTS, SF_I0, SF_I1
        A = csc_matrix(A).copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        return A


class RobinBoundaryOperator(NumpyMatrixBasedOperator):
    """Robin boundary |Operator| for linear finite elements.

    The operator represents the contribution of Robin boundary conditions to the
    stiffness matrix, where the boundary condition is supposed to be given in the
    form ::

        -[ d(x) ∇u(x) ] ⋅ n(x) = c(x) (u(x) - g(x))

    `d` and `n` are the diffusion function (see :class:`DiffusionOperatorP1`) and
    the unit outer normal in `x`, while `c` is the (scalar) Robin parameter
    function and `g` is the (also scalar) Robin boundary value function.

    Parameters
    ----------
    grid
        The |Grid| over which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    robin_data
        Tuple providing two |Functions| that represent the Robin parameter and boundary
        value function. If `None`, the resulting operator is zero.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, robin_data=None, order=2, solver_options=None, name=None):
        assert robin_data is None or (isinstance(robin_data, tuple) and len(robin_data) == 2)
        assert robin_data is None or all([isinstance(f, FunctionInterface)
                                          and f.dim_domain == grid.dim_outer
                                          and (f.shape_range == ()
                                               or f.shape_range == (grid.dim_outer,)
                                               ) for f in robin_data])
        self.source = self.range = NumpyVectorSpace(grid.size(grid.dim))
        self.grid = grid
        self.boundary_info = boundary_info
        self.robin_data = robin_data
        self.solver_options = solver_options
        self.name = name
        self.order = order
        if self.robin_data is not None:
            self.build_parameter_type(inherits=(self.robin_data[0],))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        if g.dim > 2:
            raise NotImplementedError

        if bi is None or not bi.has_robin or self.robin_data is None:
            return coo_matrix((g.size(g.dim), g.size(g.dim))).tocsc()

        RI = bi.robin_boundaries(1)
        if g.dim == 1:
            robin_c = self.robin_data[0](g.centers(1)[RI], mu=mu)
            I = coo_matrix((robin_c, (RI, RI)), shape=(g.size(g.dim), g.size(g.dim)))
            return csc_matrix(I).copy()
        else:
            xref = g.quadrature_points(1, order=self.order)[RI]
            # xref(robin-index, quadraturepoint-index)
            if self.robin_data[0].shape_range == ():
                robin_c = self.robin_data[0](xref, mu=mu)
            else:
                robin_elements = g.superentities(1, 0)[RI, 0]
                robin_indices = g.superentity_indices(1, 0)[RI, 0]
                normals = g.unit_outer_normals()[robin_elements, robin_indices]
                robin_values = self.robin_data[0](xref, mu=mu)
                robin_c = np.einsum('ei,eqi->eq', normals, robin_values)

            # robin_c(robin-index, quadraturepoint-index)
            q, w = line.quadrature(order=self.order)
            SF = np.squeeze(np.array([1 - q, q]))
            SF_INTS = np.einsum('ep,pi,pj,e,p->eij', robin_c, SF, SF, g.integration_elements(1)[RI], w).ravel()
            SF_I0 = np.repeat(g.subentities(1, g.dim)[RI], 2).ravel()
            SF_I1 = np.tile(g.subentities(1, g.dim)[RI], [1, 2]).ravel()
            I = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
            return csc_matrix(I).copy()


class InterpolationOperator(NumpyMatrixBasedOperator):
    """Vector-like Lagrange interpolation |Operator| for continuous finite element spaces.

    Parameters
    ----------
    grid
        The |Grid| on which to interpolate.
    function
        The |Function| to interpolate.
    """

    source = NumpyVectorSpace(1)
    linear = True

    def __init__(self, grid, function):
        assert function.dim_domain == grid.dim_outer
        assert function.shape_range == ()
        self.grid = grid
        self.function = function
        self.range = NumpyVectorSpace(grid.size(grid.dim))
        self.build_parameter_type(inherits=(function,))

    def _assemble(self, mu=None):
        return self.function.evaluate(self.grid.centers(self.grid.dim), mu=mu).reshape((-1, 1))
