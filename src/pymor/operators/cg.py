# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

""" This module provides some operators for continuous finite element discretizations."""

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from pymor.functions.interfaces import FunctionInterface
from pymor.grids.referenceelements import triangle, line, square
from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


def CGVectorSpace(grid, id='STATE'):
    return NumpyVectorSpace(grid.size(grid.dim), id)


class L2ProductFunctionalP1(NumpyMatrixBasedOperator):
    """Linear finite element functional representing the inner product with an L2-|Function|.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    function
        The |Function| with which to take the inner product.
    dirichlet_clear_dofs
        If `True`, set dirichlet boundary DOFs to zero.
    boundary_info
        |BoundaryInfo| determining the Dirichlet boundaries in case
        `dirichlet_clear_dofs` is set to `True`.
    name
        The name of the functional.
    """

    sparse = False
    source = NumpyVectorSpace(1)

    def __init__(self, grid, function, dirichlet_clear_dofs=False, boundary_info=None, name=None):
        assert grid.reference_element(0) in {line, triangle}
        assert function.shape_range == ()
        assert not dirichlet_clear_dofs or boundary_info
        self.__auto_init(locals())
        self.range = CGVectorSpace(grid)
        self.build_parameter_type(function)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # evaluate function at element centers
        F = self.function(g.centers(0), mu=mu)

        # evaluate the shape functions at the quadrature points on the reference
        # element -> shape = (number of shape functions, number of quadrature points)
        q, w = g.reference_element.quadrature(order=1)
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
        SF_INTS = np.einsum('e,pi,e,i->ep', F, SF, g.integration_elements(0), w).ravel()

        # map local DOFs to global DOFs
        # FIXME This implementation is horrible, find a better way!
        SF_I = g.subentities(0, g.dim).ravel()
        I = coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).toarray().ravel()

        if self.dirichlet_clear_dofs and bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(g.dim)
            I[DI] = 0

        return I.reshape((-1, 1))


class BoundaryL2ProductFunctional(NumpyMatrixBasedOperator):
    """Linear finite element functional representing the inner product with an L2-|Function| on the boundary.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    function
        The |Function| with which to take the inner product.
    boundary_type
        The type of domain boundary (e.g. 'neumann') on which to assemble the functional.
        If `None` the functional is assembled over the whole boundary.
    dirichlet_clear_dofs
        If `True`, set dirichlet boundary DOFs to zero.
    boundary_info
        If `boundary_type` is specified or `dirichlet_clear_dofs` is `True`, the
        |BoundaryInfo| determining which boundary entity belongs to which physical boundary.
    name
        The name of the functional.
    """

    sparse = False
    source = NumpyVectorSpace(1)

    def __init__(self, grid, function, boundary_type=None, dirichlet_clear_dofs=False, boundary_info=None, name=None):
        assert grid.reference_element(0) in {line, triangle, square}
        assert function.shape_range == ()
        assert not (boundary_type or dirichlet_clear_dofs) or boundary_info
        self.__auto_init(locals())
        self.range = CGVectorSpace(grid)
        self.build_parameter_type(function)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        NI = bi.boundaries(self.boundary_type, 1) if self.boundary_type else g.boundaries(1)
        if g.dim == 1:
            I = np.zeros(self.range.dim)
            I[NI] = self.function(g.centers(1)[NI])
        else:
            F = self.function(g.centers(1)[NI], mu=mu)
            q, w = line.quadrature(order=1)
            # remove last dimension of q, as line coordinates are one dimensional
            q = q[:, 0]
            SF = np.array([1 - q, q])
            SF_INTS = np.einsum('e,pi,e,i->ep', F, SF, g.integration_elements(1)[NI], w).ravel()
            SF_I = g.subentities(1, 2)[NI].ravel()
            I = coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).toarray().ravel()

        if self.dirichlet_clear_dofs and bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(g.dim)
            I[DI] = 0

        return I.reshape((-1, 1))


class BoundaryDirichletFunctional(NumpyMatrixBasedOperator):
    """Linear finite element functional for enforcing Dirichlet boundary values.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    dirichlet_data
        |Function| providing the Dirichlet boundary values.
    boundary_info
        |BoundaryInfo| determining the Dirichlet boundaries.
    name
        The name of the functional.
    """

    sparse = False
    source = NumpyVectorSpace(1)

    def __init__(self, grid, dirichlet_data, boundary_info, name=None):
        assert grid.reference_element(0) in {line, triangle, square}
        self.__auto_init(locals())
        self.range = CGVectorSpace(grid)
        self.build_parameter_type(dirichlet_data)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        I = np.zeros(self.range.dim)
        DI = bi.dirichlet_boundaries(g.dim)
        I[DI] = self.dirichlet_data(g.centers(g.dim)[DI], mu=mu)

        return I.reshape((-1, 1))


class L2ProductFunctionalQ1(NumpyMatrixBasedOperator):
    """Bilinear finite element functional representing the inner product with an L2-|Function|.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    function
        The |Function| with which to take the inner product.
    dirichlet_clear_dofs
        If `True`, set dirichlet boundary DOFs to zero.
    boundary_info
        |BoundaryInfo| determining the Dirichlet boundaries in case
        `dirichlet_clear_dofs` is set to `True`.
    name
        The name of the functional.
    """

    sparse = False
    source = NumpyVectorSpace(1)

    def __init__(self, grid, function, dirichlet_clear_dofs=False, boundary_info=None, name=None):
        assert grid.reference_element(0) in {square}
        assert function.shape_range == ()
        assert not dirichlet_clear_dofs or boundary_info
        self.__auto_init(locals())
        self.range = CGVectorSpace(grid)
        self.build_parameter_type(function)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # evaluate function at all quadrature points -> shape = (g.size(0), number of quadrature points)
        F = self.function(g.centers(0), mu=mu)

        # evaluate the shape functions at the quadrature points on the reference
        # element -> shape = (number of shape functions, number of quadrature points)
        q, w = g.reference_element.quadrature(order=1)
        if g.dim == 2:
            SF = np.array(((1 - q[..., 0]) * (1 - q[..., 1]),
                           (1 - q[..., 1]) * (q[..., 0]),
                           (q[..., 0]) * (q[..., 1]),
                           (q[..., 1]) * (1 - q[..., 0])))
        else:
            raise NotImplementedError

        # integrate the products of the function with the shape functions on each element
        # -> shape = (g.size(0), number of shape functions)
        SF_INTS = np.einsum('e,pi,e,i->ep', F, SF, g.integration_elements(0), w).ravel()

        # map local DOFs to global DOFs
        # FIXME This implementation is horrible, find a better way!
        SF_I = g.subentities(0, g.dim).ravel()
        I = coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).toarray().ravel()

        if self.dirichlet_clear_dofs and bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(g.dim)
            I[DI] = 0

        return I.reshape((-1, 1))


class L2ProductP1(NumpyMatrixBasedOperator):
    """|Operator| representing the L2-product between linear finite element functions.

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
    solver_options
        The |solver_options| for the operator.
    name
        The name of the product.
    """

    sparse = True

    def __init__(self, grid, boundary_info, dirichlet_clear_rows=True, dirichlet_clear_columns=False,
                 dirichlet_clear_diag=False, coefficient_function=None, solver_options=None, name=None):
        assert grid.reference_element in (line, triangle)
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid)
        self.build_parameter_type(coefficient_function)

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
    solver_options
        The |solver_options| for the operator.
    name
        The name of the product.
    """

    sparse = True

    def __init__(self, grid, boundary_info, dirichlet_clear_rows=True, dirichlet_clear_columns=False,
                 dirichlet_clear_diag=False, coefficient_function=None, solver_options=None, name=None):
        assert grid.reference_element in {square}
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid)
        self.build_parameter_type(coefficient_function)

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
            C = self.coefficient_function(self.grid.centers(0), mu=mu)
            SF_INTS = np.einsum('iq,jq,q,e,e->eij', SFQ, SFQ, w, g.integration_elements(0), C).ravel()
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

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    diffusion_function
        The |Function| `d(x)` with `shape_range == ()` or
        `shape_range = (grid.dim, grid.dim)`. If `None`, constant one is
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
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, diffusion_function=None, diffusion_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element(0) in {triangle, line}, 'A simplicial grid is expected!'
        assert diffusion_function is None \
            or (isinstance(diffusion_function, FunctionInterface)
                and diffusion_function.dim_domain == grid.dim
                and diffusion_function.shape_range == ()
                or diffusion_function.shape_range == (grid.dim,) * 2)
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid)
        if diffusion_function is not None:
            self.build_parameter_type(diffusion_function)

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

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    diffusion_function
        The |Function| `d(x)` with `shape_range == ()` or
        `shape_range = (grid.dim, grid.dim)`. If `None`, constant one is
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
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, diffusion_function=None, diffusion_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element(0) in {square}, 'A square grid is expected!'
        assert diffusion_function is None \
            or (isinstance(diffusion_function, FunctionInterface)
                and diffusion_function.dim_domain == grid.dim
                and diffusion_function.shape_range == ()
                or diffusion_function.shape_range == (grid.dim,) * 2)
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid)
        if diffusion_function is not None:
            self.build_parameter_type(diffusion_function)

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
            D = self.diffusion_function(self.grid.centers(0), mu=mu)
            SF_INTS = np.einsum('epic,eqic,c,e,e->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0), D).ravel()
            del D
        elif self.diffusion_function is not None:
            D = self.diffusion_function(self.grid.centers(0), mu=mu)
            SF_INTS = np.einsum('epic,eqjc,c,e,eij->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0), D).ravel()
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

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    advection_function
        The |Function| `v(x)` with `shape_range = (grid.dim, )`.
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
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, advection_function=None, advection_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element(0) in {triangle, line}, 'A simplicial grid is expected!'
        assert advection_function is None \
            or (isinstance(advection_function, FunctionInterface)
                and advection_function.dim_domain == grid.dim
                and advection_function.shape_range == (grid.dim,))
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid)
        if advection_function is not None:
            self.build_parameter_type(advection_function)

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

        q, w = g.reference_element.quadrature(order=1)

        self.logger.info('Calulate gradients of shape functions transformed by reference map ...')
        SF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), SF_GRAD)
        # SF_GRADS(element, function, component)

        SFQ = np.array(tuple(f(q) for f in SF))
        # SFQ(function, quadraturepoint)

        self.logger.info('Calculate all local scalar products beween gradients ...')
        D = self.advection_function(self.grid.centers(0), mu=mu)
        SF_INTS = - np.einsum('pc,eqi,c,e,ec->eqp', SFQ, SF_GRADS, w, g.integration_elements(0), D).ravel()
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

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    advection_function
        The |Function| `v(x)` with `shape_range = (grid.dim, )`.
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
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, advection_function=None, advection_constant=None,
                 dirichlet_clear_columns=False, dirichlet_clear_diag=False,
                 solver_options=None, name=None):
        assert grid.reference_element(0) in {square}, 'A square grid is expected!'
        assert advection_function is None \
            or (isinstance(advection_function, FunctionInterface)
                and advection_function.dim_domain == grid.dim
                and advection_function.shape_range == (grid.dim,))
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid)
        if advection_function is not None:
            self.build_parameter_type(advection_function)

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

        D = self.advection_function(self.grid.centers(0), mu=mu)
        SF_INTS = - np.einsum('pc,eqic,c,e,ei->eqp', SFQ, SF_GRADS, w, g.integration_elements(0), D).ravel()
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
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, robin_data=None, solver_options=None, name=None):
        assert robin_data is None or (isinstance(robin_data, tuple) and len(robin_data) == 2)
        assert robin_data is None or all([isinstance(f, FunctionInterface)
                                          and f.dim_domain == grid.dim
                                          and (f.shape_range == ()
                                               or f.shape_range == (grid.dim,))
                                          for f in robin_data])
        self.__auto_init(locals())
        self.source = self.range = CGVectorSpace(grid)
        if self.robin_data is not None:
            self.build_parameter_type(self.robin_data[0])

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
            xref = g.centers(1)[RI]
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
            q, w = line.quadrature(order=2)
            # remove last dimension of q, as line coordinates are one dimensional
            q = q[:, 0]
            SF = np.array([1 - q, q])
            SF_INTS = np.einsum('e,pi,pj,e,p->eij', robin_c, SF, SF, g.integration_elements(1)[RI], w).ravel()
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
        assert function.dim_domain == grid.dim
        assert function.shape_range == ()
        self.__auto_init(locals())
        self.range = CGVectorSpace(grid)
        self.build_parameter_type(function)

    def _assemble(self, mu=None):
        return self.function.evaluate(self.grid.centers(self.grid.dim), mu=mu).reshape((-1, 1))


class BoundaryInterpolationOperator(InterpolationOperator):
    """|InterpolationOperator| restricted to a part of the boundary..

    Parameters
    ----------
    grid
        The |Grid| on which to interpolate.
    function
        The |Function| to interpolate.
    boundary_info
        The |BoundaryInfo| defining the physical boundary (has to match grid).
    boundary_type
        The type of the physical boundary.
    """

    def __init__(self, grid, function, boundary_info, boundary_type='dirichlet'):
        super().__init__(grid, function)
        self.__auto_init(locals())
        self.boundary_mask = boundary_info.boundaries(boundary_type, grid.dim)

    def _assemble(self, mu=None):
        result = np.zeros(self.range.dim)
        interpolation = super()._assemble(mu=mu)
        result[self.boundary_mask] = interpolation[self.boundary_mask].ravel()
        return result.reshape((-1, 1))
