# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from pymor.la import NumpyVectorArray
from pymor.functions import FunctionInterface
from pymor.grids.referenceelements import triangle, line
from pymor.grids.subgrid import SubGrid
from pymor.grids.boundaryinfos import SubGridBoundaryInfo
from pymor.operators import OperatorBase, NumpyMatrixBasedOperator, NumpyMatrixOperator
from pymor.operators.constructions import Concatenation, ComponentProjection
from pymor.tools.inplace import iadd_masked, isub_masked


class NonlinearAdvectionLaxFriedrichs(OperatorBase):
    '''Nonlinear Finite Volume Advection operator using Lax-Friedrichs-Flux.
    '''

    type_source = type_range = NumpyVectorArray

    def __init__(self, grid, boundary_info, flux, lxf_lambda=1.0, dirichlet_data=None, name=None):
        assert dirichlet_data is None or isinstance(dirichlet_data, FunctionInterface)

        super(NonlinearAdvectionLaxFriedrichs, self).__init__()
        self.grid = grid
        self.boundary_info = boundary_info
        self.flux = flux
        self.lxf_lambda = lxf_lambda
        self.dirichlet_data = dirichlet_data
        self.name = name
        if isinstance(dirichlet_data, FunctionInterface) and boundary_info.has_dirichlet:
            if dirichlet_data.parametric:
                self.build_parameter_type(inherits=(flux, dirichlet_data))
            else:
                self._dirichlet_values = self.dirichlet_data(grid.centers(1)[boundary_info.dirichlet_boundaries(1)])
                self._dirichlet_values = self._dirichlet_values.ravel()
                self._dirichlet_values_flux_shaped = self._dirichlet_values.reshape((-1,1))
                self.build_parameter_type(inherits=(flux,))
        else:
            self.build_parameter_type(inherits=(flux,))
        self.dim_source = self.dim_range = grid.size(0)

    def restricted(self, components):
        source_dofs = np.setdiff1d(np.union1d(self.grid.neighbours(0, 0)[components].ravel(), components),
                                   np.array([-1], dtype=np.int32),
                                   assume_unique=True)
        sub_grid = SubGrid(self.grid, entities=source_dofs)
        sub_boundary_info = SubGridBoundaryInfo(sub_grid, self.grid, self.boundary_info)
        op = NonlinearAdvectionLaxFriedrichs(sub_grid, sub_boundary_info, self.flux, self.lxf_lambda,
                                             self.dirichlet_data, '{}_restricted'.format(self.name))
        sub_grid_indices = sub_grid.indices_from_parent_indices(components, codim=0)
        proj = ComponentProjection(sub_grid_indices, op.dim_range, op.type_range)
        return Concatenation(proj, op), sub_grid.parent_indices(0)

    def apply(self, U, ind=None, mu=None):
        assert isinstance(U, NumpyVectorArray)
        assert U.dim == self.dim_source
        mu = self.parse_parameter(mu)

        ind = xrange(len(U)) if ind is None else ind
        U = U._array
        R = np.zeros((len(ind), self.dim_source))

        g = self.grid
        bi = self.boundary_info
        N = g.neighbours(0, 0)
        SUPE = g.superentities(1, 0)
        SUPI = g.superentity_indices(1, 0)
        assert SUPE.ndim == 2
        VOLS = g.volumes(1)
        boundaries = g.boundaries(1)
        unit_outer_normals = g.unit_outer_normals()[SUPE[:,0], SUPI[:,0]]

        for i, j in enumerate(ind):
            Ui = U[j]
            Ri = R[i]

            F = self.flux(Ui[..., np.newaxis], mu=mu)
            F_edge = F[SUPE]
            U_edge = Ui[SUPE]

            # boundary treatment

            F_edge[boundaries, 1] = F_edge[boundaries, 0]
            U_edge[boundaries, 1] = U_edge[boundaries, 0]

            if bi.has_dirichlet:
                dirichlet_boundaries = bi.dirichlet_boundaries(1)
                if hasattr(self, '_dirichlet_values'):
                    F_edge[dirichlet_boundaries, 1] = self.flux(self._dirichlet_values_flux_shaped, mu=mu)
                    U_edge[dirichlet_boundaries, 1] = self._dirichlet_values
                elif self.dirichlet_data is not None:
                    dirichlet_values = self.dirichlet_data(g.centers(1)[dirichlet_boundaries],
                                                           mu=mu)
                    F_edge[dirichlet_boundaries, 1] = self.flux(dirichlet_values.reshape((-1,1)), mu=mu)
                    U_edge[dirichlet_boundaries, 1] = dirichlet_values
                else:
                    F_edge[dirichlet_boundaries, 1] = 0
                    U_edge[dirichlet_boundaries, 1] = 0

            if bi.has_neumann:
                F_edge[bi.neumann_boundaries(1)] = 0
                U_edge[bi.neumann_boundaries(1)] = 0

            F_edge = np.sum(np.sum(F_edge, axis=1) * unit_outer_normals, axis=1)
            U_edge = (U_edge[:,0] - U_edge[:,1]) * (1. / self.lxf_lambda)

            TOT_edge = F_edge + U_edge
            TOT_edge *= 0.5 * VOLS

            iadd_masked(Ri, TOT_edge, SUPE[:,0])
            isub_masked(Ri, TOT_edge, SUPE[:,1])

        R /= g.volumes(0)

        return NumpyVectorArray(R)


class NonlinearAdvectionEngquistOsher(OperatorBase):
    '''Nonlinear Finite Volume Advection operator using EngquistOsher-Flux.

    It is assumed that the flux is zero at 0 and its derivative only changes sign at 0.
    '''

    type_source = type_range = NumpyVectorArray

    def __init__(self, grid, boundary_info, flux, flux_derivative, dirichlet_data=None, name=None):
        assert dirichlet_data is None or isinstance(dirichlet_data, FunctionInterface)

        super(NonlinearAdvectionEngquistOsher, self).__init__()
        self.grid = grid
        self.boundary_info = boundary_info
        self.flux = flux
        self.flux_derivative = flux_derivative
        self.dirichlet_data = dirichlet_data
        self.name = name
        if isinstance(dirichlet_data, FunctionInterface) and boundary_info.has_dirichlet:
            if dirichlet_data.parametric:
                self.build_parameter_type(inherits=(flux, flux_derivative, dirichlet_data))
            else:
                self._dirichlet_values = self.dirichlet_data(grid.centers(1)[boundary_info.dirichlet_boundaries(1)])
                self._dirichlet_values = self._dirichlet_values.ravel()
                self._dirichlet_values_flux_shaped = self._dirichlet_values.reshape((-1,1))
                self.build_parameter_type(inherits=(flux, flux_derivative))
        else:
            self.build_parameter_type(inherits=(flux, flux_derivative))
        self.dim_source = self.dim_range = grid.size(0)

    def restricted(self, components):
        source_dofs = np.setdiff1d(np.union1d(self.grid.neighbours(0, 0)[components].ravel(), components),
                                   np.array([-1], dtype=np.int32),
                                   assume_unique=True)
        sub_grid = SubGrid(self.grid, entities=source_dofs)
        sub_boundary_info = SubGridBoundaryInfo(sub_grid, self.grid, self.boundary_info)
        op = NonlinearAdvectionEngquistOsher(sub_grid, sub_boundary_info, self.flux, self.flux_derivative,
                                             self.dirichlet_data, '{}_restricted'.format(self.name))
        sub_grid_indices = sub_grid.indices_from_parent_indices(components, codim=0)
        proj = ComponentProjection(sub_grid_indices, op.dim_range, op.type_range)
        return Concatenation(proj, op), sub_grid.parent_indices(0)

    def apply(self, U, ind=None, mu=None):
        assert isinstance(U, NumpyVectorArray)
        assert U.dim == self.dim_source
        mu = self.parse_parameter(mu)

        ind = xrange(len(U)) if ind is None else ind
        U = U._array
        R = np.zeros((len(ind), self.dim_source))

        g = self.grid
        bi = self.boundary_info
        N = g.neighbours(0, 0)
        SUPE = g.superentities(1, 0)
        SUPI = g.superentity_indices(1, 0)
        assert SUPE.ndim == 2
        VOLS = g.volumes(1)
        boundaries = g.boundaries(1)
        unit_outer_normals = g.unit_outer_normals()[SUPE[:,0], SUPI[:,0]][:, np.newaxis, :]

        for i, j in enumerate(ind):
            Ui = U[j]
            Ri = R[i]

            F = self.flux(Ui[..., np.newaxis], mu=mu)
            F_d = self.flux_derivative(Ui[..., np.newaxis], mu=mu)
            F_edge = F[SUPE]
            F_d_edge = F_d[SUPE]

            # boundary treatment

            F_edge[boundaries, 1] = 0
            F_d_edge[boundaries, 1] = 0

            if bi.has_dirichlet and self.dirichlet_data is not None:
                dirichlet_boundaries = bi.dirichlet_boundaries(1)
                if hasattr(self, '_dirichlet_values'):
                    F_d_edge[dirichlet_boundaries, 1] = self.flux_derivative(self._dirichlet_values_flux_shaped, mu=mu)
                    F_edge[dirichlet_boundaries, 1] = self.flux(self._dirichlet_values_flux_shaped, mu=mu)
                else:
                    dirichlet_values = self.dirichlet_data(g.centers(1)[dirichlet_boundaries], mu=mu).reshape((-1,1))
                    F_d_edge[dirichlet_boundaries, 1] = self.flux_derivative(dirichlet_values, mu=mu)
                    F_edge[dirichlet_boundaries, 1] = self.flux(dirichlet_values, mu=mu)

            F_d_edge = np.sum(F_d_edge * unit_outer_normals, axis=2)
            F_edge = np.sum(F_edge * unit_outer_normals, axis=2)
            F_edge[:, 0] = np.where(np.greater_equal(F_d_edge[:, 0], 0), F_edge[:, 0], 0)
            F_edge[:, 1] = np.where(np.less_equal(F_d_edge[:, 1], 0), F_edge[:, 1], 0)
            F_edge = np.sum(F_edge, axis=1)
            F_edge *= VOLS

            if bi.has_neumann:
                F_edge[bi.neumann_boundaries(1)] = 0

            iadd_masked(Ri, F_edge, SUPE[:,0])
            isub_masked(Ri, F_edge, SUPE[:,1])

        R /= g.volumes(0)

        return NumpyVectorArray(R)


class L2Product(NumpyMatrixBasedOperator):
    '''Operator representing the L2-product for finite volume functions.

    To evaluate the product use the apply2 method.

    Parameters
    ----------
    grid
        The grid on which to assemble the product.
    name
        The name of the product.
    '''

    type_source = type_range = NumpyVectorArray
    sparse = True

    def __init__(self, grid, name=None):
        super(L2Product, self).__init__()
        self.dim_source = grid.size(0)
        self.dim_range = self.dim_source
        self.grid = grid
        self.name = name

    def _assemble(self, mu=None):
        assert self.check_parameter(mu)

        A = diags(self.grid.volumes(0), 0)

        return NumpyMatrixOperator(A)

class L2ProductFunctional(NumpyMatrixBasedOperator):
    '''Scalar product with an L2-function for finite volume functions.

    Parameters
    ----------
    grid
        Grid over which to assemble the functional.
    function
        The `Function` with which to take the scalar product.
    name
        The name of the functional.
    '''

    type_source = type_range = NumpyVectorArray
    sparse = False

    def __init__(self, grid, function, name=None):
        assert function.shape_range == tuple()
        super(L2ProductFunctional, self).__init__()
        self.dim_source = grid.size(0)
        self.dim_range = 1
        self.grid = grid
        self.function = function
        self.name = name
        self.build_parameter_type(inherits=(function,))

    def _assemble(self, mu=None):
        mu = self.parse_parameter(mu)
        g = self.grid

        # evaluate function at all quadrature points -> shape = (g.size(0), number of quadrature points, 1)
        F = self.function(g.quadrature_points(0, order=2), mu=mu)

        _, w = g.reference_element.quadrature(order=2)

        # integrate the products of the function with the shape functions on each element
        # -> shape = (g.size(0), number of shape functions)
        F_INTS = np.einsum('ei,e,i->e', F, g.integration_elements(0), w).ravel()
        F_INTS /= g.volumes(0)

        return NumpyMatrixOperator(F_INTS.reshape((1, -1)))
