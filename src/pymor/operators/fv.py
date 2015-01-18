# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

""" This module provides some operators for finite volume discretizations."""

from __future__ import absolute_import, division, print_function

from itertools import izip
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, dia_matrix

from pymor.core.interfaces import ImmutableInterface, abstractmethod
from pymor.functions.interfaces import FunctionInterface
from pymor.grids.interfaces import AffineGridWithOrthogonalCentersInterface
from pymor.grids.boundaryinfos import SubGridBoundaryInfo
from pymor.grids.subgrid import SubGrid
from pymor.la.numpyvectorarray import NumpyVectorArray, NumpyVectorSpace
from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import Concatenation, ComponentProjection
from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.parameters.base import Parametric
from pymor.tools.arguments import method_arguments
from pymor.tools.inplace import iadd_masked, isub_masked
from pymor.tools.quadratures import GaussQuadratures


class NumericalConvectiveFluxInterface(ImmutableInterface, Parametric):
    """Interface for numerical convective fluxes for finite volume schemes.

    Numerical fluxes defined by this interfaces are functions of
    the form `F(U_inner, U_outer, unit_outer_normal, edge_volume, mu)`.

    The flux evaluation is vectorized and happens in two stages:
      1. `evaluate_stage1` receives a |NumPy array| `U` of all values which
         appear as `U_inner` or `U_outer` for one of the edges the
         flux shall be evaluated at and returns a `tuple` of |NumPy arrays|
         each of the same length as `U`.
      2. `evaluate_stage2` receives the reordered `stage1_data` for each
         edge as well as the unit outer normal and the volume of the edges.

         `stage1_data` is given as follows: If `R_l` is `l`-th entry of the
         `tuple` returned by `evaluate_stage1`, the `l`-th entry `D_l` of
         of the `stage1_data` tuple has the shape `(num_edges, 2) + R_l.shape[1:]`.
         If for edge `k` the values `U_inner` and `U_outer` are the `i`-th
         and `j`-th value in the `U` array provided to `evaluate_stage1`,
         we have ::

             D_l[k, 0] == R_l[i],    D_l[k, 1] == R_l[j].

         `evaluate_stage2` returns a |NumPy array| of the flux evaluations
         for each edge.
    """

    @abstractmethod
    def evaluate_stage1(self, U, mu=None):
        pass

    @abstractmethod
    def evaluate_stage2(self, stage1_data, unit_outer_normals, volumes, mu=None):
        pass


class LaxFriedrichsFlux(NumericalConvectiveFluxInterface):
    """Lax-Friedrichs numerical flux.

    If `f` is the analytical flux, the Lax-Friedrichs flux is given
    by ::

      F(U_in, U_out, normal, vol) = vol * [normal⋅(f(U_in) + f(U_out))/2 + (U_in - U_out)/(2*λ)]

    Parameters
    ----------
    flux
        |Function| defining the analytical flux `f`.
    lxf_lambda
        The stabilization parameter `λ`.
    """

    def __init__(self, flux, lxf_lambda=1.0):
        self.flux = flux
        self.lxf_lambda = lxf_lambda
        self.build_parameter_type(inherits=(flux,))

    def evaluate_stage1(self, U, mu=None):
        return U, self.flux(U[..., np.newaxis], mu)

    def evaluate_stage2(self, stage1_data, unit_outer_normals, volumes, mu=None):
        U, F = stage1_data
        return (np.sum(np.sum(F, axis=1) * unit_outer_normals, axis=1) * 0.5
                + (U[..., 0] - U[..., 1]) * (0.5 / self.lxf_lambda)) * volumes


class SimplifiedEngquistOsherFlux(NumericalConvectiveFluxInterface):
    """Engquist-Osher numerical flux. Simplified Implementation for special case.

    For the definition of the Enquist-Osher flux see :class:`EngquistOsherFlux`.
    This class provides a faster and more accurate implementation for the special
    case that `f(0) == 0` and `f'` only changes sign at `0`.

    Parameters
    ----------
    flux
        |Function| defining the analytical flux `f`.
    flux_derivative
        |Function| defining the analytical flux derivative `f'`.
    """

    def __init__(self, flux, flux_derivative):
        self.flux = flux
        self.flux_derivative = flux_derivative
        self.build_parameter_type(inherits=(flux, flux_derivative))

    def evaluate_stage1(self, U, mu=None):
        return self.flux(U[..., np.newaxis], mu), self.flux_derivative(U[..., np.newaxis], mu)

    def evaluate_stage2(self, stage1_data, unit_outer_normals, volumes, mu=None):
        F_edge, F_d_edge = stage1_data
        unit_outer_normals = unit_outer_normals[:, np.newaxis, :]
        F_d_edge = np.sum(F_d_edge * unit_outer_normals, axis=2)
        F_edge = np.sum(F_edge * unit_outer_normals, axis=2)
        F_edge[:, 0] = np.where(np.greater_equal(F_d_edge[:, 0], 0), F_edge[:, 0], 0)
        F_edge[:, 1] = np.where(np.less_equal(F_d_edge[:, 1], 0), F_edge[:, 1], 0)
        F_edge = np.sum(F_edge, axis=1)
        F_edge *= volumes
        return F_edge


class EngquistOsherFlux(NumericalConvectiveFluxInterface):
    """Engquist-Osher numerical flux.

    If `f` is the analytical flux, and `f'` its derivative, the Engquist-Osher flux is
    given by ::

      F(U_in, U_out, normal, vol) = vol * [c^+(U_in, normal)  +  c^-(U_out, normal)]

                                         U_in
      c^+(U_in, normal)  = f(0)⋅normal +  ∫   max(f'(s)⋅normal, 0) ds
                                         s=0

                                        U_out
      c^-(U_out, normal) =                ∫   min(f'(s)⋅normal, 0) ds
                                         s=0


    Parameters
    ----------
    flux
        |Function| defining the analytical flux `f`.
    flux_derivative
        |Function| defining the analytical flux derivative `f'`.
    gausspoints
        Number of Gauss quadrature points to be used for integration.
    intervals
        Number of subintervals to be used for integration.
    """

    def __init__(self, flux, flux_derivative, gausspoints=5, intervals=1):
        self.flux = flux
        self.flux_derivative = flux_derivative
        self.gausspoints = gausspoints
        self.intervals = intervals
        self.build_parameter_type(inherits=(flux, flux_derivative))
        points, weights = GaussQuadratures.quadrature(npoints=self.gausspoints)
        points = points / intervals
        points = ((np.arange(self.intervals, dtype=np.float)[:, np.newaxis] * (1 / intervals))
                  + points[np.newaxis, :]).ravel()
        weights = np.tile(weights, intervals) * (1 / intervals)
        self.points = points
        self.weights = weights

    def evaluate_stage1(self, U, mu=None):
        int_els = np.abs(U)[:, np.newaxis, np.newaxis]
        return [np.concatenate([self.flux_derivative(U[:, np.newaxis] * p, mu)[:, np.newaxis, :] * int_els * w
                               for p, w in izip(self.points, self.weights)], axis=1)]

    def evaluate_stage2(self, stage1_data, unit_outer_normals, volumes, mu=None):
        F0 = np.sum(self.flux.evaluate(np.array([[0.]]), mu=mu) * unit_outer_normals, axis=1)
        Fs = np.sum(stage1_data[0] * unit_outer_normals[:, np.newaxis, np.newaxis, :], axis=3)
        Fs[:, 0, :] = np.maximum(Fs[:, 0, :], 0)
        Fs[:, 1, :] = np.minimum(Fs[:, 1, :], 0)
        Fs = np.sum(np.sum(Fs, axis=2), axis=1) + F0
        Fs *= volumes
        return Fs


class NonlinearAdvectionOperator(OperatorBase):
    """Nonlinear finite volume advection |Operator|.

    The operator is of the form ::

        L(u, mu)(x) = ∇ ⋅ f(u(x), mu)

    .. note ::
        For Neumann boundaries, currently only zero boundary values are implemented.

    Parameters
    ----------
    grid
        |Grid| over which to evaluate the operator.
    boundary_info
        |BoundaryInfo| determining the Dirichlet and Neumann boundaries.
    numerical_flux
        The :class:`NumericalConvectiveFlux <NumericalConvectiveFluxInterface>` to use.
    dirichlet_data
        |Function| providing the Dirichlet boundary values. If `None`, constant-zero
        boundary is assumed.
    name
        The name of the operator.
    """

    linear = False

    def __init__(self, grid, boundary_info, numerical_flux, dirichlet_data=None, name=None):
        assert dirichlet_data is None or isinstance(dirichlet_data, FunctionInterface)

        self.grid = grid
        self.boundary_info = boundary_info
        self.numerical_flux = numerical_flux
        self.dirichlet_data = dirichlet_data
        self.name = name
        if (isinstance(dirichlet_data, FunctionInterface) and boundary_info.has_dirichlet
                and not dirichlet_data.parametric):
            self._dirichlet_values = self.dirichlet_data(grid.centers(1)[boundary_info.dirichlet_boundaries(1)])
            self._dirichlet_values = self._dirichlet_values.ravel()
            self._dirichlet_values_flux_shaped = self._dirichlet_values.reshape((-1, 1))
        self.build_parameter_type(inherits=(numerical_flux, dirichlet_data))
        self.source = self.range = NumpyVectorSpace(grid.size(0))
        self.with_arguments = self.with_arguments.union('numerical_flux_{}'.format(arg)
                                                        for arg in numerical_flux.with_arguments)

    with_arguments = frozenset(method_arguments(__init__))

    def with_(self, **kwargs):
        assert 'numerical_flux' not in kwargs or not any(arg.startswith('numerical_flux_') for arg in kwargs)
        num_flux_args = {arg[len('numerical_flux_'):]: kwargs.pop(arg)
                         for arg in list(kwargs) if arg.startswith('numerical_flux_')}
        if num_flux_args:
            kwargs['numerical_flux'] = self.numerical_flux.with_(**num_flux_args)
        return self._with_via_init(kwargs)

    def restricted(self, components):
        source_dofs = np.setdiff1d(np.union1d(self.grid.neighbours(0, 0)[components].ravel(), components),
                                   np.array([-1], dtype=np.int32),
                                   assume_unique=True)
        sub_grid = SubGrid(self.grid, entities=source_dofs)
        sub_boundary_info = SubGridBoundaryInfo(sub_grid, self.grid, self.boundary_info)
        op = self.with_(grid=sub_grid, boundary_info=sub_boundary_info, name='{}_restricted'.format(self.name))
        sub_grid_indices = sub_grid.indices_from_parent_indices(components, codim=0)
        proj = ComponentProjection(sub_grid_indices, op.range)
        return Concatenation(proj, op), sub_grid.parent_indices(0)

    def apply(self, U, ind=None, mu=None):
        assert isinstance(U, NumpyVectorArray)
        assert U in self.source
        mu = self.parse_parameter(mu)

        ind = xrange(len(U)) if ind is None else ind
        U = U.data
        R = np.zeros((len(ind), self.source.dim))

        g = self.grid
        bi = self.boundary_info
        SUPE = g.superentities(1, 0)
        SUPI = g.superentity_indices(1, 0)
        assert SUPE.ndim == 2
        VOLS = g.volumes(1)
        boundaries = g.boundaries(1)
        unit_outer_normals = g.unit_outer_normals()[SUPE[:, 0], SUPI[:, 0]]

        if bi.has_dirichlet:
            dirichlet_boundaries = bi.dirichlet_boundaries(1)
            if hasattr(self, '_dirichlet_values'):
                dirichlet_values = self._dirichlet_values
            elif self.dirichlet_data is not None:
                dirichlet_values = self.dirichlet_data(g.centers(1)[dirichlet_boundaries], mu=mu)
            else:
                dirichlet_values = np.zeros_like(dirichlet_boundaries)
            F_dirichlet = self.numerical_flux.evaluate_stage1(dirichlet_values, mu)

        for i, j in enumerate(ind):
            Ui = U[j]
            Ri = R[i]

            F = self.numerical_flux.evaluate_stage1(Ui, mu)
            F_edge = [f[SUPE] for f in F]

            for f in F_edge:
                f[boundaries, 1] = f[boundaries, 0]
            if bi.has_dirichlet:
                for f, f_d in izip(F_edge, F_dirichlet):
                    f[dirichlet_boundaries, 1] = f_d

            NUM_FLUX = self.numerical_flux.evaluate_stage2(F_edge, unit_outer_normals, VOLS, mu)

            if bi.has_neumann:
                NUM_FLUX[bi.neumann_boundaries(1)] = 0

            iadd_masked(Ri, NUM_FLUX, SUPE[:, 0])
            isub_masked(Ri, NUM_FLUX, SUPE[:, 1])

        R /= g.volumes(0)

        return NumpyVectorArray(R)


def nonlinear_advection_lax_friedrichs_operator(grid, boundary_info, flux, lxf_lambda=1.0,
                                                dirichlet_data=None, name=None):
    """Instantiate a :class:`NonlinearAdvectionOperator` using :class:`LaxFriedrichsFlux`."""
    num_flux = LaxFriedrichsFlux(flux, lxf_lambda)
    return NonlinearAdvectionOperator(grid, boundary_info, num_flux, dirichlet_data, name)


def nonlinear_advection_simplified_engquist_osher_operator(grid, boundary_info, flux, flux_derivative,
                                                           dirichlet_data=None, name=None):
    """Instantiate a :class:`NonlinearAdvectionOperator` using :class:`SimplifiedEngquistOsherFlux`."""
    num_flux = SimplifiedEngquistOsherFlux(flux, flux_derivative)
    return NonlinearAdvectionOperator(grid, boundary_info, num_flux, dirichlet_data, name)


def nonlinear_advection_engquist_osher_operator(grid, boundary_info, flux, flux_derivative, gausspoints=5, intervals=1,
                                                dirichlet_data=None, name=None):
    """Instantiate a :class:`NonlinearAdvectionOperator` using :class:`EngquistOsherFlux`."""
    num_flux = EngquistOsherFlux(flux, flux_derivative, gausspoints=gausspoints, intervals=intervals)
    return NonlinearAdvectionOperator(grid, boundary_info, num_flux, dirichlet_data, name)


class LinearAdvectionLaxFriedrichs(NumpyMatrixBasedOperator):
    """Linear advection finite Volume |Operator| using Lax-Friedrichs flux.

    The operator is of the form ::

        L(u, mu)(x) = ∇ ⋅ (v(x, mu)⋅u(x))

    See :class:`LaxFriedrichsFlux` for the definition of the Lax-Friedrichs flux.

    Parameters
    ----------
    grid
        |Grid| over which to assemble the operator.
    boundary_info
        |BoundaryInfo| determining the Dirichlet and Neumann boundaries.
    velocity_field
        |Function| defining the velocity field `v`.
    lxf_lambda
        The stabilization parameter `λ`.
    name
        The name of the operator.
    """

    def __init__(self, grid, boundary_info, velocity_field, lxf_lambda=1.0, name=None):
        self.grid = grid
        self.boundary_info = boundary_info
        self.velocity_field = velocity_field
        self.lxf_lambda = lxf_lambda
        self.name = name
        self.build_parameter_type(inherits=(velocity_field,))
        self.source = self.range = NumpyVectorSpace(grid.size(0))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info
        SUPE = g.superentities(1, 0)
        SUPI = g.superentity_indices(1, 0)
        assert SUPE.ndim == 2
        edge_volumes = g.volumes(1)
        boundary_edges = g.boundaries(1)
        inner_edges = np.setdiff1d(np.arange(g.size(1)), boundary_edges)
        dirichlet_edges = bi.dirichlet_boundaries(1) if bi.has_dirichlet else np.array([], ndmin=1, dtype=np.int)
        neumann_edges = bi.neumann_boundaries(1) if bi.has_neumann else np.array([], ndmin=1, dtype=np.int)
        outflow_edges = np.setdiff1d(boundary_edges, np.hstack([dirichlet_edges, neumann_edges]))
        normal_velocities = np.einsum('ei,ei->e',
                                      self.velocity_field(g.centers(1), mu=mu),
                                      g.unit_outer_normals()[SUPE[:, 0], SUPI[:, 0]])

        nv_inner = normal_velocities[inner_edges]
        l_inner = np.ones_like(nv_inner) * (1. / self.lxf_lambda)
        I0_inner = np.hstack([SUPE[inner_edges, 0], SUPE[inner_edges, 0], SUPE[inner_edges, 1], SUPE[inner_edges, 1]])
        I1_inner = np.hstack([SUPE[inner_edges, 0], SUPE[inner_edges, 1], SUPE[inner_edges, 0], SUPE[inner_edges, 1]])
        V_inner = np.hstack([nv_inner, nv_inner, -nv_inner, -nv_inner])
        V_inner += np.hstack([l_inner, -l_inner, -l_inner, l_inner])
        V_inner *= np.tile(0.5 * edge_volumes[inner_edges], 4)

        I_out = SUPE[outflow_edges, 0]
        V_out = edge_volumes[outflow_edges] * normal_velocities[outflow_edges]

        I_dir = SUPE[dirichlet_edges, 0]
        V_dir = edge_volumes[outflow_edges] * (0.5 * normal_velocities[dirichlet_edges] + 0.5 / self.lxf_lambda)

        I0 = np.hstack([I0_inner, I_out, I_dir])
        I1 = np.hstack([I1_inner, I_out, I_dir])
        V = np.hstack([V_inner, V_out, V_dir])

        A = coo_matrix((V, (I0, I1)), shape=(g.size(0), g.size(0)))
        A = csc_matrix(A).copy()   # See pymor.operators.cg.DiffusionOperatorP1 for why copy() is necessary
        A = dia_matrix(([1. / g.volumes(0)], [0]), shape=(g.size(0),) * 2) * A

        return A


class L2Product(NumpyMatrixBasedOperator):
    """|Operator| representing the L2-product for finite volume functions.

    To evaluate the product use the :meth:`~pymor.operators.interfaces module.OperatorInterface.apply2`
    method.

    Parameters
    ----------
    grid
        The |Grid| over which to assemble the product.
    name
        The name of the product.
    """

    sparse = True

    def __init__(self, grid, name=None):
        self.source = self.range = NumpyVectorSpace(grid.size(0))
        self.grid = grid
        self.name = name

    def _assemble(self, mu=None):

        A = dia_matrix((self.grid.volumes(0), [0]), shape=(self.grid.size(0),) * 2)

        return A


class L2ProductFunctional(NumpyMatrixBasedOperator):
    """Finite volume |Functional| representing the scalar product with an L2-|Function|.

    Parameters
    ----------
    grid
        |Grid| over which to assemble the functional.
    function
        The |Function| with which to take the scalar product.
    order
        Order of the Gauss quadrature to use for numerical integration.
    name
        The name of the functional.
    """

    range = NumpyVectorSpace(1)
    sparse = False

    def __init__(self, grid, function, order=2, name=None):
        assert function.shape_range == tuple()
        self.source = NumpyVectorSpace(grid.size(0))
        self.grid = grid
        self.function = function
        self.order = order
        self.name = name
        self.build_parameter_type(inherits=(function,))

    def _assemble(self, mu=None):
        g = self.grid

        # evaluate function at all quadrature points -> shape = (g.size(0), number of quadrature points, 1)
        F = self.function(g.quadrature_points(0, order=self.order), mu=mu)

        _, w = g.reference_element.quadrature(order=self.order)

        # integrate the products of the function with the shape functions on each element
        # -> shape = (g.size(0), number of shape functions)
        F_INTS = np.einsum('ei,e,i->e', F, g.integration_elements(0), w).ravel()
        F_INTS /= g.volumes(0)

        return F_INTS.reshape((1, -1))


class DiffusionOperator(NumpyMatrixBasedOperator):
    """Linear finite volume diffusion |Operator| using finite differences.

    The operator is of the form ::

    L(u, mu)(x) = ∇ ⋅ (mu ∇ u(x))

    Parameters
    ----------
    grid
        |Grid| over which to assemble the operator.
    boundary_info
        |BoundaryInfo| determining the Dirichlet and Neumann boundaries.
    diffusion
        Diffusion scalar 'mu'.
    name
        The name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, diffusion=1.0, name=None):
        super(DiffusionOperator, self).__init__()
        self.grid = grid
        self.boundary_info = boundary_info
        self.diffusion = diffusion
        self.name = name
        self.source = self.range = NumpyVectorSpace(grid.size(0))

    def _assemble(self, mu=None):
        diffusion = self.diffusion
        grid = self.grid
        assert isinstance(grid, AffineGridWithOrthogonalCentersInterface)
        centers = grid.centers(1)
        orthogonal_centers = grid.orthogonal_centers()
        boundary_mask = grid.boundary_mask(1)
        VOLS = grid.volumes(1)
        DVOLS = np.zeros(VOLS.size)

        SE_I0 = grid.superentities(1, 0)[:, 0]
        SE_I1 = grid.superentities(1, 0)[:, 1]

        SE_I0_I = SE_I0[~boundary_mask]
        SE_I1_I = SE_I1[~boundary_mask]

        DVOLS[~boundary_mask] = np.abs(orthogonal_centers[SE_I0_I, 0] - orthogonal_centers[SE_I1_I, 0])
        DVOLS[~boundary_mask] += np.abs(orthogonal_centers[SE_I0_I, 1] - orthogonal_centers[SE_I1_I, 1])

        # compute shift for periodic boundaries
        embeddings = grid.embeddings(0)
        reference_element = grid.reference_element(0)
        sub_reference_element_centers = reference_element.sub_reference_element(1).center()
        subentity_embedding = reference_element.subentity_embedding(1)
        se_w_i = grid._superentities_with_indices(1, 0)
        x = subentity_embedding[0][se_w_i[1][:, 0], :, 0] * sub_reference_element_centers + subentity_embedding[1][se_w_i[1][:, 0]]
        y = subentity_embedding[0][se_w_i[1][:, 1], :, 0] * sub_reference_element_centers + subentity_embedding[1][se_w_i[1][:, 1]]
        for i in xrange(len(x)):
            x[i] = np.dot(embeddings[0][se_w_i[0][i, 0]],x[i]) + embeddings[1][se_w_i[0][i, 0]]
            y[i] = np.dot(embeddings[0][se_w_i[0][i, 1]],y[i]) + embeddings[1][se_w_i[0][i, 1]]
        DVOLS[~boundary_mask] += (x[:, 0]-y[:, 0])[~boundary_mask]
        DVOLS[~boundary_mask] += (x[:, 1]-y[:, 1])[~boundary_mask]
        DVOLS = np.abs(DVOLS)

        SE_I0_B = SE_I0[boundary_mask]
        DVOLS[boundary_mask] = np.abs(centers[boundary_mask, 0] - orthogonal_centers[SE_I0_B, 0])
        DVOLS[boundary_mask] += np.abs(centers[boundary_mask, 1] - orthogonal_centers[SE_I0_B, 1])

        SE_INTS = diffusion * VOLS[~boundary_mask] / DVOLS[~boundary_mask]

        A1 = coo_matrix((- SE_INTS, (SE_I0_I, SE_I1_I)), shape=(self.source.dim, self.source.dim))
        A2 = coo_matrix((- SE_INTS, (SE_I1_I, SE_I0_I)), shape=(self.source.dim, self.source.dim))
        A3 = coo_matrix((SE_INTS, (SE_I0_I, SE_I0_I)), shape=(self.source.dim, self.source.dim))
        A4 = coo_matrix((SE_INTS, (SE_I1_I, SE_I1_I)), shape=(self.source.dim, self.source.dim))

        if self.boundary_info.has_dirichlet:
            dirichlet_mask = self.boundary_info.dirichlet_mask(1)
            SE_I0_D = SE_I0[dirichlet_mask]
            SE_INTS = diffusion * VOLS[dirichlet_mask] / DVOLS[dirichlet_mask]

            A5 = coo_matrix((SE_INTS, (SE_I0_D, SE_I0_D)), shape=(self.source.dim, self.source.dim))

            A = csc_matrix(A1 + A2 + A3 + A4 + A5).copy()
        else:
            A = csc_matrix(A1 + A2 + A3 + A4).copy()

        A = dia_matrix(([1. / grid.volumes(0)], [0]), shape=(grid.size(0),) * 2) * A

        return A


class DiffusionRHSOperatorFunctional(NumpyMatrixBasedOperator):
    """Finite volume |Functional| representing the rhs of the DiffusionOperator
       + the scalar product with an L2-|Function|.

    Parameters
    ----------
    grid
        |Grid| over which to assemble the functional.
    boundary_info
        |BoundaryInfo| determining the Dirichlet and Neumann boundaries.
    function
        The |Function| with which to take the scalar product.
    neumann_data
        |Function| providing the Neumann boundary values.
    dirichlet_data
        |Function| providing the Dirichlet boundary values.
    diffusion
        Diffusion scalar.
    order
        Order of the Gauss quadrature to use for numerical integration.
    name
        The name of the functional.
    """

    range = NumpyVectorSpace(1)
    sparse = False

    def __init__(self, grid, boundary_info, function, neumann_data, dirichlet_data, diffusion, order=2, name=None):
        assert function.shape_range == tuple()
        super(DiffusionRHSOperatorFunctional, self).__init__()
        self.source = NumpyVectorSpace(grid.size(0))
        self.grid = grid
        self.boundary_info = boundary_info
        self.g_neumann = neumann_data
        self.g_dirichlet = dirichlet_data
        self.function = function
        self.diffusion = diffusion
        self.order = order
        self.name = name
        self.build_parameter_type(inherits=(function,))

    def _assemble(self, mu=None):
        mu = self.parse_parameter(mu)
        grid = self.grid
        assert isinstance(grid, AffineGridWithOrthogonalCentersInterface)
        boundary_info = self.boundary_info
        g_neumann = self.g_neumann
        g_dirichlet = self.g_dirichlet
        diffusion = self.diffusion
        centers = grid.centers(1)
        orthogonal_centers = grid.orthogonal_centers()
        boundary_mask = grid.boundary_mask(1)
        VOLS = grid.volumes(1)
        DVOLS = np.zeros(VOLS.size)

        SE_I0 = grid.superentities(1, 0)[:, 0]
        SE_I1 = grid.superentities(1, 0)[:, 1]

        SE_I0_I = SE_I0[~boundary_mask]
        SE_I1_I = SE_I1[~boundary_mask]

        DVOLS[~boundary_mask] = np.abs(orthogonal_centers[SE_I0_I, 0] - orthogonal_centers[SE_I1_I, 0])
        DVOLS[~boundary_mask] += np.abs(orthogonal_centers[SE_I0_I, 1] - orthogonal_centers[SE_I1_I, 1])

        # compute shift for periodic boundaries
        embeddings = grid.embeddings(0)
        reference_element = grid.reference_element(0)
        sub_reference_element_centers = reference_element.sub_reference_element(1).center()
        subentity_embedding = reference_element.subentity_embedding(1)
        se_w_i = grid._superentities_with_indices(1, 0)
        x = subentity_embedding[0][se_w_i[1][:, 0], :, 0] * sub_reference_element_centers + subentity_embedding[1][se_w_i[1][:, 0]]
        y = subentity_embedding[0][se_w_i[1][:, 1], :, 0] * sub_reference_element_centers + subentity_embedding[1][se_w_i[1][:, 1]]
        for i in xrange(len(x)):
            x[i] = np.dot(embeddings[0][se_w_i[0][i, 0]],x[i]) + embeddings[1][se_w_i[0][i, 0]]
            y[i] = np.dot(embeddings[0][se_w_i[0][i, 1]],y[i]) + embeddings[1][se_w_i[0][i, 1]]
        DVOLS[~boundary_mask] += (x[:, 0]-y[:, 0])[~boundary_mask]
        DVOLS[~boundary_mask] += (x[:, 1]-y[:, 1])[~boundary_mask]
        DVOLS = np.abs(DVOLS)

        SE_I0_B = SE_I0[boundary_mask]
        DVOLS[boundary_mask] = np.abs(centers[boundary_mask, 0] - orthogonal_centers[SE_I0_B, 0])
        DVOLS[boundary_mask] += np.abs(centers[boundary_mask, 1] - orthogonal_centers[SE_I0_B, 1])

        # evaluate function at all quadrature points -> shape = (g.size(0), number of quadrature points, 1)
        F = self.function(grid.quadrature_points(0, order=self.order), mu=mu)

        _, w = grid.reference_element.quadrature(order=self.order)

        # integrate the products of the function with the shape functions on each element
        # -> shape = (g.size(0), number of shape functions)
        F_INTS = np.einsum('ei,e,i->e', F, grid.integration_elements(0), w).ravel()
        F_INTS /= grid.volumes(0)
        F_INTS = F_INTS.reshape((1, -1))

        if boundary_info.has_dirichlet or boundary_info.has_neumann:
            SE_INTS = np.zeros(grid.size(1))

            if boundary_info.has_dirichlet:
                dirichlet_mask = boundary_info.dirichlet_mask(1)
                SE_INTS[dirichlet_mask] = diffusion * VOLS[dirichlet_mask] * g_dirichlet(centers[dirichlet_mask]).ravel() / DVOLS[dirichlet_mask]

            if boundary_info.has_neumann:
                neumann_mask = boundary_info.neumann_mask(1)
                SE_INTS[neumann_mask] += g_neumann(centers[neumann_mask]).ravel()

            BV = np.bincount(SE_I0, weights=SE_INTS)
            BV /= grid.volumes(0)
            F_INTS += BV

        return F_INTS.reshape((1, -1))
