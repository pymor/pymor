# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""This module provides some operators for finite volume discretizations."""
from functools import partial

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, dia_matrix

from pymor.algorithms.preassemble import preassemble as preassemble_
from pymor.algorithms.timestepping import ExplicitEulerTimeStepper, ImplicitEulerTimeStepper
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import Function, LincombFunction
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.core.base import abstractmethod
from pymor.core.defaults import defaults
from pymor.discretizers.builtin.domaindiscretizers.default import discretize_domain_default
from pymor.discretizers.builtin.grids.interfaces import GridWithOrthogonalCenters
from pymor.discretizers.builtin.grids.referenceelements import line, triangle, square
from pymor.discretizers.builtin.grids.subgrid import SubGrid, make_sub_grid_boundary_info
from pymor.discretizers.builtin.gui.visualizers import PatchVisualizer, OnedVisualizer
from pymor.discretizers.builtin.quadratures import GaussQuadratures
from pymor.models.basic import StationaryModel, InstationaryModel
from pymor.operators.constructions import ComponentProjectionOperator, LincombOperator, ZeroOperator
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyGenericOperator, NumpyMatrixBasedOperator, NumpyMatrixOperator
from pymor.parameters.base import ParametricObject
from pymor.vectorarrays.numpy import NumpyVectorSpace


def FVVectorSpace(grid, id='STATE'):
    return NumpyVectorSpace(grid.size(0), id)


class NumericalConvectiveFlux(ParametricObject):
    """Interface for numerical convective fluxes for finite volume schemes.

    Numerical fluxes defined by this interfaces are functions of
    the form `F(U_inner, U_outer, unit_outer_normal, edge_volume, mu)`.

    The flux evaluation is vectorized and happens in two stages:
      1. `evaluate_stage1` receives a |NumPy array| `U` of all values which
         appear as `U_inner` or `U_outer` for all edges the flux shall be
         evaluated at and returns a `tuple` of |NumPy arrays|
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


class LaxFriedrichsFlux(NumericalConvectiveFlux):
    """Lax-Friedrichs numerical flux.

    If `f` is the analytical flux, the Lax-Friedrichs flux `F` is given
    by::

      F(U_in, U_out, normal, vol) = vol * [normal⋅(f(U_in) + f(U_out))/2 + (U_in - U_out)/(2*λ)]

    Parameters
    ----------
    flux
        |Function| defining the analytical flux `f`.
    lxf_lambda
        The stabilization parameter `λ`.
    """

    def __init__(self, flux, lxf_lambda=1.0):
        self.__auto_init(locals())

    def evaluate_stage1(self, U, mu=None):
        return U, self.flux(U[..., np.newaxis], mu)

    def evaluate_stage2(self, stage1_data, unit_outer_normals, volumes, mu=None):
        U, F = stage1_data
        return (np.sum(np.sum(F, axis=1) * unit_outer_normals, axis=1) * 0.5
                + (U[..., 0] - U[..., 1]) * (0.5 / self.lxf_lambda)) * volumes


class SimplifiedEngquistOsherFlux(NumericalConvectiveFlux):
    """Engquist-Osher numerical flux. Simplified Implementation for special case.

    For the definition of the Engquist-Osher flux see :class:`EngquistOsherFlux`.
    This class provides a faster and more accurate implementation for the special
    case that `f(0) == 0` and the derivative of `f` only changes sign at `0`.

    Parameters
    ----------
    flux
        |Function| defining the analytical flux `f`.
    flux_derivative
        |Function| defining the analytical flux derivative `f'`.
    """

    def __init__(self, flux, flux_derivative):
        self.__auto_init(locals())

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


class EngquistOsherFlux(NumericalConvectiveFlux):
    """Engquist-Osher numerical flux.

    If `f` is the analytical flux, and `f'` its derivative, the Engquist-Osher flux is
    given by::

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
        self.__auto_init(locals())
        points, weights = GaussQuadratures.quadrature(npoints=self.gausspoints)
        points = points / intervals
        points = ((np.arange(self.intervals, dtype=np.float64)[:, np.newaxis] * (1 / intervals))
                  + points[np.newaxis, :]).ravel()
        weights = np.tile(weights, intervals) * (1 / intervals)
        self.points = points
        self.weights = weights

    def evaluate_stage1(self, U, mu=None):
        int_els = np.abs(U)[:, np.newaxis, np.newaxis]
        return [np.concatenate([self.flux_derivative(U[:, np.newaxis] * p, mu)[:, np.newaxis, :] * int_els * w
                               for p, w in zip(self.points, self.weights)], axis=1)]

    def evaluate_stage2(self, stage1_data, unit_outer_normals, volumes, mu=None):
        F0 = np.sum(self.flux.evaluate(np.array([[0.]]), mu=mu) * unit_outer_normals, axis=1)
        Fs = np.sum(stage1_data[0] * unit_outer_normals[:, np.newaxis, np.newaxis, :], axis=3)
        Fs[:, 0, :] = np.maximum(Fs[:, 0, :], 0)
        Fs[:, 1, :] = np.minimum(Fs[:, 1, :], 0)
        Fs = np.sum(np.sum(Fs, axis=2), axis=1) + F0
        Fs *= volumes
        return Fs


@defaults('delta')
def jacobian_options(delta=1e-7):
    return {'delta': delta}


class NonlinearAdvectionOperator(Operator):
    """Nonlinear finite volume advection |Operator|.

    The operator is of the form ::

        L(u, mu)(x) = ∇ ⋅ f(u(x), mu)

    Parameters
    ----------
    grid
        |Grid| for which to evaluate the operator.
    boundary_info
        |BoundaryInfo| determining the Dirichlet and Neumann boundaries.
    numerical_flux
        The :class:`NumericalConvectiveFlux <NumericalConvectiveFlux>` to use.
    dirichlet_data
        |Function| providing the Dirichlet boundary values. If `None`, constant-zero
        boundary is assumed.
    solver_options
        The |solver_options| for the operator.
    name
        The name of the operator.
    """

    linear = False

    def __init__(self, grid, boundary_info, numerical_flux, dirichlet_data=None, solver_options=None,
                 space_id='STATE', name=None):
        assert dirichlet_data is None or isinstance(dirichlet_data, Function)

        self.__auto_init(locals())
        if (isinstance(dirichlet_data, Function) and boundary_info.has_dirichlet
                and not dirichlet_data.parametric):
            self._dirichlet_values = self.dirichlet_data(grid.centers(1)[boundary_info.dirichlet_boundaries(1)])
            self._dirichlet_values = self._dirichlet_values.ravel()
            self._dirichlet_values_flux_shaped = self._dirichlet_values.reshape((-1, 1))
        self.source = self.range = FVVectorSpace(grid, space_id)

    def with_numerical_flux(self, **kwargs):
        return self.with_(numerical_flux=self.numerical_flux.with_(**kwargs))

    def restricted(self, dofs):
        source_dofs = np.setdiff1d(np.union1d(self.grid.neighbours(0, 0)[dofs].ravel(), dofs),
                                   np.array([-1], dtype=np.int32),
                                   assume_unique=True)
        sub_grid = SubGrid(self.grid, source_dofs)
        sub_boundary_info = make_sub_grid_boundary_info(sub_grid, self.grid, self.boundary_info)
        op = self.with_(grid=sub_grid, boundary_info=sub_boundary_info, space_id=None,
                        name=f'{self.name}_restricted')
        sub_grid_indices = sub_grid.indices_from_parent_indices(dofs, codim=0)
        proj = ComponentProjectionOperator(sub_grid_indices, op.range)
        return proj @ op, sub_grid.parent_indices(0)

    def _fetch_grid_data(self):
        # pre-fetch all grid-associated data to avoid searching the cache for each operator
        # application
        g = self.grid
        bi = self.boundary_info
        self._grid_data = dict(SUPE=g.superentities(1, 0),
                               SUPI=g.superentity_indices(1, 0),
                               VOLS0=g.volumes(0),
                               VOLS1=g.volumes(1),
                               BOUNDARIES=g.boundaries(1),
                               CENTERS=g.centers(1),
                               DIRICHLET_BOUNDARIES=bi.dirichlet_boundaries(1) if bi.has_dirichlet else None,
                               NEUMANN_BOUNDARIES=bi.neumann_boundaries(1) if bi.has_neumann else None)
        self._grid_data.update(UNIT_OUTER_NORMALS=g.unit_outer_normals()[self._grid_data['SUPE'][:, 0],
                                                                         self._grid_data['SUPI'][:, 0]])

    def apply(self, U, mu=None):
        assert U in self.source
        assert self.parameters.assert_compatible(mu)

        if not hasattr(self, '_grid_data'):
            self._fetch_grid_data()

        U = U.to_numpy()
        R = np.zeros((len(U), self.source.dim + 1))

        bi = self.boundary_info
        gd = self._grid_data
        SUPE = gd['SUPE']
        VOLS0 = gd['VOLS0']
        VOLS1 = gd['VOLS1']
        BOUNDARIES = gd['BOUNDARIES']
        CENTERS = gd['CENTERS']
        DIRICHLET_BOUNDARIES = gd['DIRICHLET_BOUNDARIES']
        NEUMANN_BOUNDARIES = gd['NEUMANN_BOUNDARIES']
        UNIT_OUTER_NORMALS = gd['UNIT_OUTER_NORMALS']

        if bi.has_dirichlet:
            if hasattr(self, '_dirichlet_values'):
                dirichlet_values = self._dirichlet_values
            elif self.dirichlet_data is not None:
                dirichlet_values = self.dirichlet_data(CENTERS[DIRICHLET_BOUNDARIES], mu=mu)
            else:
                dirichlet_values = np.zeros_like(DIRICHLET_BOUNDARIES)
            F_dirichlet = self.numerical_flux.evaluate_stage1(dirichlet_values, mu)

        for i, j in enumerate(range(len(U))):
            Ui = U[j]
            Ri = R[i]

            F = self.numerical_flux.evaluate_stage1(Ui, mu)
            F_edge = [f[SUPE] for f in F]

            for f in F_edge:
                f[BOUNDARIES, 1] = f[BOUNDARIES, 0]
            if bi.has_dirichlet:
                for f, f_d in zip(F_edge, F_dirichlet):
                    f[DIRICHLET_BOUNDARIES, 1] = f_d

            NUM_FLUX = self.numerical_flux.evaluate_stage2(F_edge, UNIT_OUTER_NORMALS, VOLS1, mu)

            if bi.has_neumann:
                NUM_FLUX[NEUMANN_BOUNDARIES] = 0

            np.add.at(Ri, SUPE[:, 0], NUM_FLUX)
            np.subtract.at(Ri, SUPE[:, 1], NUM_FLUX)

        R[:, :-1] /= VOLS0

        return self.range.make_array(R[:, :-1])

    def jacobian(self, U, mu=None):
        assert U in self.source and len(U) == 1
        assert self.parameters.assert_compatible(mu)

        if not hasattr(self, '_grid_data'):
            self._fetch_grid_data()

        U = U.to_numpy().ravel()

        g = self.grid
        bi = self.boundary_info
        gd = self._grid_data
        SUPE = gd['SUPE']
        VOLS0 = gd['VOLS0']
        VOLS1 = gd['VOLS1']
        BOUNDARIES = gd['BOUNDARIES']
        CENTERS = gd['CENTERS']
        DIRICHLET_BOUNDARIES = gd['DIRICHLET_BOUNDARIES']
        NEUMANN_BOUNDARIES = gd['NEUMANN_BOUNDARIES']
        UNIT_OUTER_NORMALS = gd['UNIT_OUTER_NORMALS']
        INNER = np.setdiff1d(np.arange(g.size(1)), BOUNDARIES)

        solver_options = self.solver_options
        delta = solver_options.get('jacobian_delta') if solver_options else None
        if delta is None:
            delta = jacobian_options()['delta']

        if bi.has_dirichlet:
            if hasattr(self, '_dirichlet_values'):
                dirichlet_values = self._dirichlet_values
            elif self.dirichlet_data is not None:
                dirichlet_values = self.dirichlet_data(CENTERS[DIRICHLET_BOUNDARIES], mu=mu)
            else:
                dirichlet_values = np.zeros_like(DIRICHLET_BOUNDARIES)
            F_dirichlet = self.numerical_flux.evaluate_stage1(dirichlet_values, mu)

        UP = U + delta
        UM = U - delta
        F = self.numerical_flux.evaluate_stage1(U, mu)
        FP = self.numerical_flux.evaluate_stage1(UP, mu)
        FM = self.numerical_flux.evaluate_stage1(UM, mu)
        del UP, UM

        F_edge = [f[SUPE] for f in F]
        FP_edge = [f[SUPE] for f in FP]
        FM_edge = [f[SUPE] for f in FM]
        del F, FP, FM

        F0P_edge = [f.copy() for f in F_edge]
        for f, ff in zip(F0P_edge, FP_edge):
            f[:, 0] = ff[:, 0]
            f[BOUNDARIES, 1] = f[BOUNDARIES, 0]
        if bi.has_dirichlet:
            for f, f_d in zip(F0P_edge, F_dirichlet):
                f[DIRICHLET_BOUNDARIES, 1] = f_d
        NUM_FLUX_0P = self.numerical_flux.evaluate_stage2(F0P_edge, UNIT_OUTER_NORMALS, VOLS1, mu)
        del F0P_edge

        F0M_edge = [f.copy() for f in F_edge]
        for f, ff in zip(F0M_edge, FM_edge):
            f[:, 0] = ff[:, 0]
            f[BOUNDARIES, 1] = f[BOUNDARIES, 0]
        if bi.has_dirichlet:
            for f, f_d in zip(F0M_edge, F_dirichlet):
                f[DIRICHLET_BOUNDARIES, 1] = f_d
        NUM_FLUX_0M = self.numerical_flux.evaluate_stage2(F0M_edge, UNIT_OUTER_NORMALS, VOLS1, mu)
        del F0M_edge

        D_NUM_FLUX_0 = (NUM_FLUX_0P - NUM_FLUX_0M)
        D_NUM_FLUX_0 /= (2 * delta)
        if bi.has_neumann:
            D_NUM_FLUX_0[NEUMANN_BOUNDARIES] = 0
        del NUM_FLUX_0P, NUM_FLUX_0M

        F1P_edge = [f.copy() for f in F_edge]
        for f, ff in zip(F1P_edge, FP_edge):
            f[:, 1] = ff[:, 1]
            f[BOUNDARIES, 1] = f[BOUNDARIES, 0]
        if bi.has_dirichlet:
            for f, f_d in zip(F1P_edge, F_dirichlet):
                f[DIRICHLET_BOUNDARIES, 1] = f_d
        NUM_FLUX_1P = self.numerical_flux.evaluate_stage2(F1P_edge, UNIT_OUTER_NORMALS, VOLS1, mu)
        del F1P_edge, FP_edge

        F1M_edge = F_edge
        for f, ff in zip(F1M_edge, FM_edge):
            f[:, 1] = ff[:, 1]
            f[BOUNDARIES, 1] = f[BOUNDARIES, 0]
        if bi.has_dirichlet:
            for f, f_d in zip(F1M_edge, F_dirichlet):
                f[DIRICHLET_BOUNDARIES, 1] = f_d
        NUM_FLUX_1M = self.numerical_flux.evaluate_stage2(F1M_edge, UNIT_OUTER_NORMALS, VOLS1, mu)
        del F1M_edge, FM_edge
        D_NUM_FLUX_1 = (NUM_FLUX_1P - NUM_FLUX_1M)
        D_NUM_FLUX_1 /= (2 * delta)
        if bi.has_neumann:
            D_NUM_FLUX_1[NEUMANN_BOUNDARIES] = 0
        del NUM_FLUX_1P, NUM_FLUX_1M

        I1 = np.hstack([SUPE[INNER, 0], SUPE[INNER, 0], SUPE[INNER, 1], SUPE[INNER, 1], SUPE[BOUNDARIES, 0]])
        I0 = np.hstack([SUPE[INNER, 0], SUPE[INNER, 1], SUPE[INNER, 0], SUPE[INNER, 1], SUPE[BOUNDARIES, 0]])
        V = np.hstack([D_NUM_FLUX_0[INNER], -D_NUM_FLUX_0[INNER], D_NUM_FLUX_1[INNER], -D_NUM_FLUX_1[INNER],
                       D_NUM_FLUX_0[BOUNDARIES]])

        A = coo_matrix((V, (I0, I1)), shape=(g.size(0), g.size(0)))
        A = csc_matrix(A).copy()   # See pymor.discretizers.builtin.cg.DiffusionOperatorP1 for why copy() is necessary
        A = dia_matrix(([1. / VOLS0], [0]), shape=(g.size(0),) * 2) * A

        return NumpyMatrixOperator(A, source_id=self.source.id, range_id=self.range.id)


def nonlinear_advection_lax_friedrichs_operator(grid, boundary_info, flux, lxf_lambda=1.0,
                                                dirichlet_data=None, solver_options=None, name=None):
    """Instantiate a :class:`NonlinearAdvectionOperator` using :class:`LaxFriedrichsFlux`."""
    num_flux = LaxFriedrichsFlux(flux, lxf_lambda)
    return NonlinearAdvectionOperator(grid, boundary_info, num_flux, dirichlet_data, solver_options, name=name)


def nonlinear_advection_simplified_engquist_osher_operator(grid, boundary_info, flux, flux_derivative,
                                                           dirichlet_data=None, solver_options=None, name=None):
    """Create a :class:`NonlinearAdvectionOperator` using :class:`SimplifiedEngquistOsherFlux`."""
    num_flux = SimplifiedEngquistOsherFlux(flux, flux_derivative)
    return NonlinearAdvectionOperator(grid, boundary_info, num_flux, dirichlet_data, solver_options, name=name)


def nonlinear_advection_engquist_osher_operator(grid, boundary_info, flux, flux_derivative, gausspoints=5, intervals=1,
                                                dirichlet_data=None, solver_options=None, name=None):
    """Instantiate a :class:`NonlinearAdvectionOperator` using :class:`EngquistOsherFlux`."""
    num_flux = EngquistOsherFlux(flux, flux_derivative, gausspoints=gausspoints, intervals=intervals)
    return NonlinearAdvectionOperator(grid, boundary_info, num_flux, dirichlet_data, solver_options, name=name)


class LinearAdvectionLaxFriedrichsOperator(NumpyMatrixBasedOperator):
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
    solver_options
        The |solver_options| for the operator.
    name
        The name of the operator.
    """

    def __init__(self, grid, boundary_info, velocity_field, lxf_lambda=1.0, solver_options=None, name=None):
        self.__auto_init(locals())
        self.source = self.range = FVVectorSpace(grid)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info
        SUPE = g.superentities(1, 0)
        SUPI = g.superentity_indices(1, 0)
        assert SUPE.ndim == 2
        edge_volumes = g.volumes(1)
        boundary_edges = g.boundaries(1)
        inner_edges = np.setdiff1d(np.arange(g.size(1)), boundary_edges)
        dirichlet_edges = bi.dirichlet_boundaries(1) if bi.has_dirichlet else np.array([], ndmin=1, dtype=int)
        neumann_edges = bi.neumann_boundaries(1) if bi.has_neumann else np.array([], ndmin=1, dtype=int)
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
        V_dir = edge_volumes[dirichlet_edges] * (0.5 * normal_velocities[dirichlet_edges] + 0.5 / self.lxf_lambda)

        I0 = np.hstack([I0_inner, I_out, I_dir])
        I1 = np.hstack([I1_inner, I_out, I_dir])
        V = np.hstack([V_inner, V_out, V_dir])

        A = coo_matrix((V, (I0, I1)), shape=(g.size(0), g.size(0)))
        A = csc_matrix(A).copy()   # See pymor.discretizers.builtin.cg.DiffusionOperatorP1 for why copy() is necessary
        A = dia_matrix(([1. / g.volumes(0)], [0]), shape=(g.size(0),) * 2) * A

        return A


class L2Product(NumpyMatrixBasedOperator):
    """|Operator| representing the L2-product between finite volume functions.

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the product.
    solver_options
        The |solver_options| for the operator.
    name
        The name of the product.
    """

    sparse = True

    def __init__(self, grid, solver_options=None, name=None):
        self.__auto_init(locals())
        self.source = self.range = FVVectorSpace(grid)

    def _assemble(self, mu=None):

        A = dia_matrix((self.grid.volumes(0), [0]), shape=(self.grid.size(0),) * 2)

        return A


class ReactionOperator(NumpyMatrixBasedOperator):
    """Finite Volume reaction |Operator|.

    The operator is of the form ::

        L(u, mu)(x) = c(x, mu)⋅u(x)

    Parameters
    ----------
    grid
        The |Grid| for which to assemble the operator.
    reaction_coefficient
        The function 'c'
    solver_options
        The |solver_options| for the operator.
    name
        The name of the operator.
    """

    sparse = True

    def __init__(self, grid, reaction_coefficient, solver_options=None, name=None):
        assert reaction_coefficient.dim_domain == grid.dim and reaction_coefficient.shape_range == ()
        self.__auto_init(locals())
        self.source = self.range = FVVectorSpace(grid)

    def _assemble(self, mu=None):

        A = dia_matrix((self.reaction_coefficient.evaluate(self.grid.centers(0), mu=mu), [0]),
                       shape=(self.grid.size(0),) * 2)

        return A


class NonlinearReactionOperator(Operator):

    linear = False

    def __init__(self, grid, reaction_function, reaction_function_derivative=None, space_id='STATE', name=None):
        self.__auto_init(locals())
        self.source = self.range = FVVectorSpace(grid, space_id)

    def apply(self, U, ind=None, mu=None):
        assert U in self.source

        R = U.to_numpy() if ind is None else U.to_numpy()[ind]
        R = self.reaction_function.evaluate(R.reshape(R.shape + (1,)), mu=mu)

        return self.range.make_array(R)

    def jacobian(self, U, mu=None):
        if self.reaction_function_derivative is None:
            raise NotImplementedError

        U = U.to_numpy()
        A = dia_matrix((self.reaction_function_derivative.evaluate(U.reshape(U.shape + (1,)), mu=mu), [0]),
                       shape=(self.grid.size(0),) * 2)

        return NumpyMatrixOperator(A, source_id=self.source.id, range_id=self.range.id)


class L2Functional(NumpyMatrixBasedOperator):
    """Finite volume functional representing the inner product with an L2-|Function|.

    Additionally, boundary conditions can be enforced by providing `dirichlet_data`
    and `neumann_data` functions.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    function
        The |Function| with which to take the inner product or `None`.
    boundary_info
        |BoundaryInfo| determining the Dirichlet and Neumann boundaries or `None`.
        If `None`, no boundary treatment is performed.
    dirichlet_data
        |Function| providing the Dirichlet boundary values. If `None`,
        constant-zero boundary is assumed.
    diffusion_function
        See :class:`DiffusionOperator`. Has to be specified in case `dirichlet_data`
        is given.
    diffusion_constant
        See :class:`DiffusionOperator`. Has to be specified in case `dirichlet_data`
        is given.
    neumann_data
        |Function| providing the Neumann boundary values. If `None`,
        constant-zero is assumed.
    order
        Order of the Gauss quadrature to use for numerical integration.
    name
        The name of the functional.
    """

    source = NumpyVectorSpace(1)
    sparse = False

    def __init__(self, grid, function=None, boundary_info=None, dirichlet_data=None, diffusion_function=None,
                 diffusion_constant=None, neumann_data=None, order=1, name=None):
        assert function is None or function.shape_range == ()
        self.__auto_init(locals())
        self.range = FVVectorSpace(grid)

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        if self.function is not None:
            # evaluate function at all quadrature points
            # -> shape = (g.size(0), number of quadrature points, 1)
            F = self.function(g.quadrature_points(0, order=self.order), mu=mu)

            _, w = g.reference_element.quadrature(order=self.order)

            # integrate the products of the function with the shape functions on each element
            # -> shape = (g.size(0), number of shape functions)
            F_INTS = np.einsum('ei,e,i->e', F, g.integration_elements(0), w).ravel()
        else:
            F_INTS = np.zeros(g.size(0))

        if bi is not None and (bi.has_dirichlet and self.dirichlet_data is not None
                               or bi.has_neumann and self.neumann_data):
            centers = g.centers(1)
            superentities = g.superentities(1, 0)
            superentity_indices = g.superentity_indices(1, 0)
            SE_I0 = superentities[:, 0]
            VOLS = g.volumes(1)
            FLUXES = np.zeros(g.size(1))

            if bi.has_dirichlet and self.dirichlet_data is not None:
                dirichlet_mask = bi.dirichlet_mask(1)
                SE_I0_D = SE_I0[dirichlet_mask]
                boundary_normals = g.unit_outer_normals()[SE_I0_D, superentity_indices[:, 0][dirichlet_mask]]
                BOUNDARY_DISTS = np.sum((centers[dirichlet_mask, :] - g.orthogonal_centers()[SE_I0_D, :])
                                        * boundary_normals,
                                        axis=-1)
                DIRICHLET_FLUXES = (VOLS[dirichlet_mask]
                                    * self.dirichlet_data(centers[dirichlet_mask], mu=mu) / BOUNDARY_DISTS)
                if self.diffusion_function is not None:
                    DIRICHLET_FLUXES *= self.diffusion_function(centers[dirichlet_mask], mu=mu)
                if self.diffusion_constant is not None:
                    DIRICHLET_FLUXES *= self.diffusion_constant
                FLUXES[dirichlet_mask] = DIRICHLET_FLUXES

            if bi.has_neumann and self.neumann_data is not None:
                neumann_mask = bi.neumann_mask(1)
                FLUXES[neumann_mask] -= VOLS[neumann_mask] * self.neumann_data(centers[neumann_mask], mu=mu)

            F_INTS += np.bincount(SE_I0, weights=FLUXES, minlength=len(F_INTS))

        F_INTS /= g.volumes(0)

        return F_INTS.reshape((-1, 1))


class BoundaryL2Functional(NumpyMatrixBasedOperator):
    """FV functional representing the inner product with an L2-|Function| on the boundary.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    function
        The |Function| with which to take the inner product.
    boundary_type
        The type of domain boundary (e.g. 'neumann') on which to assemble the functional.
        If `None` the functional is assembled over the whole boundary.
    boundary_info
        If `boundary_type` is specified, the
        |BoundaryInfo| determining which boundary entity belongs to which physical boundary.
    name
        The name of the functional.
    """

    sparse = False
    source = NumpyVectorSpace(1)

    def __init__(self, grid, function, boundary_type=None, boundary_info=None, name=None):
        assert grid.reference_element(0) in {line, triangle, square}
        assert function.shape_range == ()
        assert not boundary_type or boundary_info
        self.__auto_init(locals())
        self.range = FVVectorSpace(grid)

    def _assemble(self, mu=None):
        g = self.grid

        NI = self.boundary_info.boundaries(self.boundary_type, 1) if self.boundary_type else g.boundaries(1)
        SUPE = g.superentities(1, 0)[NI][:, 0]
        F = self.function(g.centers(1)[NI], mu=mu)
        if g.dim == 1:
            I = np.zeros(self.range.dim)
            I[SUPE] = F
        else:
            I = F*g.integration_elements(1)[NI]
            I = coo_matrix((I, (np.zeros_like(SUPE), SUPE)), shape=(1, g.size(0))).toarray().ravel()

        return I.reshape((-1, 1))


class DiffusionOperator(NumpyMatrixBasedOperator):
    """Finite Volume Diffusion |Operator|.

    The operator is of the form ::

        (Lu)(x) = c ∇ ⋅ [ d(x) ∇ u(x) ]

    Parameters
    ----------
    grid
        The |Grid| over which to assemble the operator.
    boundary_info
        |BoundaryInfo| for the treatment of Dirichlet boundary conditions.
    diffusion_function
        The scalar-valued |Function| `d(x)`. If `None`, constant one is assumed.
    diffusion_constant
        The constant `c`. If `None`, `c` is set to one.
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    sparse = True

    def __init__(self, grid, boundary_info, diffusion_function=None, diffusion_constant=None, solver_options=None,
                 name=None):
        super().__init__()
        assert isinstance(grid, GridWithOrthogonalCenters)
        assert (diffusion_function is None
                or (isinstance(diffusion_function, Function)
                    and diffusion_function.dim_domain == grid.dim
                    and diffusion_function.shape_range == ()))
        self.__auto_init(locals())
        self.source = self.range = FVVectorSpace(grid)

    def _assemble(self, mu=None):
        grid = self.grid

        # compute the local coordinates of the codim-1 subentity centers in the reference element
        reference_element = grid.reference_element(0)
        subentity_embedding = reference_element.subentity_embedding(1)
        subentity_centers = (np.einsum('eij,j->ei',
                                       subentity_embedding[0], reference_element.sub_reference_element(1).center())
                             + subentity_embedding[1])

        # compute shift for periodic boundaries
        embeddings = grid.embeddings(0)
        superentities = grid.superentities(1, 0)
        superentity_indices = grid.superentity_indices(1, 0)
        boundary_mask = grid.boundary_mask(1)
        inner_mask = ~boundary_mask
        SE_I0 = superentities[:, 0]
        SE_I1 = superentities[:, 1]
        SE_I0_I = SE_I0[inner_mask]
        SE_I1_I = SE_I1[inner_mask]

        SHIFTS = (np.einsum('eij,ej->ei',
                            embeddings[0][SE_I0_I, :, :],
                            subentity_centers[superentity_indices[:, 0][inner_mask]])
                  + embeddings[1][SE_I0_I, :])
        SHIFTS -= (np.einsum('eij,ej->ei',
                             embeddings[0][SE_I1_I, :, :],
                             subentity_centers[superentity_indices[:, 1][inner_mask]])
                   + embeddings[1][SE_I1_I, :])

        # comute distances for gradient approximations
        centers = grid.centers(1)
        orthogonal_centers = grid.orthogonal_centers()
        VOLS = grid.volumes(1)

        INNER_DISTS = np.linalg.norm(orthogonal_centers[SE_I0_I, :] - orthogonal_centers[SE_I1_I, :] - SHIFTS,
                                     axis=1)
        del SHIFTS

        # assemble matrix
        FLUXES = VOLS[inner_mask] / INNER_DISTS
        if self.diffusion_function is not None:
            FLUXES *= self.diffusion_function(centers[inner_mask], mu=mu)
        if self.diffusion_constant is not None:
            FLUXES *= self.diffusion_constant
        del INNER_DISTS

        FLUXES = np.concatenate((-FLUXES, -FLUXES, FLUXES, FLUXES))
        FLUXES_I0 = np.concatenate((SE_I0_I, SE_I1_I, SE_I0_I, SE_I1_I))
        FLUXES_I1 = np.concatenate((SE_I1_I, SE_I0_I, SE_I0_I, SE_I1_I))

        if self.boundary_info.has_dirichlet:
            dirichlet_mask = self.boundary_info.dirichlet_mask(1)
            SE_I0_D = SE_I0[dirichlet_mask]
            boundary_normals = grid.unit_outer_normals()[SE_I0_D, superentity_indices[:, 0][dirichlet_mask]]
            BOUNDARY_DISTS = np.sum((centers[dirichlet_mask, :] - orthogonal_centers[SE_I0_D, :]) * boundary_normals,
                                    axis=-1)

            DIRICHLET_FLUXES = VOLS[dirichlet_mask] / BOUNDARY_DISTS
            if self.diffusion_function is not None:
                DIRICHLET_FLUXES *= self.diffusion_function(centers[dirichlet_mask], mu=mu)
            if self.diffusion_constant is not None:
                DIRICHLET_FLUXES *= self.diffusion_constant

            FLUXES = np.concatenate((FLUXES, DIRICHLET_FLUXES))
            FLUXES_I0 = np.concatenate((FLUXES_I0, SE_I0_D))
            FLUXES_I1 = np.concatenate((FLUXES_I1, SE_I0_D))

        A = coo_matrix((FLUXES, (FLUXES_I0, FLUXES_I1)), shape=(self.source.dim, self.source.dim))
        A = (dia_matrix(([1. / grid.volumes(0)], [0]), shape=(grid.size(0),) * 2) * A).tocsc()

        return A


class InterpolationOperator(NumpyMatrixBasedOperator):
    """Vector-like L^2-projection interpolation |Operator| for finite volume spaces.

    Parameters
    ----------
    grid
        The |Grid| on which to interpolate.
    function
        The |Function| to interpolate.
    order
        The quadrature order to compute the element-wise averages
    """

    source = NumpyVectorSpace(1)
    linear = True

    def __init__(self, grid, function, order=0):
        assert function.dim_domain == grid.dim
        assert function.shape_range == ()
        if order != 0:
            raise NotImplementedError('Higher orders not implemented yet!')
        self.__auto_init(locals())
        self.range = FVVectorSpace(grid)

    def _assemble(self, mu=None):
        return self.function.evaluate(self.grid.centers(0), mu=mu).reshape((-1, 1))


def discretize_stationary_fv(analytical_problem, diameter=None, domain_discretizer=None, grid_type=None,
                             num_flux='lax_friedrichs', lxf_lambda=1., eo_gausspoints=5, eo_intervals=1,
                             grid=None, boundary_info=None, preassemble=True):
    """Discretizes a |StationaryProblem| using the finite volume method.

    Parameters
    ----------
    analytical_problem
        The |StationaryProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed as an argument to the `domain_discretizer`.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter, ...)`.
        If `None`, |discretize_domain_default| is used.
    grid_type
        If not `None`, this parameter is forwarded to `domain_discretizer` to specify
        the type of the generated |Grid|.
    num_flux
        The numerical flux to use in the finite volume formulation. Allowed
        values are `'lax_friedrichs'`, `'engquist_osher'`, `'simplified_engquist_osher'`
        (see :mod:`pymor.discretizers.builtin.fv`).
    lxf_lambda
        The stabilization parameter for the Lax-Friedrichs numerical flux
        (ignored, if different flux is chosen).
    eo_gausspoints
        Number of Gauss points for the Engquist-Osher numerical flux
        (ignored, if different flux is chosen).
    eo_intervals
        Number of sub-intervals to use for integration when using Engquist-Osher
        numerical flux (ignored, if different flux is chosen).
    grid
        Instead of using a domain discretizer, the |Grid| can also be passed directly
        using this parameter.
    boundary_info
        A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
        Must be provided if `grid` is specified.
    preassemble
        If `True`, preassemble all operators in the resulting |Model|.

    Returns
    -------
    m
        The |Model| that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.
            :unassembled_m:  In case `preassemble` is `True`, the generated |Model|
                             before preassembling operators.
    """
    assert isinstance(analytical_problem, StationaryProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None
    assert grid_type is None or grid is None

    p = analytical_problem

    if analytical_problem.robin_data is not None:
        raise NotImplementedError

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if grid_type:
            domain_discretizer = partial(domain_discretizer, grid_type=grid_type)
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    L, L_coefficients = [], []
    F, F_coefficients = [], []

    if p.rhs is not None or p.neumann_data is not None:
        F += [L2Functional(grid, p.rhs, boundary_info=boundary_info, neumann_data=p.neumann_data)]
        F_coefficients += [1.]

    # diffusion part
    if isinstance(p.diffusion, LincombFunction):
        L += [DiffusionOperator(grid, boundary_info, diffusion_function=df, name=f'diffusion_{i}')
              for i, df in enumerate(p.diffusion.functions)]
        L_coefficients += p.diffusion.coefficients
        if p.dirichlet_data is not None:
            F += [L2Functional(grid, None, boundary_info=boundary_info, dirichlet_data=p.dirichlet_data,
                               diffusion_function=df, name=f'dirichlet_{i}')
                  for i, df in enumerate(p.diffusion.functions)]
            F_coefficients += p.diffusion.coefficients

    elif p.diffusion is not None:
        L += [DiffusionOperator(grid, boundary_info, diffusion_function=p.diffusion, name='diffusion')]
        L_coefficients += [1.]
        if p.dirichlet_data is not None:
            F += [L2Functional(grid, None, boundary_info=boundary_info, dirichlet_data=p.dirichlet_data,
                               diffusion_function=p.diffusion, name='dirichlet')]
            F_coefficients += [1.]

    # advection part
    if isinstance(p.advection, LincombFunction):
        L += [LinearAdvectionLaxFriedrichsOperator(grid, boundary_info, af, name=f'advection_{i}')
              for i, af in enumerate(p.advection.functions)]
        L_coefficients += list(p.advection.coefficients)
    elif p.advection is not None:
        L += [LinearAdvectionLaxFriedrichsOperator(grid, boundary_info, p.advection, name='advection')]
        L_coefficients.append(1.)

    # nonlinear advection part
    if p.nonlinear_advection is not None:
        if num_flux == 'lax_friedrichs':
            L += [nonlinear_advection_lax_friedrichs_operator(grid, boundary_info, p.nonlinear_advection,
                                                              dirichlet_data=p.dirichlet_data, lxf_lambda=lxf_lambda)]
        elif num_flux == 'engquist_osher':
            L += [nonlinear_advection_engquist_osher_operator(grid, boundary_info, p.nonlinear_advection,
                                                              p.nonlinear_advection_derivative,
                                                              gausspoints=eo_gausspoints, intervals=eo_intervals,
                                                              dirichlet_data=p.dirichlet_data)]
        elif num_flux == 'simplified_engquist_osher':
            L += [nonlinear_advection_simplified_engquist_osher_operator(grid, boundary_info, p.nonlinear_advection,
                                                                         p.nonlinear_advection_derivative,
                                                                         dirichlet_data=p.dirichlet_data)]
        else:
            raise NotImplementedError
        L_coefficients.append(1.)

    # reaction part
    if isinstance(p.reaction, LincombFunction):
        L += [ReactionOperator(grid, rf, name=f'reaction_{i}')
              for i, rf in enumerate(p.reaction.functions)]
        L_coefficients += list(p.reaction.coefficients)
    elif p.reaction is not None:
        L += [ReactionOperator(grid, p.reaction, name='reaction')]
        L_coefficients += [1.]

    # nonlinear reaction part
    if p.nonlinear_reaction is not None:
        L += [NonlinearReactionOperator(grid, p.nonlinear_reaction, p.nonlinear_reaction_derivative)]
        L_coefficients += [1.]

    # system operator
    if len(L_coefficients) == 1 and L_coefficients[0] == 1.:
        L = L[0]
    else:
        L = LincombOperator(operators=L, coefficients=L_coefficients, name='elliptic_operator')

    # rhs
    if len(F_coefficients) == 0:
        F = ZeroOperator(L.range, NumpyVectorSpace(1))
    elif len(F_coefficients) == 1 and F_coefficients[0] == 1.:
        F = F[0]
    else:
        F = LincombOperator(operators=F, coefficients=F_coefficients, name='rhs')

    if grid.reference_element in (triangle, square):
        visualizer = PatchVisualizer(grid=grid, codim=0)
    elif grid.reference_element is line:
        visualizer = OnedVisualizer(grid=grid, codim=0)
    else:
        visualizer = None

    l2_product = L2Product(grid, name='l2')
    products = {'l2': l2_product}

    # assemble additional output functionals
    if p.outputs:
        if any(v[0] not in ('l2', 'l2_boundary') for v in p.outputs):
            raise NotImplementedError
        outputs = []
        for v in p.outputs:
            if v[0] == 'l2':
                if isinstance(v[1], LincombFunction):
                    ops = [L2Functional(grid, vv).H
                           for vv in v[1].functions]
                    outputs.append(LincombOperator(ops, v[1].coefficients))
                else:
                    outputs.append(L2Functional(grid, v[1]).H)
            else:
                if isinstance(v[1], LincombFunction):
                    ops = [BoundaryL2Functional(grid, vv).H
                           for vv in v[1].functions]
                    outputs.append(LincombOperator(ops, v[1].coefficients))
                else:
                    outputs.append(BoundaryL2Functional(grid, v[1]).H)
        if len(outputs) > 1:
            from pymor.operators.block import BlockColumnOperator
            from pymor.operators.constructions import NumpyConversionOperator
            output_functional = BlockColumnOperator(outputs)
            output_functional = NumpyConversionOperator(output_functional.range) @ output_functional
        else:
            output_functional = outputs[0]
    else:
        output_functional = None

    m = StationaryModel(L, F, products=products, output_functional=output_functional, visualizer=visualizer,
                        name=f'{p.name}_FV')

    data = {'grid': grid, 'boundary_info': boundary_info}

    if preassemble:
        data['unassembled_m'] = m
        m = preassemble_(m)

    return m, data


def discretize_instationary_fv(analytical_problem, diameter=None, domain_discretizer=None, grid_type=None,
                               num_flux='lax_friedrichs', lxf_lambda=1., eo_gausspoints=5, eo_intervals=1,
                               grid=None, boundary_info=None, num_values=None, time_stepper=None, nt=None,
                               preassemble=True):
    """FV Discretization of an |InstationaryProblem| with a |StationaryProblem| as stationary part

    Parameters
    ----------
    analytical_problem
        The |InstationaryProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed to the `domain_discretizer`.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter, ...)`.
        If further arguments should be passed to the discretizer, use
        :func:`functools.partial`. If `None`, |discretize_domain_default| is used.
    grid_type
        If not `None`, this parameter is forwarded to `domain_discretizer` to specify
        the type of the generated |Grid|.
    num_flux
        The numerical flux to use in the finite volume formulation. Allowed
        values are `'lax_friedrichs'`, `'engquist_osher'`, `'simplified_engquist_osher'`
        (see :mod:`pymor.discretizers.builtin.fv`).
    lxf_lambda
        The stabilization parameter for the Lax-Friedrichs numerical flux
        (ignored, if different flux is chosen).
    eo_gausspoints
        Number of Gauss points for the Engquist-Osher numerical flux
        (ignored, if different flux is chosen).
    eo_intervals
        Number of sub-intervals to use for integration when using Engquist-Osher
        numerical flux (ignored, if different flux is chosen).
    grid
        Instead of using a domain discretizer, the |Grid| can also be passed directly
        using this parameter.
    boundary_info
        A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
        Must be provided if `grid` is specified.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each
        intermediate vector that is calculated is returned.
    time_stepper
        The :class:`time-stepper <pymor.algorithms.timestepping.TimeStepper>`
        to be used by :class:`~pymor.models.basic.InstationaryModel.solve`.
    nt
        If `time_stepper` is not specified, the number of time steps for implicit
        Euler time stepping.
    preassemble
        If `True`, preassemble all operators in the resulting |Model|.

    Returns
    -------
    m
        The |Model| that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.
            :unassembled_m:  In case `preassemble` is `True`, the generated |Model|
                             before preassembling operators.
    """
    assert isinstance(analytical_problem, InstationaryProblem)
    assert isinstance(analytical_problem.stationary_part, StationaryProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None
    assert (time_stepper is None) != (nt is None)

    p = analytical_problem

    m, data = discretize_stationary_fv(p.stationary_part, diameter=diameter, domain_discretizer=domain_discretizer,
                                       grid_type=grid_type, num_flux=num_flux, lxf_lambda=lxf_lambda,
                                       eo_gausspoints=eo_gausspoints, eo_intervals=eo_intervals, grid=grid,
                                       boundary_info=boundary_info)
    grid = data['grid']

    if p.initial_data.parametric:
        def initial_projection(U, mu):
            I = p.initial_data.evaluate(grid.quadrature_points(0, order=2), mu).squeeze()
            I = np.sum(I * grid.reference_element.quadrature(order=2)[1], axis=1) * (1. / grid.reference_element.volume)
            I = m.solution_space.make_array(I)
            return I.lincomb(U).to_numpy()
        I = NumpyGenericOperator(initial_projection, dim_range=grid.size(0), linear=True, range_id=m.solution_space.id,
                                 parameters=p.initial_data.parameters)
    else:
        I = p.initial_data.evaluate(grid.quadrature_points(0, order=2)).squeeze()
        I = np.sum(I * grid.reference_element.quadrature(order=2)[1], axis=1) * (1. / grid.reference_element.volume)
        I = m.solution_space.make_array(I)

    if time_stepper is None:
        if p.stationary_part.diffusion is None:
            time_stepper = ExplicitEulerTimeStepper(nt=nt)
        else:
            time_stepper = ImplicitEulerTimeStepper(nt=nt)

    rhs = None if isinstance(m.rhs, ZeroOperator) else m.rhs

    m = InstationaryModel(operator=m.operator, rhs=rhs, mass=None, initial_data=I, T=p.T,
                          products=m.products,
                          output_functional=m.output_functional,
                          time_stepper=time_stepper,
                          visualizer=m.visualizer,
                          num_values=num_values, name=f'{p.name}_FV')

    if preassemble:
        data['unassembled_m'] = m
        m = preassemble_(m)

    return m, data
