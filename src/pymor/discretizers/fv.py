# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from functools import partial

import numpy as np

from pymor.algorithms.timestepping import ExplicitEulerTimeStepper, ImplicitEulerTimeStepper
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.algorithms.preassemble import preassemble as preassemble_
from pymor.discretizations.basic import StationaryDiscretization, InstationaryDiscretization
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.functions.basic import LincombFunction
from pymor.grids.referenceelements import line, triangle, square
from pymor.gui.visualizers import PatchVisualizer, OnedVisualizer
from pymor.operators.constructions import LincombOperator, ZeroOperator
from pymor.operators.numpy import NumpyGenericOperator
from pymor.operators.fv import (DiffusionOperator, LinearAdvectionLaxFriedrichs, ReactionOperator,
                                nonlinear_advection_lax_friedrichs_operator,
                                nonlinear_advection_engquist_osher_operator,
                                nonlinear_advection_simplified_engquist_osher_operator,
                                NonlinearReactionOperator, L2Product, L2ProductFunctional)
from pymor.vectorarrays.numpy import NumpyVectorSpace


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
        (see :mod:`pymor.operators.fv`).
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
        If `True`, preassemble all operators in the resulting |Discretization|.

    Returns
    -------
    d
        The |Discretization| that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.
            :unassembled_d:  In case `preassemble` is `True`, the generated |Discretization|
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
    if p.functionals:
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
        F += [L2ProductFunctional(grid, p.rhs, boundary_info=boundary_info, neumann_data=p.neumann_data)]
        F_coefficients += [1.]

    # diffusion part
    if isinstance(p.diffusion, LincombFunction):
        L += [DiffusionOperator(grid, boundary_info, diffusion_function=df, name='diffusion_{}'.format(i))
              for i, df in enumerate(p.diffusion.functions)]
        L_coefficients += p.diffusion.coefficients
        if p.dirichlet_data is not None:
            F += [L2ProductFunctional(grid, None, boundary_info=boundary_info, dirichlet_data=p.dirichlet_data,
                                      diffusion_function=df, name='dirichlet_{}'.format(i))
                  for i, df in enumerate(p.diffusion.functions)]
            F_coefficients += p.diffusion.coefficients

    elif p.diffusion is not None:
        L += [DiffusionOperator(grid, boundary_info, diffusion_function=p.diffusion, name='diffusion')]
        L_coefficients += [1.]
        if p.dirichlet_data is not None:
            F += [L2ProductFunctional(grid, None, boundary_info=boundary_info, dirichlet_data=p.dirichlet_data,
                                      diffusion_function=p.diffusion, name='dirichlet')]
            F_coefficients += [1.]

    # advection part
    if isinstance(p.advection, LincombFunction):
        L += [LinearAdvectionLaxFriedrichs(grid, boundary_info, af, name='advection_{}'.format(i))
              for i, af in enumerate(p.advection.functions)]
        L_coefficients += list(p.advection.coefficients)
    elif p.advection is not None:
        L += [LinearAdvectionLaxFriedrichs(grid, boundary_info, p.advection, name='advection')]
        L_coefficients.append(1.)

    # nonlinear advection part
    if p.nonlinear_advection is not None:
        if num_flux == 'lax_friedrichs':
            L += [nonlinear_advection_lax_friedrichs_operator(grid, boundary_info, p.nonlinear_advection,
                                                              dirichlet_data=p.dirichlet_data, lxf_lambda=lxf_lambda)]
        elif num_flux == 'upwind':
            L += [nonlinear_advection_upwind_operator(grid, boundary_info, p.nonlinear_advection,
                                                      p.nonlinear_advection_derivative,
                                                      dirichlet_data=p.dirichlet_data)]
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
        raise NotImplementedError
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
        visualizer = PatchVisualizer(grid=grid, bounding_box=grid.bounding_box(), codim=0)
    elif grid.reference_element is line:
        visualizer = OnedVisualizer(grid=grid, codim=0)
    else:
        visualizer = None

    l2_product = L2Product(grid, name='l2')
    products = {'l2': l2_product}

    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None

    d = StationaryDiscretization(L, F, products=products, visualizer=visualizer,
                                 parameter_space=parameter_space, name='{}_FV'.format(p.name))

    data = {'grid': grid, 'boundary_info': boundary_info}

    if preassemble:
        data['unassembled_discretization'] = d
        d = preassemble_(d)

    return d, data


def discretize_instationary_fv(analytical_problem, diameter=None, domain_discretizer=None, grid_type=None,
                               num_flux='lax_friedrichs', lxf_lambda=1., eo_gausspoints=5, eo_intervals=1,
                               grid=None, boundary_info=None, num_values=None, time_stepper=None, nt=None,
                               preassemble=True):
    """Discretizes an |InstationaryProblem| with a |StationaryProblem| as stationary part
    using the finite volume method.

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
        (see :mod:`pymor.operators.fv`).
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
        The :class:`time-stepper <pymor.algorithms.timestepping.TimeStepperInterface>`
        to be used by :class:`~pymor.discretizations.basic.InstationaryDiscretization.solve`.
    nt
        If `time_stepper` is not specified, the number of time steps for implicit
        Euler time stepping.
    preassemble
        If `True`, preassemble all operators in the resulting |Discretization|.

    Returns
    -------
    d
        The |Discretization| that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.
            :unassembled_d:  In case `preassemble` is `True`, the generated |Discretization|
                             before preassembling operators.
    """

    assert isinstance(analytical_problem, InstationaryProblem)
    assert isinstance(analytical_problem.stationary_part, StationaryProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None
    assert (time_stepper is None) != (nt is None)

    p = analytical_problem

    d, data = discretize_stationary_fv(p.stationary_part, diameter=diameter, domain_discretizer=domain_discretizer,
                                       grid_type=grid_type, num_flux=num_flux, lxf_lambda=lxf_lambda,
                                       eo_gausspoints=eo_gausspoints, eo_intervals=eo_intervals, grid=grid,
                                       boundary_info=boundary_info)
    grid = data['grid']

    if p.initial_data.parametric:
        def initial_projection(U, mu):
            I = p.initial_data.evaluate(grid.quadrature_points(0, order=2), mu).squeeze()
            I = np.sum(I * grid.reference_element.quadrature(order=2)[1], axis=1) * (1. / grid.reference_element.volume)
            I = d.solution_space.make_array(I)
            return I.lincomb(U).to_numpy()
        I = NumpyGenericOperator(initial_projection, dim_range=grid.size(0), linear=True, range_id=d.solution_space.id,
                                 parameter_type=p.initial_data.parameter_type)
    else:
        I = p.initial_data.evaluate(grid.quadrature_points(0, order=2)).squeeze()
        I = np.sum(I * grid.reference_element.quadrature(order=2)[1], axis=1) * (1. / grid.reference_element.volume)
        I = d.solution_space.make_array(I)

    if time_stepper is None:
        if p.stationary_part.diffusion is None:
            time_stepper = ExplicitEulerTimeStepper(nt=nt)
        else:
            time_stepper = ImplicitEulerTimeStepper(nt=nt)

    rhs = None if isinstance(d.rhs, ZeroOperator) else d.rhs

    d = InstationaryDiscretization(operator=d.operator, rhs=rhs, mass=None, initial_data=I, T=p.T,
                                   products=d.products, time_stepper=time_stepper,
                                   parameter_space=p.parameter_space, visualizer=d.visualizer,
                                   num_values=num_values, name='{}_FV'.format(p.name))

    if preassemble:
        data['unassembled_d'] = d
        d = preassemble_(d)

    return d, data
