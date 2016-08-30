# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.algorithms.timestepping import ExplicitEulerTimeStepper
from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.discretizations.basic import InstationaryDiscretization
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.gui.qt import PatchVisualizer, Matplotlib1DVisualizer
from pymor.operators.numpy import NumpyGenericOperator
from pymor.operators.fv import (nonlinear_advection_lax_friedrichs_operator,
                                nonlinear_advection_engquist_osher_operator,
                                nonlinear_advection_simplified_engquist_osher_operator,
                                L2Product, L2ProductFunctional)
from pymor.vectorarrays.numpy import NumpyVectorArray


def discretize_nonlinear_instationary_advection_fv(analytical_problem, diameter=None, nt=100, num_flux='lax_friedrichs',
                                                   lxf_lambda=1., eo_gausspoints=5, eo_intervals=1, num_values=None,
                                                   domain_discretizer=None, grid=None, boundary_info=None):
    """Discretizes an |InstationaryAdvectionProblem| using the finite volume method.

    Explicit Euler time-stepping is used for time discretization.

    Parameters
    ----------
    analytical_problem
        The |InstationaryAdvectionProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed as an argument to the
        `domain_discretizer`.
    nt
        The number of time steps.
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
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each
        intermediate vector that is calculated is returned.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter)`.
        If `None`, |discretize_domain_default| is used.
    grid
        Instead of using a domain discretizer, the |Grid| can also be passed directly
        using this parameter.
    boundary_info
        A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
        Must be provided if `grid` is specified.

    Returns
    -------
    discretization
        The |Discretization| that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.


    """

    assert isinstance(analytical_problem, InstationaryAdvectionProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None
    assert num_flux in ('lax_friedrichs', 'engquist_osher', 'simplified_engquist_osher')

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    p = analytical_problem

    if num_flux == 'lax_friedrichs':
        L = nonlinear_advection_lax_friedrichs_operator(grid, boundary_info, p.flux_function,
                                                        dirichlet_data=p.dirichlet_data, lxf_lambda=lxf_lambda)
    elif num_flux == 'engquist_osher':
        L = nonlinear_advection_engquist_osher_operator(grid, boundary_info, p.flux_function,
                                                        p.flux_function_derivative,
                                                        gausspoints=eo_gausspoints, intervals=eo_intervals,
                                                        dirichlet_data=p.dirichlet_data)
    else:
        L = nonlinear_advection_simplified_engquist_osher_operator(grid, boundary_info, p.flux_function,
                                                                   p.flux_function_derivative,
                                                                   dirichlet_data=p.dirichlet_data)
    F = None if p.rhs is None else L2ProductFunctional(grid, p.rhs)

    if p.initial_data.parametric:
        def initial_projection(U, mu):
            I = p.initial_data.evaluate(grid.quadrature_points(0, order=2), mu).squeeze()
            I = np.sum(I * grid.reference_element.quadrature(order=2)[1], axis=1) * (1. / grid.reference_element.volume)
            I = NumpyVectorArray(I, copy=False)
            return I.lincomb(U).data
        I = NumpyGenericOperator(initial_projection, dim_range=grid.size(0), linear=True,
                                 parameter_type=p.initial_data.parameter_type)
    else:
        I = p.initial_data.evaluate(grid.quadrature_points(0, order=2)).squeeze()
        I = np.sum(I * grid.reference_element.quadrature(order=2)[1], axis=1) * (1. / grid.reference_element.volume)
        I = NumpyVectorArray(I, copy=False)

    products = {'l2': L2Product(grid, boundary_info)}
    if grid.dim == 2:
        visualizer = PatchVisualizer(grid=grid, bounding_box=grid.bounding_box(), codim=0)
    elif grid.dim == 1:
        visualizer = Matplotlib1DVisualizer(grid, codim=0)
    else:
        visualizer = None
    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None
    time_stepper = ExplicitEulerTimeStepper(nt=nt)

    discretization = InstationaryDiscretization(operator=L, rhs=F, initial_data=I, T=p.T, products=products,
                                                time_stepper=time_stepper,
                                                parameter_space=parameter_space, visualizer=visualizer,
                                                num_values=num_values, name='{}_FV'.format(p.name))

    return discretization, {'grid': grid, 'boundary_info': boundary_info}
