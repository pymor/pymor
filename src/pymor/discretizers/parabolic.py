# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.discretizers.elliptic import discretize_elliptic_cg, discretize_elliptic_fv
from pymor.analyticalproblems.parabolic import ParabolicProblem
from pymor.discretizations.basic import InstationaryDiscretization
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.operators.cg import InterpolationOperator
from pymor.operators.numpy import NumpyGenericOperator
from pymor.vectorarrays.numpy import NumpyVectorArray


def discretize_parabolic_cg(analytical_problem, diameter=None, domain_discretizer=None,
                           grid=None, boundary_info=None, num_values=None, time_stepper=None, nt=None):
    """Discretizes an |ParabolicProblem| using finite elements.

    Parameters
    ----------
    analytical_problem
        The |ParabolicProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed to the `domain_discretizer`.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter, ...)`.
        If further arguments should be passed to the discretizer, use
        :func:`functools.partial`. If `None`, |discretize_domain_default| is used.
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
        The time-stepper to be used by :class:`~pymor.discretizations.basic.InstationaryDiscretization.solve`. Has to
        satisfy the :class:`~pymor.algorithms.timestepping.TimeStepperInterface`.
    nt
        The number of time-steps. If provided implicit euler time-stepping is used.

    Returns
    -------
    discretization
        The |Discretization| that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.
    """

    assert isinstance(analytical_problem, ParabolicProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None
    assert time_stepper is None or nt is None

    p = analytical_problem

    d, data = discretize_elliptic_cg(p.elliptic_part(), diameter=diameter, domain_discretizer=domain_discretizer,
        grid=grid, boundary_info=boundary_info)

    if p.initial_data.parametric:
        I = InterpolationOperator(data['grid'], p.initial_data)
    else:
        I = p.initial_data.evaluate(data['grid'].centers(data['grid'].dim))
        I = NumpyVectorArray(I, copy=False)

    if time_stepper is None:
        time_stepper = ImplicitEulerTimeStepper(nt=nt)

    mass = d.l2_0_product

    discretization = InstationaryDiscretization(operator=d.operator, rhs=d.rhs, mass=mass, initial_data=I, T=p.T,
                                                products=d.products, time_stepper=time_stepper,
                                                parameter_space=d.parameter_space, visualizer=d.visualizer,
                                                num_values=num_values, name='{}_CG'.format(p.name))

    return discretization, data


def discretize_parabolic_fv(analytical_problem, diameter=None, domain_discretizer=None,
                           grid=None, boundary_info=None, num_values=None, time_stepper=None, nt=None):
    """Discretizes an |ParabolicProblem| using the finite volume method.

    Parameters
    ----------
    analytical_problem
        The |ParabolicProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed to the `domain_discretizer`.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter, ...)`.
        If further arguments should be passed to the discretizer, use
        :func:`functools.partial`. If `None`, |discretize_domain_default| is used.
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
        The time-stepper to be used by :class:`~pymor.discretizations.basic.InstationaryDiscretization.solve`. Has to
        satisfy the :class:`~pymor.algorithms.timestepping.TimeStepperInterface`.
    nt
        The number of time-steps. If provided implicit euler time-stepping is used.

    Returns
    -------
    discretization
        The |Discretization| that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.
    """

    assert isinstance(analytical_problem, ParabolicProblem)
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None
    assert time_stepper is None or nt is None

    p = analytical_problem

    d, data = discretize_elliptic_fv(p.elliptic_part(), diameter=diameter, domain_discretizer=domain_discretizer,
                                         grid=grid, boundary_info=boundary_info)

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

    if time_stepper is None:
        time_stepper = ImplicitEulerTimeStepper(nt=nt)

    discretization = InstationaryDiscretization(operator=d.operator, rhs=d.rhs, mass=None, initial_data=I, T=p.T,
                                                products=d.products, time_stepper=time_stepper,
                                                parameter_space=d.parameter_space, visualizer=d.visualizer,
                                                num_values=num_values, name='{}_FV'.format(p.name))

    return discretization, data
