# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.domaindescriptions import LineDomain, RectDomain, TorusDomain, CircleDomain
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction


def burgers_problem(v=1., circle=True, initial_data_type='sin', parameter_range=(1., 2.)):
    """One-dimensional Burgers-type problem.

    The problem is to solve ::

        ∂_t u(x, t, μ)  +  ∂_x (v * u(x, t, μ)^μ) = 0
                                       u(x, 0, μ) = u_0(x)

    for u with t in [0, 0.3] and x in [0, 2].

    Parameters
    ----------
    v
        The velocity v.
    circle
        If `True`, impose periodic boundary conditions. Otherwise Dirichlet left,
        outflow right.
    initial_data_type
        Type of initial data (`'sin'` or `'bump'`).
    parameter_range
        The interval in which μ is allowed to vary.
    """
    assert initial_data_type in ('sin', 'bump')

    if initial_data_type == 'sin':
        initial_data = ExpressionFunction('0.5 * (sin(2 * pi * x[0]) + 1.)', 1)
        dirichlet_data = ConstantFunction(dim_domain=1, value=0.5)
    else:
        initial_data = ExpressionFunction('(x[0] >= 0.5) * (x[0] <= 1) * 1.', 1)
        dirichlet_data = ConstantFunction(dim_domain=1, value=0.)

    return InstationaryProblem(

        StationaryProblem(
            domain=CircleDomain([0, 2]) if circle else LineDomain([0, 2], right=None),

            dirichlet_data=dirichlet_data,

            rhs=None,

            nonlinear_advection=ExpressionFunction('abs(x)**exponent[0] * v',
                                                   1, {'exponent': 1}, {'v': v}),

            nonlinear_advection_derivative=ExpressionFunction('exponent * abs(x)**(exponent[0]-1) * sign(x) * v',
                                                              1, {'exponent': 1}, {'v': v}),
        ),

        T=0.3,

        initial_data=initial_data,

        parameter_ranges={'exponent': parameter_range},

        name=f"burgers_problem({v}, {circle}, '{initial_data_type}')"
    )


def burgers_problem_2d(vx=1., vy=1., torus=True, initial_data_type='sin', parameter_range=(1., 2.)):
    """Two-dimensional Burgers-type problem.

    The problem is to solve ::

        ∂_t u(x, t, μ)  +  ∇ ⋅ (v * u(x, t, μ)^μ) = 0
                                       u(x, 0, μ) = u_0(x)

    for u with t in [0, 0.3], x in [0, 2] x [0, 1].

    Parameters
    ----------
    vx
        The x component of the velocity vector v.
    vy
        The y component of the velocity vector v.
    torus
        If `True`, impose periodic boundary conditions. Otherwise,
        Dirichlet left and bottom, outflow top and right.
    initial_data_type
        Type of initial data (`'sin'` or `'bump'`).
    parameter_range
        The interval in which μ is allowed to vary.
    """
    assert initial_data_type in ('sin', 'bump')

    if initial_data_type == 'sin':
        initial_data = ExpressionFunction("0.5 * (sin(2 * pi * x[0]) * sin(2 * pi * x[1]) + 1.)", 2)
        dirichlet_data = ConstantFunction(dim_domain=2, value=0.5)
    else:
        initial_data = ExpressionFunction("(x[0] >= 0.5) * (x[0] <= 1) * 1", 2)
        dirichlet_data = ConstantFunction(dim_domain=2, value=0.)

    return InstationaryProblem(

        StationaryProblem(
            domain=TorusDomain([[0, 0], [2, 1]]) if torus else RectDomain([[0, 0], [2, 1]], right=None, top=None),

            dirichlet_data=dirichlet_data,

            rhs=None,

            nonlinear_advection=ExpressionFunction("abs(x[0])**exponent * v",
                                                   1, {'exponent': 1}, {'v': [vx, vy]}),

            nonlinear_advection_derivative=ExpressionFunction("exponent * abs(x[0])**(exponent-1) * sign(x[0]) * v",
                                                              1, {'exponent': 1}, {'v': [vx, vy]}),
        ),

        initial_data=initial_data,

        T=0.3,

        parameter_ranges=parameter_range,

        name=f"burgers_problem_2d({vx}, {vy}, {torus}, '{initial_data_type}')"
    )
