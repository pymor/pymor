# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.domaindescriptions.basic import LineDomain, RectDomain, TorusDomain, CircleDomain
from pymor.functions.basic import ConstantFunction, ExpressionFunction
from pymor.parameters.spaces import CubicParameterSpace


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
        initial_data = ExpressionFunction('0.5 * (sin(2 * pi * x[..., 0]) + 1.)', 1, ())
        dirichlet_data = ConstantFunction(dim_domain=1, value=0.5)
    else:
        initial_data = ExpressionFunction('(x[..., 0] >= 0.5) * (x[..., 0] <= 1) * 1.', 1, ())
        dirichlet_data = ConstantFunction(dim_domain=1, value=0.)

    return InstationaryAdvectionProblem(

        domain=CircleDomain([0, 2]) if circle else LineDomain([0, 2], right=None),

        T=0.3,

        initial_data=initial_data,

        dirichlet_data=dirichlet_data,

        rhs=None,

        flux_function=ExpressionFunction("sign(x) * abs(x)**mu['exponent'] * v",
                                         1, (1,), {'exponent': ()}, {'v': v}),

        flux_function_derivative=ExpressionFunction("mu['exponent'] * sign(x) * abs(x)**(mu['exponent']-1) * v",
                                                    1, (1,), {'exponent': ()}, {'v': v}),

        parameter_space=CubicParameterSpace({'exponent': 0}, *parameter_range),

        name="burgers_problem({}, {}, '{}')".format(v, circle, initial_data_type)

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
        initial_data = ExpressionFunction("0.5 * (sin(2 * pi * x[..., 0]) * sin(2 * pi * x[..., 1]) + 1.)", 2, ())
        dirichlet_data = ConstantFunction(dim_domain=2, value=0.5)
    else:
        initial_data = ExpressionFunction("(x[..., 0] >= 0.5) * (x[..., 0] <= 1) * 1", 2, ())
        dirichlet_data = ConstantFunction(dim_domain=2, value=0.)

    return InstationaryAdvectionProblem(

        domain=TorusDomain([[0, 0], [2, 1]]) if torus else RectDomain([[0, 0], [2, 1]], right=None, top=None),

        T=0.3,

        initial_data=initial_data,

        dirichlet_data=dirichlet_data,

        rhs=None,

        flux_function=ExpressionFunction("sign(x) * abs(x)**mu['exponent'] * v",
                                         1, (2,), {'exponent': ()}, {'v': np.array([vx, vy])}),

        flux_function_derivative=ExpressionFunction("mu['exponent'] * sign(x) * abs(x)**(mu['exponent']-1) * v",
                                                    1, (2,), {'exponent': ()}, {'v': np.array([vx, vy])}),

        parameter_space=CubicParameterSpace({'exponent': 0}, *parameter_range),

        name="burgers_problem_2d({}, {}, {}, '{}')".format(vx, vy, torus, initial_data_type)

    )
