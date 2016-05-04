# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.domaindescriptions.basic import LineDomain, RectDomain, TorusDomain, CircleDomain
from pymor.functions.basic import ConstantFunction
from pymor.functions.interfaces import FunctionInterface
from pymor.parameters.spaces import CubicParameterSpace


class BurgersProblem(InstationaryAdvectionProblem):
    """One-dimensional Burgers-type problem.

    The problem is to solve ::

        ∂_t u(x, t, μ)  +  ∂_x (v * u(x, t, μ)^μ) = 0
                                       u(x, 0, μ) = u_0(x)

    for u with t in [0, 0.3], x in [0, 2].

    Parameters
    ----------
    v
        The velocity v.
    circle
        If `True` impose periodic boundary conditions. Otherwise Dirichlet left,
        outflow right.
    initial_data_type
        Type of initial data (`'sin'` or `'bump'`).
    parameter_range
        The interval in which μ is allowed to vary.
    """

    def __init__(self, v=1., circle=True, initial_data_type='sin', parameter_range=(1., 2.)):

        assert initial_data_type in ('sin', 'bump')

        flux_function = BurgersFlux(v)
        flux_function_derivative = BurgersFluxDerivative(v)

        if initial_data_type == 'sin':
            initial_data = BurgersSinInitialData()
            dirichlet_data = ConstantFunction(dim_domain=1, value=0.5)
        else:
            initial_data = BurgersBumpInitialData()
            dirichlet_data = ConstantFunction(dim_domain=1, value=0)

        if circle:
            domain = CircleDomain([0, 2])
        else:
            domain = LineDomain([0, 2], right=None)

        super(BurgersProblem, self).__init__(domain=domain,
                                             rhs=None,
                                             flux_function=flux_function,
                                             flux_function_derivative=flux_function_derivative,
                                             initial_data=initial_data,
                                             dirichlet_data=dirichlet_data,
                                             T=0.3, name='BurgersProblem')

        self.parameter_space = CubicParameterSpace({'exponent': 0}, *parameter_range)
        self.parameter_range = parameter_range
        self.initial_data_type = initial_data_type
        self.circle = circle
        self.v = v


class Burgers2DProblem(InstationaryAdvectionProblem):
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
        If `True` impose periodic boundary conditions. Otherwise Dirichlet left and bottom,
        outflow top and right.
    initial_data_type
        Type of initial data (`'sin'` or `'bump'`).
    parameter_range
        The interval in which μ is allowed to vary.
    """
    def __init__(self, vx=1., vy=1., torus=True, initial_data_type='sin', parameter_range=(1., 2.)):

        assert initial_data_type in ('sin', 'bump')

        flux_function = Burgers2DFlux(vx, vy)
        flux_function_derivative = Burgers2DFluxDerivative(vx, vy)

        if initial_data_type == 'sin':
            initial_data = Burgers2DSinInitialData()
            dirichlet_data = ConstantFunction(dim_domain=2, value=0.5)
        else:
            initial_data = Burgers2DBumpInitialData()
            dirichlet_data = ConstantFunction(dim_domain=2, value=0)

        domain = TorusDomain([[0, 0], [2, 1]]) if torus else RectDomain([[0, 0], [2, 1]], right=None, top=None)

        super(Burgers2DProblem, self).__init__(domain=domain,
                                               rhs=None,
                                               flux_function=flux_function,
                                               flux_function_derivative=flux_function_derivative,
                                               initial_data=initial_data,
                                               dirichlet_data=dirichlet_data,
                                               T=0.3, name='Burgers2DProblem')

        self.parameter_space = CubicParameterSpace({'exponent': 0}, *parameter_range)
        self.parameter_range = parameter_range
        self.initial_data_type = initial_data_type
        self.torus = torus
        self.vx = vx
        self.vy = vy


class BurgersFlux(FunctionInterface):

    dim_domain = 1
    shape_range = (1,)

    def __init__(self, v):
        self.v = v
        self.build_parameter_type({'exponent': tuple()}, local_global=True)

    def evaluate(self, U, mu=None):
        mu = self.parse_parameter(mu)
        U_exp = np.sign(U) * np.power(np.abs(U), mu['exponent'])
        R = U_exp * self.v
        return R


class BurgersFluxDerivative(FunctionInterface):

    dim_domain = 1
    shape_range = (1,)

    def __init__(self, v):
        self.v = v
        self.build_parameter_type({'exponent': tuple()}, local_global=True)

    def evaluate(self, U, mu=None):
        mu = self.parse_parameter(mu)
        U_exp = mu['exponent'] * (np.sign(U) * np.power(np.abs(U), mu['exponent']-1))
        R = U_exp * self.v
        return R


class BurgersSinInitialData(FunctionInterface):

    dim_domain = 1
    shape_range = tuple()

    def evaluate(self, x, mu=None):
        return 0.5 * (np.sin(2 * np.pi * x[..., 0]) + 1.)


class BurgersBumpInitialData(FunctionInterface):

    dim_domain = 1
    shape_range = tuple()

    def evaluate(self, x, mu=None):
        return (x[..., 0] >= 0.5) * (x[..., 0] <= 1) * 1


class Burgers2DFlux(FunctionInterface):

    dim_domain = 1
    shape_range = (2,)

    def __init__(self, vx, vy):
        self.vx = vx
        self.vy = vy
        self.build_parameter_type({'exponent': tuple()}, local_global=True)

    def evaluate(self, U, mu=None):
        mu = self.parse_parameter(mu)
        U = U.reshape(U.shape[:-1])
        U_exp = np.sign(U) * np.power(np.abs(U), mu['exponent'])
        R = np.empty(U.shape + (2,))
        R[..., 0] = U_exp * self.vx
        R[..., 1] = U_exp * self.vy
        return R


class Burgers2DFluxDerivative(FunctionInterface):

    dim_domain = 1
    shape_range = (2,)

    def __init__(self, vx, vy):
        self.vx = vx
        self.vy = vy
        self.build_parameter_type({'exponent': tuple()}, local_global=True)

    def evaluate(self, U, mu=None):
        mu = self.parse_parameter(mu)
        U = U.reshape(U.shape[:-1])
        U_exp = mu['exponent'] * (np.sign(U) * np.power(np.abs(U), mu['exponent']-1))
        R = np.empty(U.shape + (2,))
        R[..., 0] = U_exp * self.vx
        R[..., 1] = U_exp * self.vy
        return R


class Burgers2DSinInitialData(FunctionInterface):

    dim_domain = 2
    shape_range = tuple()

    def evaluate(self, x, mu=None):
        return 0.5 * (np.sin(2 * np.pi * x[..., 0]) * np.sin(2 * np.pi * x[..., 1]) + 1.)


class Burgers2DBumpInitialData(FunctionInterface):

    dim_domain = 2
    shape_range = tuple()

    def evaluate(self, x, mu=None):
        return (x[..., 0] >= 0.5) * (x[..., 0] <= 1) * 1
