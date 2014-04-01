# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.core import Unpicklable, inject_sid
from pymor.domaindescriptions import LineDomain, RectDomain, TorusDomain, CircleDomain
from pymor.functions import ConstantFunction, GenericFunction
from pymor.parameters.spaces import CubicParameterSpace


class BurgersProblem(InstationaryAdvectionProblem, Unpicklable):
    '''One-dimensional Burgers-type problem.

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
    '''

    def __init__(self, v=1., circle=True, initial_data_type='sin', parameter_range=(1., 2.)):

        assert initial_data_type in ('sin', 'bump')

        def burgers_flux(U, mu):
            U_exp = np.sign(U) * np.power(np.abs(U), mu['exponent'])
            R = U_exp * v
            return R
        inject_sid(burgers_flux, str(BurgersProblem) + '.burgers_flux', v)

        def burgers_flux_derivative(U, mu):
            U_exp = mu['exponent'] * (np.sign(U) * np.power(np.abs(U), mu['exponent']-1))
            R = U_exp * v
            return R
        inject_sid(burgers_flux_derivative, str(BurgersProblem) + '.burgers_flux_derivative', v)

        flux_function = GenericFunction(burgers_flux, dim_domain=1, shape_range=(1,),
                                        parameter_type={'exponent': 0},
                                        name='burgers_flux')

        flux_function_derivative = GenericFunction(burgers_flux_derivative, dim_domain=1, shape_range=(1,),
                                                   parameter_type={'exponent': 0},
                                                   name='burgers_flux')

        if initial_data_type == 'sin':
            def initial_data(x):
                return 0.5 * (np.sin(2 * np.pi * x[..., 0]) + 1.)
            inject_sid(initial_data, str(BurgersProblem) + '.initial_data_sin')
            dirichlet_data = ConstantFunction(dim_domain=1, value=0.5)
        else:
            def initial_data(x):
                return (x[..., 0] >= 0.5) * (x[..., 0] <= 1) * 1
            inject_sid(initial_data, str(BurgersProblem) + '.initial_data_bump')
            dirichlet_data = ConstantFunction(dim_domain=1, value=0)

        initial_data = GenericFunction(initial_data, dim_domain=1)

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


class Burgers2DProblem(InstationaryAdvectionProblem, Unpicklable):
    '''Two-dimensional Burgers-type problem.

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
    '''
    def __init__(self, vx=1., vy=1., torus=True, initial_data_type='sin', parameter_range=(1., 2.)):

        assert initial_data_type in ('sin', 'bump')

        def burgers_flux(U, mu):
            U = U.reshape(U.shape[:-1])
            U_exp = np.sign(U) * np.power(np.abs(U), mu['exponent'])
            R = np.empty(U.shape + (2,))
            R[..., 0] = U_exp * vx
            R[..., 1] = U_exp * vy
            return R
        inject_sid(burgers_flux, str(Burgers2DProblem) + '.burgers_flux', vx, vy)

        def burgers_flux_derivative(U, mu):
            U = U.reshape(U.shape[:-1])
            U_exp = mu['exponent'] * (np.sign(U) * np.power(np.abs(U), mu['exponent']-1))
            R = np.empty(U.shape + (2,))
            R[..., 0] = U_exp * vx
            R[..., 1] = U_exp * vy
            return R
        inject_sid(burgers_flux_derivative, str(Burgers2DProblem) + '.burgers_flux_derivative', vx, vy)

        flux_function = GenericFunction(burgers_flux, dim_domain=1, shape_range=(2,),
                                        parameter_type={'exponent': 0},
                                        name='burgers_flux')

        flux_function_derivative = GenericFunction(burgers_flux_derivative, dim_domain=1, shape_range=(2,),
                                                   parameter_type={'exponent': 0},
                                                   name='burgers_flux')

        if initial_data_type == 'sin':
            def initial_data(x):
                return 0.5 * (np.sin(2 * np.pi * x[..., 0]) * np.sin(2 * np.pi * x[..., 1]) + 1.)
            inject_sid(initial_data, str(Burgers2DProblem) + '.initial_data_sin')
            dirichlet_data = ConstantFunction(dim_domain=2, value=0.5)
        else:
            def initial_data(x):
                return (x[..., 0] >= 0.5) * (x[..., 0] <= 1) * 1
            inject_sid(initial_data, str(Burgers2DProblem) + '.initial_data_bump')
            dirichlet_data = ConstantFunction(dim_domain=2, value=0)

        initial_data = GenericFunction(initial_data, dim_domain=2)

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
