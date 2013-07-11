# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

import pymor.core as core
from pymor.tools import Named
from pymor.domaindescriptions import RectDomain, TorusDomain, BoundaryType
from pymor.functions import ConstantFunction, GenericFunction
from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.parameters.spaces import CubicParameterSpace


class BurgersProblem(InstationaryAdvectionProblem):

    def __init__(self, vx = 1., vy = 1., torus=True, initial_data='sin', parameter_range=(1., 2.)):

        assert initial_data in ('sin', 'bump')

        def burgers_flux(U, mu):
            U_exp = np.sign(U) * np.power(np.abs(U), mu['exponent'])
            R = np.empty(U.shape + (2,))
            R[...,0] = U_exp * vx
            R[...,1] = U_exp * vy
            return R

        def burgers_flux_derivative(U, mu):
            U_exp = mu['exponent'] * (np.sign(U) * np.power(np.abs(U), mu['exponent']-1))
            R = np.empty(U.shape + (2,))
            R[...,0] = U_exp * vx
            R[...,1] = U_exp * vy
            return R

        flux_function = GenericFunction(burgers_flux, dim_domain=0, dim_range=2,
                                        parameter_type={'exponent': 0},
                                        name_map={'.exponent': 'exponent'},
                                        name='burgers_flux')

        flux_function_derivative = GenericFunction(burgers_flux_derivative, dim_domain=0, dim_range=2,
                                                   parameter_type={'exponent': 0},
                                                   name_map={'.exponent': 'exponent'},
                                                   name='burgers_flux')

        if initial_data == 'sin':
            def initial_data(x):
                return 0.5 * (np.sin(2 * np.pi * x[..., 0]) * np.sin(2 * np.pi * x[..., 1]) + 1.)
            dirichlet_data=ConstantFunction(dim_domain=2, dim_range=1, value=0.5)
        else:
            def initial_data(x):
                return (x[..., 0] >= 0.5) * (x[..., 0] <= 1) * 1
            dirichlet_data=ConstantFunction(dim_domain=2, dim_range=1, value=0)

        initial_data = GenericFunction(initial_data, dim_domain=2, dim_range=1)

        domain = TorusDomain([[0, 0], [2, 1]]) if torus else RectDomain([[0, 0], [2, 1]], right=None,
                top=None)

        super(BurgersProblem, self).__init__(domain=domain,
                                             rhs=ConstantFunction(value=0, dim_domain=2),
                                             flux_function=flux_function,
                                             flux_function_derivative=flux_function_derivative,
                                             initial_data=initial_data,
                                             dirichlet_data=dirichlet_data,
                                             T=0.3, name='Burgers Problem')

        self.parameter_space = CubicParameterSpace({'exponent': 0}, *parameter_range)
