# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

import pytest

from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.analyticalproblems.burgers import BurgersProblem, Burgers2DProblem
from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.analyticalproblems.thermalblock import ThermalBlockProblem
from pymor.functions.basic import GenericFunction, ConstantFunction
from pymor.parameters.functionals import ExpressionParameterFunctional


picklable_thermalblock_problems = \
    [ThermalBlockProblem(),
     ThermalBlockProblem(num_blocks=(3, 2)),
     ThermalBlockProblem(num_blocks=(1, 1)),
     ThermalBlockProblem(num_blocks=(2, 2), parameter_range=(1., 100.))]

thermalblock_problems = picklable_thermalblock_problems + \
    [ThermalBlockProblem(num_blocks=(1, 3), parameter_range=(0.4, 0.5),
                         rhs=GenericFunction(dim_domain=2, mapping=lambda X: X[..., 0] + X[..., 1]))]


burgers_problems = \
    [BurgersProblem(),
     BurgersProblem(v=2., circle=False),
     BurgersProblem(v=2., initial_data_type='bump'),
     BurgersProblem(parameter_range=(3., 4.)),
     Burgers2DProblem(),
     Burgers2DProblem(torus=False),
     Burgers2DProblem(torus=False, initial_data_type='bump', parameter_range=(7., 8.))]


picklable_elliptic_problems = \
    [EllipticProblem()]

elliptic_problems = picklable_thermalblock_problems + \
    [EllipticProblem(rhs=ConstantFunction(dim_domain=2, value=21.),
                     diffusion_functions=[GenericFunction(dim_domain=2,
                                                          mapping=lambda X,p=p: X[...,0]**p) for p in range(5)],
                     diffusion_functionals=[ExpressionParameterFunctional('max(mu["exp"], {})'.format(m),
                                                                          parameter_type={'exp': tuple()})
                                            for m in range(5)])]


picklable_advection_problems = \
    [InstationaryAdvectionProblem()]

advection_problems = picklable_advection_problems + \
    [InstationaryAdvectionProblem(rhs=ConstantFunction(dim_domain=2, value=42.),
                                  flux_function=GenericFunction(dim_domain=1, shape_range=(2,),
                                                                mapping=lambda X: X**2 + X),
                                  flux_function_derivative=GenericFunction(dim_domain=1, shape_range=(2,),
                                                                           mapping=lambda X: X * 2))]


@pytest.fixture(params=elliptic_problems + advection_problems + thermalblock_problems + burgers_problems)
def analytical_problem(request):
    return request.param


@pytest.fixture(params=picklable_elliptic_problems + picklable_advection_problems
                       + picklable_thermalblock_problems + burgers_problems)
def picklable_analytical_problem(request):
    return request.param
