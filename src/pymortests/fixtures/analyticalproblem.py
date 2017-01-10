# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import pytest

from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.analyticalproblems.burgers import burgers_problem, burgers_problem_2d
from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.analyticalproblems.helmholtz import helmholtz_problem
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.functions.basic import ConstantFunction, GenericFunction, LincombFunction
from pymor.parameters.functionals import ExpressionParameterFunctional

picklable_thermalblock_problems = \
    [thermal_block_problem(),
     thermal_block_problem(num_blocks=(3, 2)),
     thermal_block_problem(num_blocks=(1, 1)),
     thermal_block_problem(num_blocks=(2, 2), parameter_range=(1., 100.))]


non_picklable_thermalblock_problems = \
    [thermal_block_problem(num_blocks=(1, 3), parameter_range=(0.4, 0.5)).with_(
        rhs=GenericFunction(dim_domain=2, mapping=lambda X: X[..., 0] + X[..., 1]))]


thermalblock_problems = picklable_thermalblock_problems + non_picklable_thermalblock_problems


burgers_problems = \
    [burgers_problem(),
     burgers_problem(v=0.2, circle=False),
     burgers_problem(v=0.4, initial_data_type='bump'),
     burgers_problem(parameter_range=(1., 1.3)),
     burgers_problem_2d(),
     burgers_problem_2d(torus=False, initial_data_type='bump', parameter_range=(1.3, 1.5))]


picklable_elliptic_problems = \
    [EllipticProblem(),
     helmholtz_problem()]


non_picklable_elliptic_problems = \
    [EllipticProblem(rhs=ConstantFunction(dim_domain=2, value=21.),
                     diffusion=LincombFunction(
                         [GenericFunction(dim_domain=2, mapping=lambda X, p=p: X[..., 0]**p)
                          for p in range(5)],
                         [ExpressionParameterFunctional('max(mu["exp"], {})'.format(m), parameter_type={'exp': ()})
                          for m in range(5)]
                     ))]


elliptic_problems = picklable_thermalblock_problems + non_picklable_elliptic_problems


picklable_advection_problems = \
    [InstationaryAdvectionProblem()]


non_picklable_advection_problems = \
    [InstationaryAdvectionProblem(rhs=ConstantFunction(dim_domain=2, value=42.),
                                  flux_function=GenericFunction(dim_domain=1, shape_range=(2,),
                                                                mapping=lambda X: X**2 + X),
                                  flux_function_derivative=GenericFunction(dim_domain=1, shape_range=(2,),
                                                                           mapping=lambda X: X * 2))]


advection_problems = picklable_advection_problems + non_picklable_advection_problems


@pytest.fixture(params=elliptic_problems + advection_problems + thermalblock_problems + burgers_problems)
def analytical_problem(request):
    return request.param


@pytest.fixture(params=picklable_elliptic_problems + picklable_advection_problems
                       + picklable_thermalblock_problems + burgers_problems)
def picklable_analytical_problem(request):
    return request.param
