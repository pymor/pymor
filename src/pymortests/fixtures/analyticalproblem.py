# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import pytest

from pymor.analyticalproblems.burgers import burgers_problem, burgers_problem_2d
from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import GenericFunction, ConstantFunction, LincombFunction
from pymor.analyticalproblems.helmholtz import helmholtz_problem
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.parameters.functionals import ExpressionParameterFunctional


picklable_thermalblock_problems = [
    thermal_block_problem(),
    thermal_block_problem(num_blocks=(3, 2)),
    thermal_block_problem(num_blocks=(1, 1)),
    thermal_block_problem(num_blocks=(2, 2), parameter_range=(1., 100.)),
]


non_picklable_thermalblock_problems = [
    thermal_block_problem(
        num_blocks=(1, 3),
        parameter_range=(0.4, 0.5)).with_(
            rhs=GenericFunction(dim_domain=2, mapping=lambda X: X[..., 0] + X[..., 1])),
]


thermalblock_problems = picklable_thermalblock_problems + non_picklable_thermalblock_problems


burgers_problems = [
    burgers_problem(),
    burgers_problem(v=0.2, circle=False),
    burgers_problem(v=0.4, initial_data_type='bump'),
    burgers_problem(parameter_range=(1., 1.3)),
    burgers_problem_2d(),
    burgers_problem_2d(torus=False, initial_data_type='bump', parameter_range=(1.3, 1.5)),
]


picklable_elliptic_problems = [
    StationaryProblem(
        domain=RectDomain(),
        rhs=ConstantFunction(dim_domain=2, value=1.),
    ),
    helmholtz_problem(),
]


non_picklable_elliptic_problems = [
    StationaryProblem(
        domain=RectDomain(),
        rhs=ConstantFunction(dim_domain=2, value=21.),
        diffusion=LincombFunction(
            [GenericFunction(dim_domain=2, mapping=lambda X, p=p: X[..., 0]**p)
             for p in range(5)],
            [ExpressionParameterFunctional(f'max(exp * [1,0] + [0, {m}])', parameters={'exp': 1})
             for m in range(5)],
        ),
    ),
]


elliptic_problems = picklable_thermalblock_problems + non_picklable_elliptic_problems


@pytest.fixture(params=elliptic_problems + thermalblock_problems + burgers_problems)
def analytical_problem(request):
    return request.param


@pytest.fixture(params=picklable_elliptic_problems + picklable_thermalblock_problems + burgers_problems)
def picklable_analytical_problem(request):
    return request.param
