# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import product

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction, LincombFunction
from pymor.parameters.functionals import ProjectionParameterFunctional


def thermal_block_problem(num_blocks=(3, 3), parameter_range=(0.1, 1)):
    """Analytical description of a 2D 'thermal block' diffusion problem.

    The problem is to solve the elliptic equation ::

      - ∇ ⋅ [ d(x, μ) ∇ u(x, μ) ] = 1

    on the domain [0,1]^2 with Dirichlet zero boundary values. The domain is
    partitioned into nx x ny blocks and the diffusion function d(x, μ) is
    constant on each such block i with value μ_i. ::

           ----------------------------
           |        |        |        |
           |  μ_4   |  μ_5   |  μ_6   |
           |        |        |        |
           |---------------------------
           |        |        |        |
           |  μ_1   |  μ_2   |  μ_3   |
           |        |        |        |
           ----------------------------

    Parameters
    ----------
    num_blocks
        The tuple `(nx, ny)`
    parameter_range
        A tuple `(μ_min, μ_max)`. Each |Parameter| component μ_i is allowed
        to lie in the interval [μ_min, μ_max].
    """

    def parameter_functional_factory(ix, iy):
        return ProjectionParameterFunctional('diffusion',
                                             size=num_blocks[0]*num_blocks[1],
                                             index=ix + iy*num_blocks[0],
                                             name=f'diffusion_{ix}_{iy}')

    def diffusion_function_factory(ix, iy):
        if ix + 1 < num_blocks[0]:
            X = '(x[0] >= ix * dx) * (x[0] < (ix + 1) * dx)'
        else:
            X = '(x[0] >= ix * dx)'
        if iy + 1 < num_blocks[1]:
            Y = '(x[1] >= iy * dy) * (x[1] < (iy + 1) * dy)'
        else:
            Y = '(x[1] >= iy * dy)'
        return ExpressionFunction(f'{X} * {Y} * 1.',
                                  2, {}, {'ix': ix, 'iy': iy, 'dx': 1. / num_blocks[0], 'dy': 1. / num_blocks[1]},
                                  name=f'diffusion_{ix}_{iy}')

    return StationaryProblem(

        domain=RectDomain(),

        rhs=ConstantFunction(dim_domain=2, value=1.),

        diffusion=LincombFunction([diffusion_function_factory(ix, iy)
                                   for iy, ix in product(range(num_blocks[1]), range(num_blocks[0]))],
                                  [parameter_functional_factory(ix, iy)
                                   for iy, ix in product(range(num_blocks[1]), range(num_blocks[0]))],
                                  name='diffusion'),

        parameter_ranges=parameter_range,

        name=f'ThermalBlock({num_blocks})'

    )
