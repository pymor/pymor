# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from itertools import product

from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction, ExpressionFunction, LincombFunction
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace


def thermal_block_problem(num_blocks=(3, 3), parameter_range=(0.1, 1)):
    """Analytical description of a 2D 'thermal block' diffusion problem.

    The problem is to solve the elliptic equation ::

      - ∇ ⋅ [ d(x, μ) ∇ u(x, μ) ] = f(x, μ)

    on the domain [0,1]^2 with Dirichlet zero boundary values. The domain is
    partitioned into nx x ny blocks and the diffusion function d(x, μ) is
    constant on each such block (i,j) with value μ_ij. ::

           ----------------------------
           |        |        |        |
           |  μ_11  |  μ_12  |  μ_13  |
           |        |        |        |
           |---------------------------
           |        |        |        |
           |  μ_21  |  μ_22  |  μ_23  |
           |        |        |        |
           ----------------------------

    Parameters
    ----------
    num_blocks
        The tuple `(nx, ny)`
    parameter_range
        A tuple `(μ_min, μ_max)`. Each |Parameter| component μ_ij is allowed
        to lie in the interval [μ_min, μ_max].
    """

    def parameter_functional_factory(ix, iy):
        return ProjectionParameterFunctional(component_name='diffusion',
                                             component_shape=(num_blocks[1], num_blocks[0]),
                                             coordinates=(num_blocks[1] - iy - 1, ix),
                                             name='diffusion_{}_{}'.format(ix, iy))

    def diffusion_function_factory(ix, iy):
        if ix + 1 < num_blocks[0]:
            X = '(x[..., 0] >= ix * dx) * (x[..., 0] < (ix + 1) * dx)'
        else:
            X = '(x[..., 0] >= ix * dx)'
        if iy + 1 < num_blocks[1]:
            Y = '(x[..., 1] >= iy * dy) * (x[..., 1] < (iy + 1) * dy)'
        else:
            Y = '(x[..., 1] >= iy * dy)'
        return ExpressionFunction('{} * {} * 1.'.format(X, Y),
                                  2, (), {}, {'ix': ix, 'iy': iy, 'dx': 1. / num_blocks[0], 'dy': 1. / num_blocks[1]},
                                  name='diffusion_{}_{}'.format(ix, iy))

    return StationaryProblem(

        domain=RectDomain(),

        rhs=ConstantFunction(dim_domain=2, value=1.),

        diffusion=LincombFunction([diffusion_function_factory(ix, iy)
                                   for ix, iy in product(range(num_blocks[0]), range(num_blocks[1]))],
                                  [parameter_functional_factory(ix, iy)
                                   for ix, iy in product(range(num_blocks[0]), range(num_blocks[1]))],
                                  name='diffusion'),

        parameter_space=CubicParameterSpace({'diffusion': (num_blocks[1], num_blocks[0])}, *parameter_range),

        name='ThermalBlock({})'.format(num_blocks)

    )
