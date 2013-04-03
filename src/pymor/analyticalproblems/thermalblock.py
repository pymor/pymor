# -*- coding: utf-8 -*-
# pymor (http://www.pymor.org)
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import product

from pymor.domaindescriptions import RectDomain
from pymor.functions import GenericFunction, ConstantFunction
from pymor.parameters import CubicParameterSpace, ProjectionParameterFunctional
from pymor.analyticalproblems.elliptic import EllipticProblem


class ThermalBlockProblem(EllipticProblem):
    '''Analytical description of a 2D thermal block diffusion problem.

    This problem is to solve the elliptic equation ::

      - ∇ ⋅ [ d(x, μ) ∇ u(x, μ) ] = f(x, μ)

    on the domain [0,1]^2 with Dirichlet zero boundary values. The domain is
    partitioned into nx x ny blocks and the diffusion function d(x, μ) is
    constant on each such block (i,j) with value d(x, μ_ij). ::

           ----------------------------
           |        |        |        |
           |  μ_11  |  μ_12  |  μ_13  |
           |        |        |        |
           |---------------------------
           |        |        |        |
           |  μ_21  |  μ_22  |  μ_23  |
           |        |        |        |
           ----------------------------

    The Problem is implemented as a special EllipticProblem with the
    characteristic functions of the blocks as `diffusion_functions`.

    Parameters
    ----------
    num_blocks
        The tuple (nx, ny)
    parameter_range
        A tuple (mu_min, mu_max). Each parameter component mu_ij is allowed
        to lie in the interval [mu_min, mu_max].
    rhs
        The function f(x, mu).

    Attributes
    ----------
    parameter_space
        The `ParameterSpace` defined by `parameter_range`.

    Inherits
    --------
    EllipticProblem
    '''

    def __init__(self, num_blocks=(3, 3), parameter_range=(0.1, 1), rhs=ConstantFunction(dim_domain=2)):

        domain = RectDomain()
        parameter_space = CubicParameterSpace({'diffusion': (num_blocks[1], num_blocks[0])}, *parameter_range)
        dx = 1 / num_blocks[0]
        dy = 1 / num_blocks[1]

        def diffusion_function_factory(x, y):
            return GenericFunction(lambda X: (1 * (X[..., 0] >= x * dx) * (X[..., 0] < (x + 1) * dx)
                                                * (X[..., 1] >= y * dy) * (X[..., 1] < (y + 1) * dy)),
                                   dim_domain=2, name='diffusion_function_{}_{}'.format(x, y))

        def parameter_functional_factory(x, y):
            return ProjectionParameterFunctional(parameter_space, 'diffusion',
                                                 (num_blocks[1] - y - 1, x),
                                                 name='diffusion_{}_{}'.format(x, y))

        diffusion_functions = tuple(diffusion_function_factory(x, y)
                                    for x, y in product(xrange(num_blocks[0]), xrange(num_blocks[1])))
        parameter_functionals = tuple(parameter_functional_factory(x, y)
                                      for x, y in product(xrange(num_blocks[0]), xrange(num_blocks[1])))

        super(ThermalBlockProblem, self).__init__(domain, rhs, diffusion_functions, parameter_functionals,
                                                  name='ThermalBlock')
        self.parameter_space = parameter_space
