from __future__ import absolute_import, division, print_function, unicode_literals

from itertools import product

import numpy as np

import pymor.core as core
from pymor.domaindescriptions import BoundaryType
from pymor.domaindescriptions import RectDomain
from pymor.analyticalproblems import PoissonProblem
from pymor.functions import GenericFunction, ConstantFunction
from pymor.parameters import CubicParameterSpace, ProjectionParameterFunctional


class ThermalBlockProblem(PoissonProblem):

    def __init__(self, num_blocks=(3,3), parameter_range=(0.1,1), rhs=ConstantFunction(dim_domain=2)):

        domain = RectDomain()
        parameter_space = CubicParameterSpace({'diffusion':(num_blocks[1], num_blocks[0])}, *parameter_range)
        dx = 1 / num_blocks[0]
        dy = 1 / num_blocks[1]

        def diffusion_function_factory(x, y):
            return GenericFunction(lambda X: (1 * (X[..., 0] >= x * dx) * (X[..., 0] < (x + 1) * dx)
                                                * (X[..., 1] >= y * dy) * (X[..., 1] < (y + 1) * dy)),
                                   dim_domain=2, name='diffusion_function_{}_{}'.format(x, y))

        def parameter_functional_factory(x, y):
            return ProjectionParameterFunctional(parameter_space, 'diffusion',
                                                 (num_blocks[1]-y-1, x),
                                                 name='diffusion_{}_{}'.format(x, y))

        diffusion_functions = tuple(diffusion_function_factory(x, y)
                                    for x, y in product(xrange(num_blocks[0]), xrange(num_blocks[1])))
        parameter_functionals = tuple(parameter_functional_factory(x, y)
                                    for x, y in product(xrange(num_blocks[0]), xrange(num_blocks[1])))


        super(ThermalBlockProblem, self).__init__(domain, rhs, diffusion_functions, parameter_functionals)
        self.parameter_space = parameter_space
