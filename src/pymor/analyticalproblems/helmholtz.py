# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace


class HelmholtzProblem(EllipticProblem):
    """Helmholtz equation problem.

    This problem is to solve the Helmholtz equation ::

      - ∆ u(x, k) - k^2 u(x, k) = f(x, k)

    on a given domain.

    Parameters
    ----------
    domain
        A |DomainDescription| of the domain the problem is posed on.
    rhs
        The |Function| f(x, μ).
    parameter_range
        A tuple `(k_min, k_max)` describing the interval in which k is allowd to vary.
    dirichlet_data
        |Function| providing the Dirichlet boundary values.
    neumann_data
        |Function| providing the Neumann boundary values.
    name
        Name of the problem.
    """

    def __init__(self, domain=RectDomain(), rhs=None, parameter_range=(0., 100.),
                 dirichlet_data=None, neumann_data=None):

        self.parameter_range = parameter_range  # needed for with_
        parameter_space = CubicParameterSpace({'k': ()}, *parameter_range)
        super().__init__(
            diffusion_functions=[ConstantFunction(1., dim_domain=domain.dim)],
            diffusion_functionals=[1.],
            reaction_functions=[ConstantFunction(1., dim_domain=domain.dim)],
            reaction_functionals=[ExpressionParameterFunctional('-k**2', {'k': ()})],
            domain=domain,
            rhs=rhs or ConstantFunction(1., dim_domain=domain.dim),
            parameter_space=parameter_space,
            dirichlet_data=dirichlet_data,
            neumann_data=neumann_data)
