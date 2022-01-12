# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, LincombFunction
from pymor.parameters.functionals import ExpressionParameterFunctional


def helmholtz_problem(domain=RectDomain(), rhs=None, parameter_range=(0., 100.),
                      dirichlet_data=None, neumann_data=None):
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
    """
    return StationaryProblem(

        domain=domain,

        rhs=rhs or ConstantFunction(1., dim_domain=domain.dim),

        dirichlet_data=dirichlet_data,

        neumann_data=neumann_data,

        diffusion=ConstantFunction(1., dim_domain=domain.dim),

        reaction=LincombFunction([ConstantFunction(1., dim_domain=domain.dim)],
                                 [ExpressionParameterFunctional('-k[0]**2', {'k': 1})]),

        parameter_ranges={'k': parameter_range},

        name='helmholtz_problem'

    )
