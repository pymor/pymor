# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.interfaces import ImmutableInterface
from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction


class ParabolicProblem(ImmutableInterface):
    """Affinely decomposed linear parabolic problem.

    The problem consists in solving ::

    |                       K
    |  ∂_t u(x, t, μ) - ∇ ⋅ ∑  θ_k(μ) ⋅ d_k(x) ∇ u(x, t, μ) = f(x, t, μ)
    |                      k=0
    |                                            u(x, 0, μ) = u_0(x, μ)

    for u with t in [0, T], x in Ω.

    Parameters
    ----------
    domain
        A |DomainDescription| of the domain the problem is posed on.
    rhs
        The |Function| f(x, μ). `rhs.dim_domain` has to agree with the
        dimension of `domain`, whereas `rhs.shape_range` has to be `tuple()`.
    diffusion_functions
        List containing the |Functions| d_k(x), each having `shape_range`
        of either `tuple()` or `(dim domain, dim domain)`.
    diffusion_functionals
        List containing the |ParameterFunctionals| θ_k(μ). If
        `len(diffusion_functions) == 1`, `diffusion_functionals` is allowed
        to be `None`, in which case no parameter dependence is assumed.
    dirichlet_data
        |Function| providing the Dirichlet boundary values in global coordinates.
    neumann_data
        |Function| providing the Neumann boundary values in global coordinates.
    parameter_space
        |ParameterSpace| for the problem.
    initial_data
        |Function| providing the initial values in global coordinates.
    T
        The end time T.
    name
        Name of the problem.

    Attributes
    ----------
    domain
    rhs
    diffusion_functions
    diffusion_functionals
    dirichlet_data
    neumann_data
    initial_data
    T
    """

    def __init__(self, domain=RectDomain(), rhs=ConstantFunction(dim_domain=2),
                 diffusion_functions=(ConstantFunction(dim_domain=2),), diffusion_functionals=None, dirichlet_data=None,
                 neumann_data=None, initial_data=ConstantFunction(dim_domain=2), T=1, parameter_space=None, name=None):

        assert isinstance(diffusion_functions, (tuple, list))
        assert diffusion_functionals is None and len(diffusion_functions) == 1 \
            or len(diffusion_functionals) == len(diffusion_functions)
        assert rhs.dim_domain == diffusion_functions[0].dim_domain
        assert dirichlet_data is None or dirichlet_data.dim_domain == diffusion_functions[0].dim_domain
        assert neumann_data is None or neumann_data.dim_domain == diffusion_functions[0].dim_domain
        assert initial_data is None or initial_data.dim_domain == diffusion_functions[0].dim_domain
        self.domain = domain
        self.rhs = rhs
        self.diffusion_functions = diffusion_functions
        self.diffusion_functionals = diffusion_functionals
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.initial_data = initial_data
        self.T = T
        self.parameter_space = parameter_space
        self.name = name

    @classmethod
    def from_elliptic(cls, elliptic_problem, initial_data=ConstantFunction(dim_domain=2), T=1):
        assert isinstance(elliptic_problem, EllipticProblem)

        return cls(elliptic_problem.domain, elliptic_problem.rhs, elliptic_problem.diffusion_functions,
                   elliptic_problem.diffusion_functionals, elliptic_problem.dirichlet_data,
                   elliptic_problem.neumann_data, initial_data, T, elliptic_problem.parameter_space,
                   'ParabolicProblem_from_{}'.format(elliptic_problem.name))

    def elliptic_part(self):
        return EllipticProblem(
            domain=self.domain,
            rhs=self.rhs,
            diffusion_functions=self.diffusion_functions,
            diffusion_functionals=self.diffusion_functionals,
            dirichlet_data=self.dirichlet_data,
            neumann_data=self.neumann_data,
            parameter_space=self.parameter_space,
            name='{}_elliptic_part'.format(self.name))
