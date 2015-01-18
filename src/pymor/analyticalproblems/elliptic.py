# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Laier <m_laie01@uni-muenster.de>

from __future__ import absolute_import, division, print_function

from pymor.core.interfaces import ImmutableInterface
from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction


class EllipticProblem(ImmutableInterface):
    """Linear elliptic analytical problem.

    The problem consists in solving ::

    |        K
    |  - ∇ ⋅ ∑  θ_k(μ) ⋅ d_k(x) ∇ u(x, μ) = f(x, μ)
    |       k=0

    for u.

    Parameters
    ----------
    domain
        A |DomainDescription| of the domain the problem is posed on.
    rhs
        The |Function| f(x, μ).
    diffusion_functions
        List of the |functions| d_k(x).
    diffusion_functionals
        List of the |ParameterFunctionals| θ_k(μ). If None, and
        `len(diffusion_functions) > 1` let θ_k be the kth projection of the
        coefficient part of μ.  If None and `len(diffusion_functions) == 1`,
        no parameter dependence is assumed.
    dirichlet_data
        |Function| providing the Dirichlet boundary values in global coordinates.
    neumann_data
        |Function| providing the Neumann boundary values in global coordinates.
    parameter_space
        |ParameterSpace| for the problem.
    name
        Name of the problem.

    Attributes
    ----------
    domain
    rhs
    diffusion_functions
    diffusion_functionals
    dirichlet_data
    """

    def __init__(self, domain=RectDomain(), rhs=ConstantFunction(dim_domain=2),
                 diffusion_functions=(ConstantFunction(dim_domain=2),),
                 diffusion_functionals=None,
                 dirichlet_data=None, neumann_data=None,
                 parameter_space=None, name=None):
        assert rhs.dim_domain == diffusion_functions[0].dim_domain
        assert dirichlet_data is None or dirichlet_data.dim_domain == diffusion_functions[0].dim_domain
        assert neumann_data is None or neumann_data.dim_domain == diffusion_functions[0].dim_domain
        self.domain = domain
        self.rhs = rhs
        self.diffusion_functions = diffusion_functions
        self.diffusion_functionals = diffusion_functionals
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.parameter_space = parameter_space
        self.name = name
