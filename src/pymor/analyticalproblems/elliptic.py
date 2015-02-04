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
    """Affinely decomposed linear elliptic problem.

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
        assert isinstance(diffusion_functions, (tuple, list))
        assert diffusion_functionals is None and len(diffusion_functions) == 1 \
            or len(diffusion_functionals) == len(diffusion_functions)
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
