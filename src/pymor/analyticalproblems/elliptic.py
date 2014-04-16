# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core import ImmutableInterface
from pymor.domaindescriptions import RectDomain
from pymor.functions import ConstantFunction
from pymor.tools import Named


class EllipticProblem(ImmutableInterface, Named):
    '''Linear elliptic analytical problem.

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
    name
        Name of the problem.

    Attributes
    ----------
    domain
    rhs
    diffusion_functions
    diffusion_functionals
    dirichlet_data
    '''

    def __init__(self, domain=RectDomain(), rhs=ConstantFunction(dim_domain=2),
                 diffusion_functions=(ConstantFunction(dim_domain=2),),
                 diffusion_functionals=None,
                 dirichlet_data=ConstantFunction(value=0, dim_domain=2), name=None):
        self.domain = domain
        self.rhs = rhs
        self.diffusion_functions = diffusion_functions
        self.diffusion_functionals = diffusion_functionals
        self.dirichlet_data = dirichlet_data
        self.name = name
