# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core import ImmutableInterface
from pymor.domaindescriptions import RectDomain
from pymor.functions import ConstantFunction
from pymor.tools import Named


class EllipticProblem(ImmutableInterface, Named):
    '''Standard elliptic analytical problem.

    The problem consists in solving ::

      |       K
      | - ∇ ⋅ ∑  θ_k(μ) ⋅ d_k(x) ∇ u(x, μ) = f(x, μ)
      |      k=0

    for u.

    Parameters
    ----------
    domain
        A domain description of the domain the problem is posed on.
    rhs
        The function f(x, mu).
    diffusion_functions
        List of the functions d_k(x).
    diffusion_functionals
        List of the functionals theta_k(mu). If None, and `len(diffusion_functions) > 1`
        let theta_k be the kth projection of the coefficient part of mu.
        If None and `len(diffusion_functions) == 1`, no parameter dependence is
        assumed.
    dirichlet_data
        Function providing the Dirichlet boundary values in global coordinates.
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
