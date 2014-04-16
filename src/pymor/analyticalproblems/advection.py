# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core import ImmutableInterface
from pymor.domaindescriptions import RectDomain
from pymor.functions import ConstantFunction
from pymor.tools import Named


class InstationaryAdvectionProblem(ImmutableInterface, Named):
    '''Instationary advection problem.

    The problem is to solve ::

        ∂_t u(x, t, μ)  +  ∇ ⋅ f(u(x, t, μ), t, μ) = s(x, t, μ)
                                        u(x, 0, μ) = u_0(x, μ)

    for u with t in [0, T], x in Ω.

    Parameters
    ----------
    domain
        A |DomainDescription| of the domain Ω the problem is posed on.
    flux_function
        The |Function| f. The current time is provided by adding the key `'_t'`
        to the |Parameter| `mu`.
    flux_function_derivative
        The derivative of f with respect to u.
    rhs
        The |Function| s. The current time is provided by adding the key `'_t'`
        to the |Parameter| `mu`.
    dirichlet_data
        |Function| providing the Dirichlet boundary values in global coordinates.
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
    flux_function
    flux_function_derivative
    initial_data
    dirichlet_data
    T
    '''

    def __init__(self, domain=RectDomain(), rhs=ConstantFunction(dim_domain=2),
                 flux_function=ConstantFunction(value=np.array([0, 0]), dim_domain=2),
                 flux_function_derivative=ConstantFunction(value=np.array([0, 0]), dim_domain=2),
                 dirichlet_data=ConstantFunction(value=0, dim_domain=2),
                 initial_data=ConstantFunction(value=1, dim_domain=2), T=1, name=None):
        self.domain = domain
        self.rhs = rhs
        self.flux_function = flux_function
        self.flux_function_derivative = flux_function_derivative
        self.dirichlet_data = dirichlet_data
        self.initial_data = initial_data
        self.T = T
        self.name = name
