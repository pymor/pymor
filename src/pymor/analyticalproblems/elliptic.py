# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.interfaces import ImmutableInterface
from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction


class EllipticProblem(ImmutableInterface):
    """Linear elliptic problem description.

    The problem consists in solving ::

        - ∇ ⋅ [d(x, μ) ∇ u(x, μ)] + ∇ ⋅ [v(x, μ) u(x, μ)] + c(x, μ) u(x, μ) = f(x, μ)

    for u.

    Parameters
    ----------
    domain
        A |DomainDescription| of the domain the problem is posed on.
    rhs
        The |Function| f(x, μ). `rhs.dim_domain` has to agree with the
        dimension of `domain`, whereas `rhs.shape_range` has to be `()`.
    diffusion
        The |Function| d(x, μ) with `shape_range` of either `()` or
        `(dim domain, dim domain)`.
    advection
        The |Function| v(x, μ), with `shape_range` of `(dim domain,)`.
    reaction
        The |Function| c(x, μ), with `shape_range` of `()`.
    dirichlet_data
        |Function| providing the Dirichlet boundary values.
    neumann_data
        |Function| providing the Neumann boundary values.
    robin_data
        Tuple of two |Functions| providing the Robin parameter and boundary values.
    parameter_space
        |ParameterSpace| for the problem.
    name
        Name of the problem.

    Attributes
    ----------
    domain
    rhs
    diffusion
    advection
    reaction
    dirichlet_data
    neumann_data
    robin_data
    """

    def __init__(self, domain=RectDomain(), rhs=ConstantFunction(dim_domain=2),
                 diffusion=None, advection=None, reaction=None,
                 dirichlet_data=None, neumann_data=None, robin_data=None,
                 parameter_space=None, name=None):
        assert rhs.dim_domain == domain.dim
        assert diffusion is None or diffusion.dim_domain == domain.dim
        assert advection is None or advection.dim_domain == domain.dim
        assert reaction is None or reaction.dim_domain == domain.dim
        assert dirichlet_data is None or dirichlet_data.dim_domain == domain.dim
        assert neumann_data is None or neumann_data.dim_domain == domain.dim
        assert robin_data is None or (isinstance(robin_data, tuple) and len(robin_data) == 2)
        assert robin_data is None or np.all([f.dim_domain == domain.dim for f in robin_data])
        self.domain = domain
        self.rhs = rhs
        self.diffusion = diffusion
        self.advection = advection
        self.reaction = reaction
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.robin_data = robin_data
        self.parameter_space = parameter_space
        self.name = name
