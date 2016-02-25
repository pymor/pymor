# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.interfaces import ImmutableInterface
from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction


class EllipticProblem(ImmutableInterface):
    """Affinely decomposed linear elliptic problem.

    The problem consists in solving ::

    |        Kd                                     Kv                             Kr
    | - ∇ ⋅ ∑  θ_{d,k}(μ) ⋅ d_k(x) ∇ u(x, μ) + ∇ ⋅ ∑  θ_{v,k}(μ) v_k(x) u(x, μ) + ∑  θ_{r,k}(μ) r_k(x) u(x, μ) = f(x, μ)
    |       k=0                                    k=0                            k=0

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
        List containing the |ParameterFunctionals| θ_{d,k}(μ). If
        `len(diffusion_functions) == 1`, `diffusion_functionals` is allowed
        to be `None`, in which case no parameter dependence is assumed.
    advection_functions
        List containing the |Functions| v_k(x), each having `shape_range`
        of `(dim domain, )`.
    advection_functionals
        List containing the |ParameterFunctionals| θ_{v,k}(μ). If
        `len(advection_functions) == 1`, `advection_functionals` is allowed
        to be `None`, in which case no parameter dependence is assumed.
    reaction_functions
        List containing the |Functions| r_k(x), each having `shape_range`
        of `tuple()`.
    reaction_functionals
        List containing the |ParameterFunctionals| θ_{r,k}(μ). If
        `len(reaction_functions) == 1`, `reaction_functionals` is allowed
        to be `None`, in which case no parameter dependence is assumed.
    dirichlet_data
        |Function| providing the Dirichlet boundary values in global coordinates.
    neumann_data
        |Function| providing the Neumann boundary values in global coordinates.
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
    diffusion_functions
    diffusion_functionals
    advection_functions
    advection_functionals
    reaction_functions
    reaction_functionals
    dirichlet_data
    neumann_data
    robin_data
    """

    def __init__(self, domain=RectDomain(), rhs=ConstantFunction(dim_domain=2),
                 diffusion_functions=None,
                 diffusion_functionals=None,
                 advection_functions=None,
                 advection_functionals=None,
                 reaction_functions=None,
                 reaction_functionals=None,
                 dirichlet_data=None, neumann_data=None, robin_data=None,
                 parameter_space=None, name=None):
        assert diffusion_functions is None or isinstance(diffusion_functions, (tuple, list))
        assert advection_functions is None or isinstance(advection_functions, (tuple, list))
        assert reaction_functions is None or isinstance(reaction_functions, (tuple, list))

        assert diffusion_functionals is None and diffusion_functions is None \
            or diffusion_functionals is None and len(diffusion_functions) == 1 \
            or len(diffusion_functionals) == len(diffusion_functions)
        assert advection_functionals is None and advection_functions is None \
            or advection_functionals is None and len(advection_functions) == 1 \
            or len(advection_functionals) == len(advection_functions)
        assert reaction_functionals is None and reaction_functions is None \
            or reaction_functionals is None and len(reaction_functions) == 1 \
            or len(reaction_functionals) == len(reaction_functions)

        # for backward compatibility:
        if (diffusion_functions is None and advection_functions is None and reaction_functions is None):
            diffusion_functions = (ConstantFunction(dim_domain=2),)

        # dim_domain:
        if diffusion_functions is not None:
            dim_domain = diffusion_functions[0].dim_domain

        assert rhs.dim_domain == dim_domain
        if diffusion_functions is not None:
            for f in diffusion_functions:
                assert f.dim_domain == dim_domain
        if advection_functions is not None:
            for f in advection_functions:
                assert f.dim_domain == dim_domain
        if reaction_functions is not None:
            for f in reaction_functions:
                assert f.dim_domain == dim_domain

        assert dirichlet_data is None or dirichlet_data.dim_domain == dim_domain
        assert neumann_data is None or neumann_data.dim_domain == dim_domain
        assert robin_data is None or (isinstance(robin_data, tuple) and len(robin_data) == 2)
        assert robin_data is None or np.all([f.dim_domain == dim_domain for f in robin_data])
        self.domain = domain
        self.rhs = rhs
        self.diffusion_functions = diffusion_functions
        self.diffusion_functionals = diffusion_functionals
        self.advection_functions = advection_functions
        self.advection_functionals = advection_functionals
        self.reaction_functions = reaction_functions
        self.reaction_functionals = reaction_functionals
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.robin_data = robin_data
        self.parameter_space = parameter_space
        self.name = name
