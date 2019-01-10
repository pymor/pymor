# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.interfaces import ImmutableInterface
from pymor.tools.frozendict import FrozenDict


class StationaryProblem(ImmutableInterface):
    """Linear elliptic problem description.

    The problem consists in solving ::

        - ∇ ⋅ [d(x, μ) ∇ u(x, μ)] + ∇ ⋅ [f(x, u(x, μ), μ)] + c(x, u(x, μ), μ) = f(x, μ)

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
        The |Function| f, only depending on x, with `shape_range` of `(dim domain,)`.
    nonlinear_advection
        The |Function| f, only depending on u, with `shape_range` of `(dim domain,)`.
    nonlinear_advection_derivative
        The derivative of f, only depending on u, with respect to u.
    reaction
        The |Function| c, only depending on x, with `shape_range` of `()`.
    nonlinear_reaction
        The |Function| c, only depending on u, with `shape_range` of `()`.
    nonlinear_reaction_derivative
        The derivative of the |Function| c, only depending on u, with `shape_range` of `()`.
    dirichlet_data
        |Function| providing the Dirichlet boundary values.
    neumann_data
        |Function| providing the Neumann boundary values.
    robin_data
        Tuple of two |Functions| providing the Robin parameter and boundary values.
    functionals
        `Dict` of additional functionals to assemble. Each value must be a tuple
        of the form `(functional_type, data)` where `functional_type` is a string
        defining the type of functional to assemble and `data` is a |Function| holding
        the corresponding coefficient function. Currently implemented `functional_types`
        are:

            :l2:            Evaluate the l2-product with the given data function.
            :l2_boundary:   Evaluate the l2-product with the given data function
                            on the boundary.
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
    nonlinear_advection
    nonlinear_advection_derivative
    reaction
    nonlinear_reaction
    nonlinear_reaction_derivative
    dirichlet_data
    neumann_data
    robin_data
    functionals
    """

    def __init__(self, domain,
                 rhs=None, diffusion=None,
                 advection=None, nonlinear_advection=None, nonlinear_advection_derivative=None,
                 reaction=None, nonlinear_reaction=None, nonlinear_reaction_derivative=None,
                 dirichlet_data=None, neumann_data=None, robin_data=None, functionals=None,
                 parameter_space=None, name=None):

        assert rhs is None \
            or rhs.dim_domain == domain.dim and rhs.shape_range == ()
        assert diffusion is None \
            or diffusion.dim_domain == domain.dim and diffusion.shape_range in ((), (domain.dim, domain.dim))
        assert advection is None \
            or advection.dim_domain == domain.dim and advection.shape_range == (domain.dim,)
        assert nonlinear_advection is None \
            or nonlinear_advection.dim_domain == 1 and nonlinear_advection.shape_range == (domain.dim,)
        assert nonlinear_advection_derivative is None \
            or (nonlinear_advection_derivative.dim_domain == 1 and
                nonlinear_advection_derivative.shape_range == (domain.dim,))
        assert reaction is None \
            or reaction.dim_domain == domain.dim and reaction.shape_range == ()
        assert nonlinear_reaction is None \
            or nonlinear_reaction.dim_domain == 1 and nonlinear_reaction.shape_range == ()
        assert nonlinear_reaction_derivative is None \
            or nonlinear_reaction_derivative.dim_domain == 1 and nonlinear_reaction_derivative.shape_range == ()
        assert dirichlet_data is None \
            or dirichlet_data.dim_domain == domain.dim and dirichlet_data.shape_range == ()
        assert neumann_data is None \
            or neumann_data.dim_domain == domain.dim and neumann_data.shape_range == ()
        assert robin_data is None \
            or (isinstance(robin_data, tuple) and len(robin_data) == 2 and
                np.all([f.dim_domain == domain.dim and f.shape_range == () for f in robin_data]))
        assert functionals is None \
            or all(isinstance(v, tuple) and len(v) == 2 and v[0] in ('l2', 'l2_boundary') and
                   v[1].dim_domain == domain.dim and v[1].shape_range == () for v in functionals.values())

        self.domain = domain
        self.rhs = rhs
        self.diffusion = diffusion
        self.advection = advection
        self.nonlinear_advection = nonlinear_advection
        self.nonlinear_advection_derivative = nonlinear_advection_derivative
        self.reaction = reaction
        self.nonlinear_reaction = nonlinear_reaction
        self.nonlinear_reaction_derivative = nonlinear_reaction_derivative
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.robin_data = robin_data
        self.functionals = FrozenDict(functionals) if functionals is not None else None
        self.parameter_space = parameter_space
        self.name = name
