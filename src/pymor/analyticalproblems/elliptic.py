# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.parameters.base import ParametricObject, ParameterSpace
from pymor.tools.frozendict import FrozenDict


class StationaryProblem(ParametricObject):
    """Linear elliptic problem description.

    The problem consists in solving ::

        - ∇ ⋅ [d(x, μ) ∇ u(x, μ)] + ∇ ⋅ [f_l(x, μ)u(x, μ)]
        + ∇ ⋅ f_n(u(x, μ), μ) + c_l(x, μ) + c_n(u(x, μ), μ) = g(x, μ)

    for u.

    Parameters
    ----------
    domain
        A |DomainDescription| of the domain the problem is posed on.
    rhs
        The |Function| g. `rhs.dim_domain` has to agree with the
        dimension of `domain`, whereas `rhs.shape_range` has to be `()`.
    diffusion
        The |Function| d with `shape_range` of either `()` or
        `(dim domain, dim domain)`.
    advection
        The |Function| f_l, only depending on x, with `shape_range` of `(dim domain,)`.
    nonlinear_advection
        The |Function| f_n, only depending on u, with `shape_range` of `(dim domain,)`.
    nonlinear_advection_derivative
        The derivative of f_n, only depending on u, with respect to u.
    reaction
        The |Function| c_l, only depending on x, with `shape_range` of `()`.
    nonlinear_reaction
        The |Function| c_n, only depending on u, with `shape_range` of `()`.
    nonlinear_reaction_derivative
        The derivative of the |Function| c_n, only depending on u, with `shape_range` of `()`.
    dirichlet_data
        |Function| providing the Dirichlet boundary values.
    neumann_data
        |Function| providing the Neumann boundary values.
    robin_data
        Tuple of two |Functions| providing the Robin parameter and boundary values.
    outputs
        Tuple of additional output functionals to assemble. Each value must be a tuple
        of the form `(functional_type, data)` where `functional_type` is a string
        defining the type of functional to assemble and `data` is a |Function| holding
        the corresponding coefficient function. Currently implemented `functional_types`
        are:

            :l2:            Evaluate the l2-product with the given data function.
            :l2_boundary:   Evaluate the l2-product with the given data function
                            on the boundary.
    parameter_ranges
        Ranges of interest for the |Parameters| of the problem.
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
    outputs
    """

    def __init__(self, domain,
                 rhs=None, diffusion=None,
                 advection=None, nonlinear_advection=None, nonlinear_advection_derivative=None,
                 reaction=None, nonlinear_reaction=None, nonlinear_reaction_derivative=None,
                 dirichlet_data=None, neumann_data=None, robin_data=None, outputs=None,
                 parameter_ranges=None, name=None):

        assert (rhs is None
                or rhs.dim_domain == domain.dim and rhs.shape_range == ())
        assert (diffusion is None
                or diffusion.dim_domain == domain.dim and diffusion.shape_range in ((), (domain.dim, domain.dim)))
        assert (advection is None
                or advection.dim_domain == domain.dim and advection.shape_range == (domain.dim,))
        assert (nonlinear_advection is None
                or nonlinear_advection.dim_domain == 1 and nonlinear_advection.shape_range == (domain.dim,))
        assert (nonlinear_advection_derivative is None
                or (nonlinear_advection_derivative.dim_domain == 1
                    and nonlinear_advection_derivative.shape_range == (domain.dim,)))
        assert (reaction is None
                or reaction.dim_domain == domain.dim and reaction.shape_range == ())
        assert (nonlinear_reaction is None
                or nonlinear_reaction.dim_domain == 1 and nonlinear_reaction.shape_range == ())
        assert (nonlinear_reaction_derivative is None
                or nonlinear_reaction_derivative.dim_domain == 1 and nonlinear_reaction_derivative.shape_range == ())
        assert (dirichlet_data is None
                or dirichlet_data.dim_domain == domain.dim and dirichlet_data.shape_range == ())
        assert (neumann_data is None
                or neumann_data.dim_domain == domain.dim and neumann_data.shape_range == ())
        assert (robin_data is None
                or (isinstance(robin_data, tuple) and len(robin_data) == 2
                    and np.all([f.dim_domain == domain.dim and f.shape_range == () for f in robin_data])))
        assert (outputs is None
                or all(isinstance(v, tuple) and len(v) == 2 and v[0] in ('l2', 'l2_boundary')
                       and v[1].dim_domain == domain.dim and v[1].shape_range == () for v in outputs))
        assert (parameter_ranges is None
                or (isinstance(parameter_ranges, (list, tuple))
                    and len(parameter_ranges) == 2
                    and parameter_ranges[0] <= parameter_ranges[1])
                or (isinstance(parameter_ranges, dict)
                    and all(isinstance(v, (list, tuple)) and len(v) == 2 and v[0] <= v[1]
                            for v in parameter_ranges.values())))

        outputs = tuple(outputs) if outputs is not None else None
        parameter_ranges = (
            None if parameter_ranges is None else
            tuple(parameter_ranges) if isinstance(parameter_ranges, (list, tuple)) else
            FrozenDict((k, tuple(v)) for k, v in parameter_ranges.items())
        )

        self.__auto_init(locals())

    @property
    def parameter_space(self):
        if self.parameter_ranges is None:
            return None
        else:
            return ParameterSpace(self.parameters, self.parameter_ranges)
