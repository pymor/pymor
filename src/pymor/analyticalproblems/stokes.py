# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.parameters.base import ParameterSpace, ParametricObject
from pymor.parameters.functionals import ParameterFunctional
from pymor.tools.frozendict import FrozenDict


class StokesProblem(ParametricObject):
    r"""Problem discretization of a Stokes equation.

    The problem consists in solving

    .. math::

        - \\mu \\Delta u(x, \\mu) + \nabla p(x, \\mu) & = f(x) \text{ in } \\Omega \\
        \nabla \\cdot u(x, \\mu) & = 0 \text{ in } \\Omega

    for u and p.

    Parameters
    ----------
    domain
        A |DomainDescription| of the domain the problem is posed on.
    rhs
        The |Function| f. `rhs.dim_domain` has to agree with the
        dimension of `domain`, whereas `rhs.shape_range` has to be `(dim domain,)`.
    viscosity
        The |ParameterFunctional| representing the viscosity coefficient mu.
    dirichlet_data
        |Function| providing the Dirichlet boundary values.
        `dirichlet_data.shape_range` has to be `(dim domain,)`.
    parameter_ranges
        Ranges of interest for the |Parameter| of the problem.
    name
        Name of the problem.

    Attributes
    ----------
    domain
    rhs
    dirichlet_data
    viscosity
    parameter_ranges
    name
    """

    def __init__(self, domain, viscosity, rhs=None, dirichlet_data=None, parameter_ranges=None, name=None):

        assert isinstance(viscosity, ParameterFunctional)
        assert (rhs is None
                or rhs.dim_domain == domain.dim and rhs.shape_range == (domain.dim,))
        assert (dirichlet_data is None
                or dirichlet_data.dim_domain == domain.dim and dirichlet_data.shape_range == (domain.dim,))
        assert (parameter_ranges is None
                or (isinstance(parameter_ranges, list | tuple)
                    and len(parameter_ranges) == 2
                    and parameter_ranges[0] <= parameter_ranges[1])
                or (isinstance(parameter_ranges, dict)
                    and all(isinstance(v, list | tuple) and len(v) == 2 and v[0] <= v[1]
                            for v in parameter_ranges.values())))

        parameter_ranges = (
            None if parameter_ranges is None else
            tuple(parameter_ranges) if isinstance(parameter_ranges, list | tuple) else
            FrozenDict((k, tuple(v)) for k, v in parameter_ranges.items())
        )

        self.__auto_init(locals())

    @property
    def parameter_space(self):
        if self.parameter_ranges is None:
            return None
        else:
            return ParameterSpace(self.parameters, self.parameter_ranges)
