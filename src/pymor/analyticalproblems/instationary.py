# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.parameters.base import ParametricObject, ParameterSpace
from pymor.tools.frozendict import FrozenDict


class InstationaryProblem(ParametricObject):
    """Instationary problem description.

    This class describes an instationary problem of the form ::

    |    ∂_t u(x, t, μ) + A(u(x, t, μ), t, μ) = f(x, t, μ),
    |                              u(x, 0, μ) = u_0(x, μ)

    where A, f are given by the problem's `stationary_part` and
    t is allowed to vary in the interval [0, T].

    Parameters
    ----------
    stationary_part
        The stationary part of the problem.
    initial_data
        |Function| providing the initial values u_0.
    T
        The final time T.
    parameter_ranges
        Ranges of interest for the |Parameters| of the problem.
    name
        Name of the problem.

    Attributes
    ----------
    T
    stationary_part
    parameter_ranges
    name
    """

    def __init__(self, stationary_part, initial_data, T=1., parameter_ranges=None, name=None):
        name = name or ('instationary_' + stationary_part.name)
        assert (initial_data is None
                or initial_data.dim_domain == stationary_part.domain.dim and initial_data.shape_range == ())
        assert (parameter_ranges is None
                or (isinstance(parameter_ranges, (list, tuple))
                    and len(parameter_ranges) == 2
                    and parameter_ranges[0] <= parameter_ranges[1])
                or (isinstance(parameter_ranges, dict)
                    and all(isinstance(v, (list, tuple)) and len(v) == 2 and v[0] <= v[1]
                            for v in parameter_ranges.values())))

        parameter_ranges = (
            None if parameter_ranges is None else
            tuple(parameter_ranges) if isinstance(parameter_ranges, (list, tuple)) else
            FrozenDict((k, tuple(v)) for k, v in parameter_ranges.items())
        )
        self.__auto_init(locals())

    def with_stationary_part(self, **kwargs):
        return self.with_(stationary_part=self.stationary_part.with_(**kwargs))

    @property
    def parameter_space(self):
        if self.parameter_ranges is None:
            return None
        else:
            return ParameterSpace(self.parameters, self.parameter_ranges)
