# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.interfaces import ImmutableInterface


class InstationaryProblem(ImmutableInterface):
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
    parameter_space
        |ParameterSpace| for the problem.
    name
        Name of the problem.

    Attributes
    ----------
    T
    stationary_part
    parameter_space
    name
    """

    def __init__(self, stationary_part, initial_data, T=1., parameter_space=None, name=None):
        name = name or ('instationary_' + stationary_part.name)
        self.__auto_init(locals())

    def with_stationary_part(self, **kwargs):
        return self.with_(stationary_part=self.stationary_part.with_(**kwargs))
