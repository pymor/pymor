# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
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

    with_arguments = None  # with_arguments is an read-only property in the base class
    _own_with_arguments = frozenset({'stationary_part', 'initial_data', 'T', 'parameter_space', 'name'})

    def __init__(self, stationary_part, initial_data, T=1., parameter_space=None, name=None):

        self.stationary_part = stationary_part
        self.initial_data = initial_data
        self.T = T
        self.parameter_space = parameter_space or stationary_part.parameter_space
        self.name = name or ('instationary_' + stationary_part.name)
        self.with_arguments = self._own_with_arguments.union(stationary_part.with_arguments)

    def with_(self, **kwargs):
        arguments = {k: kwargs.pop(k, getattr(self, k)) for k in self._own_with_arguments}
        arguments['stationary_part'] = arguments['stationary_part'].with_(**kwargs)
        return InstationaryProblem(**arguments)
