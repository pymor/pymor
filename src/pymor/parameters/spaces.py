# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.base import ImmutableObject, abstractmethod
from pymor.parameters.base import Mu, Parameters


class ParameterSpace(ImmutableObject):
    """Interface for |Parameter| spaces.

    Attributes
    ----------
    parameters
        |ParameterType| of the space.
    """

    parameters = Parameters({})

    @abstractmethod
    def contains(self, mu):
        """`True` if `mu` is contained in the space."""
        pass


class CubicParameterSpace(ParameterSpace):
    """Simple |ParameterSpace| where each summand is an n-cube.

    Parameters
    ----------
    parameters
        The |ParameterType| of the space.
    minimum
        The minimum for each matrix entry of each |Parameter| component.
        Must be `None` if `ranges` is specified.
    maximum
        The maximum for each matrix entry of each |Parameter| component.
        Must be `None` if `ranges` is specified.
    ranges
        dict whose keys agree with `parameters` and whose values
        are tuples (min, max) specifying the minimum and maximum of each
        matrix entry of corresponding |Parameter| component.
        Must be `None` if `minimum` and `maximum` are specified.
    """

    def __init__(self, parameters, minimum=None, maximum=None, ranges=None):
        assert ranges is None or (minimum is None and maximum is None), 'Must specify minimum, maximum or ranges'
        assert ranges is not None or (minimum is not None and maximum is not None),\
            'Must specify minimum, maximum or ranges'
        assert minimum is None or minimum < maximum
        if ranges is None:
            ranges = {k: (minimum, maximum) for k in parameters}
        parameters = Parameters(parameters)
        self.__auto_init(locals())

    def contains(self, mu):
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        if not mu >= self.parameters:
            return False
        return all(np.all(self.ranges[k][0] <= mu[k]) and np.all(mu[k] <= self.ranges[k][1])
                   for k in self.parameters)

    def sample_uniformly(self, counts):
        """Uniformly sample |Parameters| from the space."""
        return self.parameters.sample_uniformly(counts, self.ranges)

    def sample_randomly(self, count=None, random_state=None, seed=None):
        """Randomly sample |Parameters| from the space.

        Parameters
        ----------
        count
            `None` or number of random parameters (see below).
        random_state
            :class:`~numpy.random.RandomState` to use for sampling.
            If `None`, a new random state is generated using `seed`
            as random seed, or the :func:`default <pymor.tools.random.default_random_state>`
            random state is used.
        seed
            If not `None`, a new radom state with this seed is used.

        Returns
        -------
        If `count` is `None`, an inexhaustible iterator returning random
        |Parameters|.
        Otherwise a list of `count` random |Parameters|.
        """
        return self.parameters.sample_randomly(count, self.ranges, random_state, seed)

    def __str__(self):
        rows = [(k, str(v), str(self.ranges[k])) for k, v in self.parameters.items()]
        column_widths = [max(map(len, c)) for c in zip(*rows)]
        return ('CubicParameterSpace\n'
                + '\n'.join(('key: {:' + str(column_widths[0] + 2)
                             + '} shape: {:' + str(column_widths[1] + 2)
                             + '} range: {}').format(c1 + ',', c2 + ',', c3) for (c1, c2, c3) in rows))
