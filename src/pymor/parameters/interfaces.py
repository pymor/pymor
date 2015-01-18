# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core.interfaces import ImmutableInterface, abstractmethod
from pymor.parameters.base import Parametric


class ParameterSpaceInterface(ImmutableInterface):
    """Interface for parameter spaces.

    Attributes
    ----------
    parameter_type
        |ParameterType| of the space.
    """

    parameter_type = None

    @abstractmethod
    def contains(self, mu):
        """`True` if `mu` is contained in the space."""
        pass


class ParameterFunctionalInterface(ImmutableInterface, Parametric):
    """Interface for parameter functionals.

    A parameter functional is simply a function mapping a |Parameter| to
    a number.
    """

    @abstractmethod
    def evaluate(self, mu=None):
        """Evaluate the functional for the given |Parameter| `mu`."""
        pass

    def __call__(self, mu=None):
        return self.evaluate(mu)
