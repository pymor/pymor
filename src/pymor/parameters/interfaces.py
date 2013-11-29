# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core import ImmutableInterface, abstractmethod
from pymor.parameters.base import Parametric
from pymor.tools import Named


class ParameterSpaceInterface(ImmutableInterface):
    '''Represents a parameter space.

    Attributes
    ----------
    parameter_type
        Parameter type of the space.
    '''

    parameter_type = None

    @abstractmethod
    def contains(self, mu):
        '''True if `mu` is contained in the space.'''
        pass


class ParameterFunctionalInterface(ImmutableInterface, Parametric, Named):
    '''Represents a functional on a parameter space.
    '''

    @abstractmethod
    def evaluate(self, mu=None):
        pass

    def __call__(self, mu=None):
        return self.evaluate(mu)
