# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import pymor.core as core
from pymor.tools import Named
from pymor.parameters.base import Parametric


class ParameterSpaceInterface(core.BasicInterface):
    '''Represents a parameter space.

    Attributes
    ----------
    parameter_type
        Parameter type of the space.
    '''

    parameter_type = None

    @core.interfaces.abstractmethod
    def contains(self, mu):
        '''True if `mu` is contained in the space.'''
        pass


class ParameterFunctionalInterface(core.BasicInterface, Parametric, Named):
    '''Represents a functional on a parameter space.
    '''

    def __init__(self):
        Parametric.__init__(self)

    @core.interfaces.abstractmethod
    def evaluate(self, mu=None):
        pass

    def __call__(self, mu=None):
        return self.evaluate(mu)
