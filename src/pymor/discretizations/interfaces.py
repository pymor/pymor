# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import copy

from pymor.core import BasicInterface
from pymor.core.interfaces import abstractmethod
from pymor.core.cache import Cachable, cached, DEFAULT_DISK_CONFIG
from pymor.tools import Named
from pymor.parameters import Parametric


class DiscretizationInterface(BasicInterface, Parametric, Cachable, Named):
    '''Describes a discretization.

    Note that we do not make any distinction between detailed and reduced
    discretizations at this point.

    Attributes
    ----------
    operators
        Dictionary of all operators contained in this discretization. The idea is
        that this attribute will be common to all discretizations such that it can
        be used for introspection. Compare the implementation of `reduce_generic_rb`.
        For this class, operators has the keys 'operator' and 'rhs'.
    '''

    operators = dict()
    with_arguments = set(('operators',))

    def __init__(self, operators, estimator=None, visualizer=None, name=None):
        Cachable.__init__(self, config=DEFAULT_DISK_CONFIG)
        Parametric.__init__(self)
        self.operators = operators
        self.estimator = estimator
        self.visualizer = visualizer
        self.name = name

        if estimator is not None:
            self.estimate = self.__estimate
        if visualizer is not None:
            self.visualize = self.__visualize

    @abstractmethod
    def _solve(self, mu=None):
        '''Perform the actual solving.'''
        pass

    @cached
    def solve(self, mu=None):
        '''Solve for a parameter `mu`.

        The result is cached by default.
        '''
        return self._solve(mu)

    def __visualize(self, U):
        self.visualizer.visualize(U, self)

    def __estimate(self, U, mu=None):
        return self.estimator.estimate(U, mu=mu, discretization=self)
