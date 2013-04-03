# pymor (http://www.pymor.org)
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

    Inherits
    --------
    BasicInterface, Parametric, Cachable, Named
    '''

    operators = dict()

    def __init__(self):
        Cachable.__init__(self, config=DEFAULT_DISK_CONFIG)

    def copy(self):
        c = copy.copy(self)
        c.operators = c.operators.copy()
        Cachable.__init__(c)
        return c

    @abstractmethod
    def _solve(self, mu={}):
        '''Perform the actual solving.'''
        pass

    @cached
    def solve(self, mu={}):
        '''Solve for a parameter `mu`.

        The result is cached by default.
        '''
        return self._solve(mu)
