# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import copy

from pymor.core.interfaces import abstractmethod
from pymor.core.cache import CacheableInterface, cached
from pymor.tools import Named
from pymor.parameters import Parametric


class DiscretizationInterface(CacheableInterface, Parametric, Named):
    '''Describes a discretization.

    Note that we do not make any distinction between detailed and reduced
    discretizations at this point.

    Attributes
    ----------
    dim_solution
        Dimension of the `VectorArrays` returned by solve.
    type_solution
        Type of the `VectorArrays` returned by solve.
    linear
        True if the discretization describes a linear Problem.
    operators
        Dictionary of all operators contained in this discretization. The idea is
        that this attribute will be common to all discretizations such that it can
        be used for introspection. Compare the implementation of `reduce_generic_rb`.
        For this class, operators has the keys 'operator' and 'rhs'.

    Optional Methods
    ----------------
    def estimate(self, U, mu=None):
        Estimate the error of the discrete solution U to the parameter mu against
        the real solution. (For a reduced discretization, the 'real' solution will
        in genereal be the solution of a detailed discretization.)

    def visualize(self, U):
        Visualize a solution given by the VectorArray U.
    '''

    dim_solution = None
    type_solution = None
    linear = False
    operators = dict()
    with_arguments = set(('operators',))

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
