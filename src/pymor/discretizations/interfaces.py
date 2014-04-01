# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core.cache import CacheableInterface, cached
from pymor.core.interfaces import abstractmethod
from pymor.parameters import Parametric
from pymor.tools import Named


class DiscretizationInterface(CacheableInterface, Parametric, Named):
    '''Describes a discretization.

    Note that we do not make any distinction between detailed and reduced
    discretizations.

    Attributes
    ----------
    dim_solution
        Dimension of the |VectorArrays| returned by solve.
    type_solution
        Type of the |VectorArrays| returned by solve.
    linear
        `True` if the discretization describes a linear Problem.
    operators
        Dictionary of all |Operators| contained in the discretization.
        (Compare the implementation of :func:`pymor.reductors.basic.reduce_generic_rb`.)
    functionals
        Same as operators but for |Functionals|.
    vector_operators
        Same as operators but for |Operators| representing vectors, i.e.
        linear |Operators| with `dim_source == 1`.
    products
        Same as |Operators| but for inner product operators associated to the
        discretization.

    Optional Methods:

        def estimate(self, U, mu=None):
            Estimate the error of the discrete solution `U` to the |Parameter| `mu` against
            the real solution. (For a reduced discretization, the 'real' solution will
            be the solution of a detailed discretization, in general.)

        def visualize(self, U):
            Visualize a solution given by the |VectorArray| U.
    '''

    dim_solution = None
    type_solution = None
    linear = False
    operators = dict()
    functionals = dict()
    vector_operators = dict()
    products = dict()
    with_arguments = frozenset({'operators', 'functionals', 'vector_operators, products'})

    @abstractmethod
    def _solve(self, mu=None):
        '''Perform the actual solving.'''
        pass

    @cached
    def solve(self, mu=None):
        '''Solve for the |Parameter| `mu`.

        The result is cached by default.

        Parameters
        ----------
        mu
            |Parameter| for which to solve.

        Returns
        -------
        The solution given by a |VectorArray|.
        '''
        return self._solve(mu)
