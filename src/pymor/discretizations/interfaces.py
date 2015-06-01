# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core.cache import CacheableInterface, cached
from pymor.core.interfaces import abstractmethod
from pymor.parameters.base import Parametric


class DiscretizationInterface(CacheableInterface, Parametric):
    """Describes a discretization.

    A discretization is an object describing a discrete problem
    via its type and the |Operators| it contains and which can
    :meth:`solved <DiscretizationInterface.solve>` for a given
    |Parameter| resulting in a solution |VectorArray|.

    Attributes
    ----------
    solution_space
        |VectorSpace| of the |VectorArrays| returned by solve.
    linear
        `True` if the discretization describes a linear Problem.
    operators
        Dictionary of all |Operators| contained in the discretization.
        (Compare the implementation of :func:`pymor.reductors.basic.reduce_generic_rb`.)
    functionals
        Same as `operators` but for |Functionals|.
    vector_operators
        Same as operators but for |Operators| representing vectors, i.e.
        linear |Operators| with `source.dim == 1`.
    products
        Same as |Operators| but for inner product operators associated with the
        discretization.
    """

    solution_space = None
    linear = False
    operators = dict()
    functionals = dict()
    vector_operators = dict()
    products = dict()

    @abstractmethod
    def _solve(self, mu=None):
        """Perform the actual solving."""
        pass

    @cached
    def solve(self, mu=None, **kwargs):
        """Solve for the |Parameter| `mu`.

        The result will be :mod:`cached <pymor.core.cache>` by default.

        Parameters
        ----------
        mu
            |Parameter| for which to solve.

        Returns
        -------
        The solution given as a |VectorArray|.
        """
        return self._solve(mu, **kwargs)

    def estimate(self, U, mu=None):
        """Estimate the discretization error for a given solution.

        Parameters
        ----------
        U
            The solution obtained by :meth:`~DiscretizationInterface.solve`.
        mu
            Parameter for which `U` has been obtained.

        Returns
        -------
        The estimated error.
        """
        raise NotImplementedError

    def visualize(self, U, **kwargs):
        """Visualize a solution |VectorArray| U.

        Parameters
        ----------
        U
            The |VectorArray| from :attr:`~DiscretizationInterface.solution_space`
            that shall be visualized.
        """
        raise NotImplementedError
