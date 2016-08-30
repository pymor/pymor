# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.cache import CacheableInterface, cached
from pymor.core.interfaces import abstractmethod
from pymor.parameters.base import Parametric


class DiscretizationInterface(CacheableInterface, Parametric):
    """Interface for discretization objects.

    A discretization object defines a discrete problem
    via its `class` and the |Operators| it contains.
    Furthermore, discretizatoins can be
    :meth:`solved <DiscretizationInterface.solve>` for a given
    |Parameter| resulting in a solution |VectorArray|.

    Attributes
    ----------
    solution_space
        |VectorSpace| of the |VectorArrays| returned by :meth:`solve`.
    linear
        `True` if the discretization describes a linear problem.
    operators
        Dictionary of all |Operators| contained in the discretization
        (see :func:`pymor.reductors.basic.reduce_generic_rb` for a usage
        example).
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

    def solve(self, mu=None, **kwargs):
        """Solve the discrete problem for the |Parameter| `mu`.

        The result will be :mod:`cached <pymor.core.cache>`
        in case caching has been activated for the given discretization.

        Parameters
        ----------
        mu
            |Parameter| for which to solve.

        Returns
        -------
        The solution given as a |VectorArray|.
        """
        mu = self.parse_parameter(mu)
        return self.cached_method_call(self._solve, mu=mu, **kwargs)

    def estimate(self, U, mu=None):
        """Estimate the discretization error for a given solution.

        The discretization error could be the error w.r.t. the analytical
        solution of the given problem or the model reduction error w.r.t.
        a corresponding high-dimensional |Discretization|.

        Parameters
        ----------
        U
            The solution obtained by :meth:`~solve`.
        mu
            |Parameter| for which `U` has been obtained.

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
