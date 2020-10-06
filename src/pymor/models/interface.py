# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import abstractmethod
from pymor.core.cache import CacheableObject
from pymor.operators.constructions import induced_norm
from pymor.parameters.base import ParametricObject, Mu
from pymor.tools.frozendict import FrozenDict
from pymor.tools.deprecated import Deprecated
from pymor.vectorarrays.interface import VectorArray


class Model(CacheableObject, ParametricObject):
    """Interface for model objects.

    A model object defines a discrete problem
    via its `class` and the |Operators| it contains.
    Furthermore, models can be
    :meth:`solved <Model.solve>` for given
    |parameter values| resulting in a solution |VectorArray|.

    Attributes
    ----------
    solution_space
        |VectorSpace| of the solution |VectorArrays| returned by :meth:`solve`.
    output_space
        |VectorSpace| of the model output |VectorArrays| returned by
        :meth:`output` (typically `NumpyVectorSpace(k)` where `k` is a small).
    linear
        `True` if the model describes a linear problem.
    products
        Dict of inner product operators associated with the model.
    """

    solution_space = None
    output_space = None
    linear = False
    products = FrozenDict()

    def __init__(self, products=None, error_estimator=None, visualizer=None,
                 name=None, **kwargs):
        products = FrozenDict(products or {})
        if products:
            for k, v in products.items():
                setattr(self, f'{k}_product', v)
                setattr(self, f'{k}_norm', induced_norm(v))

        self.__auto_init(locals())

    def _compute(self, solution=False, output=False,
                 solution_error_estimate=False, output_error_estimate=False,
                 mu=None, **kwargs):
        return {}

    def _compute_solution(self, mu=None, **kwargs):
        raise NotImplementedError

    def _compute_output(self, solution, mu=None, **kwargs):
        if not hasattr(self, 'output_functional'):
            raise NotImplementedError
        if self.output_functional is None:
            raise ValueError('Model has no output')
        return self.output_functional.apply(solution, mu=mu)

    def _compute_solution_error_estimate(self, solution, mu=None, **kwargs):
        if self.error_estimator is None:
            raise ValueError('Model has no error estimator')
        return self.error_estimator.estimate_error(solution, mu, self)

    def _compute_output_error_estimate(self, solution, mu=None, **kwargs):
        if self.error_estimator is None:
            raise ValueError('Model has no error estimator')
        return self.error_estimator.estimate_output_error(solution, mu, self)

    _compute_allowed_kwargs = frozenset()

    def compute(self, solution=False, output=False,
                solution_error_estimate=False, output_error_estimate=False, *,
                mu=None, **kwargs):

        # make sure no unknown kwargs are passed
        assert kwargs.keys() <= self._compute_allowed_kwargs

        # parse parameter values
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)

        # log output
        # explicitly checking if logging is disabled saves some cpu cycles
        if not self.logging_disabled:
            self.logger.info(f'Solving {self.name} for {mu} ...')

        # first call _compute to give subclasses more control
        data = self._compute(solution=solution, output=output,
                             solution_error_estimate=solution_error_estimate,
                             output_error_estimate=output_error_estimate,
                             mu=mu, **kwargs)

        if (solution or output or solution_error_estimate or output_error_estimate) and \
                'solution' not in data:
            retval = self.cached_method_call(self._compute_solution, mu=mu, **kwargs)
            if isinstance(retval, dict):
                assert 'solution' in retval
                data.update(retval)
            else:
                data['solution'] = retval

        if output and 'output' not in data:
            # TODO use caching here (requires skipping args in key generation)
            retval = self._compute_output(data['solution'], mu=mu, **kwargs)
            if isinstance(retval, dict):
                assert 'output' in retval
                data.update(retval)
            else:
                data['output'] = retval

        if solution_error_estimate and 'solution_error_estimate' not in data:
            # TODO use caching here (requires skipping args in key generation)
            retval = self._compute_solution_error_estimate(data['solution'], mu=mu, **kwargs)
            if isinstance(retval, dict):
                assert 'solution_error_estimate' in retval
                data.update(retval)
            else:
                data['solution_error_estimate'] = retval

        if output_error_estimate and 'output_error_estimate' not in data:
            # TODO use caching here (requires skipping args in key generation)
            retval = self._compute_output_error_estimate(data['solution'], mu=mu, **kwargs)
            if isinstance(retval, dict):
                assert 'output_error_estimate' in retval
                data.update(retval)
            else:
                data['output_error_estimate'] = retval

        return data

    def solve(self, mu=None, return_error_estimate=False, **kwargs):
        """Solve the discrete problem for the |parameter values| `mu`.

        The result will be :mod:`cached <pymor.core.cache>`
        in case caching has been activated for the given model.

        Parameters
        ----------
        mu
            |Parameter values| for which to solve.
        return_output
            If `True`, the model output for the given |parameter values| `mu` is
            returned as a |VectorArray| from :attr:`output_space`.

        Returns
        -------
        The solution |VectorArray|. When `return_output` is `True`,
        the output |VectorArray| is returned as second value.
        """
        data = self.compute(
            solution=True,
            solution_error_estimate=return_error_estimate,
            mu=mu,
            **kwargs
        )
        if return_error_estimate:
            return data['solution'], data['solution_error_estimate']
        else:
            return data['solution']

    def output(self, mu=None, return_error_estimate=False, **kwargs):
        """Return the model output for given |parameter values| `mu`.

        Parameters
        ----------
        mu
            |Parameter values| for which to compute the output.

        Returns
        -------
        The computed model output as a |VectorArray| from `output_space`.
        """
        data = self.compute(
            output=True,
            output_error_estimate=return_error_estimate,
            mu=mu,
            **kwargs
        )
        if return_error_estimate:
            return data['output'], data['output_error_estimate']
        else:
            return data['output']

    def estimate_error(self, mu=None, **kwargs):
        """Estimate the model error for a given solution.

        The model error could be the error w.r.t. the analytical
        solution of the given problem or the model reduction error w.r.t.
        a corresponding high-dimensional |Model|.

        Parameters
        ----------
        U
            The solution obtained by :meth:`~solve`.
        mu
            |Parameter values| for which `U` has been obtained.

        Returns
        -------
        The estimated error.
        """
        return self.compute(
            solution_error_estimate=True,
            mu=mu,
            **kwargs
        )['solution_error_estimate']

    @Deprecated('estimate_error')
    def estimate(self, U, mu=None):
        return self.estimate_error(mu)

    def estimate_output_error(self, mu=None, **kwargs):
        """Estimate the model error for a given solution.

        The model error could be the error w.r.t. the analytical
        solution of the given problem or the model reduction error w.r.t.
        a corresponding high-dimensional |Model|.

        Parameters
        ----------
        U
            The solution obtained by :meth:`~solve`.
        mu
            |Parameter values| for which `U` has been obtained.

        Returns
        -------
        The estimated error.
        """
        return self.compute(
            output_error_estimate=True,
            mu=mu,
            **kwargs
        )['output_error_estimate']

    def visualize(self, U, **kwargs):
        """Visualize a solution |VectorArray| U.

        Parameters
        ----------
        U
            The |VectorArray| from
            :attr:`~pymor.models.interface.Model.solution_space`
            that shall be visualized.
        kwargs
            See docstring of `self.visualizer.visualize`.
        """
        if getattr(self, 'visualizer') is not None:
            return self.visualizer.visualize(U, self, **kwargs)
        else:
            raise NotImplementedError('Model has no visualizer.')
