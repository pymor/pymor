# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.cache import CacheableObject
from pymor.operators.constructions import induced_norm
from pymor.parameters.base import ParametricObject, Mu
from pymor.tools.frozendict import FrozenDict
from pymor.tools.deprecated import Deprecated


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
    output_dim
        Dimension of the model output returned by :meth:`output`. 0 if the
        model has no output.
    linear
        `True` if the model describes a linear problem.
    products
        Dict of inner product operators associated with the model.
    """

    solution_space = None
    output_dim = 0
    linear = False
    products = FrozenDict()

    def __init__(self, products=None, error_estimator=None, visualizer=None,
                 name=None):
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
        """Compute the model's solution for |parameter values| `mu`.

        This method is called by the default implementation of :meth:`compute`
        in :class:`pymor.models.interface.Model`.

        Parameters
        ----------
        mu
            |Parameter values| for which to compute the solution.
        kwargs
            Additional keyword arguments to customize how the solution is
            computed or to select additional data to be returned.

        Returns
        -------
        |VectorArray| with the computed solution or a dict which at least
        must contain the key `'solution'`.
        """
        raise NotImplementedError

    def _compute_output(self, solution, mu=None, **kwargs):
        """Compute the model's output for |parameter values| `mu`.

        This method is called by the default implementation of :meth:`compute`
        in :class:`pymor.models.interface.Model`. The assumption is made
        that the output is a derived quantity from the model's internal state
        as returned my :meth:`_compute_solution`. When this is not the case,
        the computation of the output should be implemented in :meth:`_compute`.

        .. note::

            The default implementation applies the |Operator| given by the
            :attr:`output_functional` attribute to the given `solution`
            |VectorArray|.

        Parameters
        ----------
        solution
            Internal model state for the given |parameter values|.
        mu
            |Parameter values| for which to compute the output.
        kwargs
            Additional keyword arguments to customize how the output is
            computed or to select additional data to be returned.

        Returns
        -------
        |NumPy array| with the computed output or a dict which at least
        must contain the key `'output'`.
        """
        if not getattr(self, 'output_functional', None):
            return np.zeros(len(solution), 0)
        else:
            return self.output_functional.apply(solution, mu=mu).to_numpy()

    def _compute_solution_error_estimate(self, solution, mu=None, **kwargs):
        """Compute an error estimate for the computed internal state.

        This method is called by the default implementation of :meth:`compute`
        in :class:`pymor.models.interface.Model`. The assumption is made
        that the error estimate is a derived quantity from the model's internal state
        as returned my :meth:`_compute_solution`. When this is not the case,
        the computation of the error estimate should be implemented in :meth:`_compute`.

        .. note::

            The default implementation calls the `estimate_error` method of the object
            given by the :attr:`error_estimator` attribute, passing `solution`,
            `mu`, `self` and `**kwargs`.

        Parameters
        ----------
        solution
            Internal model state for the given |parameter values|.
        mu
            |Parameter values| for which to compute the error estimate.
        kwargs
            Additional keyword arguments to customize how the error estimate is
            computed or to select additional data to be returned.

        Returns
        -------
        The computed error estimate or a dict which at least must contain the key
        `'solution_error_estimate'`.
        """
        if self.error_estimator is None:
            raise ValueError('Model has no error estimator')
        return self.error_estimator.estimate_error(solution, mu, self, **kwargs)

    def _compute_output_error_estimate(self, solution, mu=None, **kwargs):
        """Compute an error estimate for the computed model output.

        This method is called by the default implementation of :meth:`compute`
        in :class:`pymor.models.interface.Model`. The assumption is made
        that the error estimate is a derived quantity from the model's internal state
        as returned my :meth:`_compute_solution`. When this is not the case,
        the computation of the error estimate should be implemented in :meth:`_compute`.

        .. note::

            The default implementation calls the `estimate_output_error` method of the object
            given by the :attr:`error_estimator` attribute, passing `solution`,
            `mu`, `self` and `**kwargs`.

        Parameters
        ----------
        solution
            Internal model state for the given |parameter values|.
        mu
            |Parameter values| for which to compute the error estimate.
        kwargs
            Additional keyword arguments to customize how the error estimate is
            computed or to select additional data to be returned.

        Returns
        -------
        The computed error estimate or a dict which at least must contain the key
        `'solution_error_estimate'`.
        """
        if self.error_estimator is None:
            raise ValueError('Model has no error estimator')
        return self.error_estimator.estimate_output_error(solution, mu, self, **kwargs)

    _compute_allowed_kwargs = frozenset()

    def compute(self, solution=False, output=False,
                solution_error_estimate=False, output_error_estimate=False, *,
                mu=None, **kwargs):
        """Compute the solution of the model and associated quantities.

        This methods computes the output of the model it's internal state
        and various associated quantities for given |parameter values|
        `mu`.

        .. note::

            The default implementation defers the actual computations to
            the methods :meth:`_compute_solution`, :meth:`_compute_output`,
            :meth:`_compute_solution_error_estimate` and :meth:`_compute_output_error_estimate`.
            The call to :meth:`_compute_solution` is :mod:`cached <pymor.core.cache>`.
            In addition, |Model| implementors may implement :meth:`_compute` to
            simultaneously compute multiple values in an optimized way. The corresponding
            `_compute_XXX` methods will not be called for values already returned by
            :meth:`_compute`.

        Parameters
        ----------
        solution
            If `True`, return the model's internal state.
        output
            If `True`, return the model output.
        solution_error_estimate
            If `True`, return an error estimate for the computed internal state.
        output_error_estimate
            If `True`, return an error estimate for the computed output.
        mu
            |Parameter values| for which to compute the values.
        kwargs
            Further keyword arguments to select further quantities that sould
            be returned or to customize how the values are computed.

        Returns
        -------
        A dict with the computed values.
        """

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

        This method returns a |VectorArray| with a internal state
        representation of the model's solution for given
        |parameter values|. It is a convenience wrapper around
        :meth:`compute`.

        The result may be :mod:`cached <pymor.core.cache>`
        in case caching has been activated for the given model.

        Parameters
        ----------
        mu
            |Parameter values| for which to solve.
        return_error_estimate
            If `True`, also return an error estimate for the computed solution.
        kwargs
            Additional keyword arguments passed to :meth:`compute` that
            might affect how the solution is computed.

        Returns
        -------
        The solution |VectorArray|. When `return_error_estimate` is `True`,
        the estimate is returned as second value.
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

        This method is a convenience wrapper around :meth:`compute`.

        Parameters
        ----------
        mu
            |Parameter values| for which to compute the output.
        return_error_estimate
            If `True`, also return an error estimate for the computed output.
        kwargs
            Additional keyword arguments passed to :meth:`compute` that
            might affect how the solution is computed.

        Returns
        -------
        The computed model output as a 2D |NumPy array|. The dimension
        of axis 1 is :attr:`output_dim`. (For stationary problems, axis 0 has
        dimension 1. For time-dependent problems, the dimension of axis 0
        depends on the number of time steps.)
        When `return_error_estimate` is `True`, the estimate is returned as
        second value.
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
        """Estimate the error for the computed internal state.

        For given |parameter values| `mu` this method returns an
        error estimate for the computed internal model state as returned
        by :meth:`solve`. It is a convenience wrapper around
        :meth:`compute`.

        The model error could be the error w.r.t. the analytical
        solution of the given problem or the model reduction error w.r.t.
        a corresponding high-dimensional |Model|.

        Parameters
        ----------
        mu
            |Parameter values| for which to estimate the error.
        kwargs
            Additional keyword arguments passed to :meth:`compute` that
            might affect how the error estimate (or the solution) is computed.

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
        """Estimate the error for the computed output.

        For given |parameter values| `mu` this method returns an
        error estimate for the computed model output as returned
        by :meth:`output`. It is a convenience wrapper around
        :meth:`compute`.

        The output error could be the error w.r.t. the analytical
        solution of the given problem or the model reduction error w.r.t.
        a corresponding high-dimensional |Model|.

        Parameters
        ----------
        mu
            |Parameter values| for which to estimate the error.
        kwargs
            Additional keyword arguments passed to :meth:`compute` that
            might affect how the error estimate (or the output) is computed.

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
        """Visualize a |VectorArray| U of the model's :attr:`solution_space`.

        Parameters
        ----------
        U
            The |VectorArray| from :attr:`solution_space`
            that shall be visualized.
        kwargs
            Additional keyword arguments to customize the visualization.
            See the docstring of `self.visualizer.visualize`.
        """
        if getattr(self, 'visualizer') is not None:
            return self.visualizer.visualize(U, self, **kwargs)
        else:
            raise NotImplementedError('Model has no visualizer.')
