# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.cache import CacheableObject
from pymor.operators.constructions import induced_norm
from pymor.parameters.base import Parameters, ParametricObject, Mu
from pymor.tools.frozendict import FrozenDict


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
    dim_output
        Dimension of the model output returned by :meth:`output`. 0 if the
        model has no output.
    linear
        `True` if the model describes a linear problem.
    products
        Dict of inner product operators associated with the model.
    order
        Dimension of the `solution_space`.
    """

    solution_space = None
    dim_output = 0
    linear = False
    products = FrozenDict()

    def __init__(self, dim_input=0, products=None, error_estimator=None, visualizer=None,
                 name=None):
        products = FrozenDict(products or {})
        if products:
            for k, v in products.items():
                setattr(self, f'{k}_product', v)
                setattr(self, f'{k}_norm', induced_norm(v))

        self.parameters_internal = {'input': dim_input}

        self.__auto_init(locals())

    @property
    def order(self):
        return self.solution_space.dim

    def _compute(self, solution=False, output=False, solution_d_mu=False, output_d_mu=False,
                 solution_error_estimate=False, output_error_estimate=False,
                 output_d_mu_return_array=False, output_error_estimate_return_vector=False,
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
        return self.output_functional.apply(solution, mu=mu).to_numpy()

    def _compute_solution_d_mu_single_direction(self, parameter, index, solution, mu=None, **kwargs):
        """Compute the partial derivative of the solution w.r.t. a parameter index

        Parameters
        ----------
        parameter
            parameter for which to compute the sensitivity
        index
            parameter index for which to compute the sensitivity
        solution
            Internal model state for the given |Parameter value|.
        mu
            |Parameter value| for which to solve

        Returns
        -------
        The sensitivity of the solution as a |VectorArray|.
        """
        raise NotImplementedError

    def _compute_solution_d_mu(self, solution, mu=None, **kwargs):
        """Compute all partial derivative of the solution w.r.t. a parameter index

        Parameters
        ----------
        solution
            Internal model state for the given |Parameter value|.
        mu
            |Parameter value| for which to solve

        Returns
        -------
        A dict of all partial sensitivities of the solution.
        """
        sensitivities = {}
        for (parameter, size) in self.parameters.items():
            sens_for_param = self.solution_space.empty()
            for l in range(size):
                sens_for_param.append(self._compute_solution_d_mu_single_direction(
                    parameter, l, solution, mu))
            sensitivities[parameter] = sens_for_param
        return sensitivities

    def _compute_output_d_mu(self, solution, mu=None, return_array=False, **kwargs):
        """Compute the gradient w.r.t. the parameter of the output functional

        Parameters
        ----------
        solution
            Internal model state for the given |Parameter value|.
        mu
            |Parameter value| for which to compute the gradient
        return_array
            if `True`, return the output gradient as a |NumPy array|.
            Otherwise, return a dict of gradients for each |Parameter|.

        Returns
        -------
        The gradient as a |NumPy array| or a dict of |NumPy arrays|.
        """
        assert self.output_functional is not None
        U_d_mus = self._compute_solution_d_mu(solution, mu)
        gradients = [] if return_array else {}
        for (parameter, size) in self.parameters.items():
            result = []
            for index in range(size):
                output_partial_dmu = self.output_functional.d_mu(parameter, index).apply(
                    solution, mu=mu).to_numpy()[0]
                U_d_mu = U_d_mus[parameter][index]
                result.append(output_partial_dmu + self.output_functional.jacobian(
                    solution, mu).apply(U_d_mu, mu).to_numpy()[0])
            result = np.array(result)
            if return_array:
                gradients.extend(result)
            else:
                gradients[parameter] = result
        if return_array:
            return np.array(gradients)
        else:
            return gradients

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

    def _compute_output_error_estimate(self, solution, mu=None, return_vector=False, **kwargs):
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
        return_vector
            If `True`, return the output estimate as a |NumPy array|,
            where each component corresponds to the respective component
            of the :attr:`output_functional`.
            Otherwise, return the euclidian norm of all components.
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
        return self.error_estimator.estimate_output_error(solution, mu, self,
                                                          return_vector=return_vector,
                                                          **kwargs)

    _compute_allowed_kwargs = frozenset()

    def compute(self, solution=False, output=False, solution_d_mu=False, output_d_mu=False,
                solution_error_estimate=False, output_error_estimate=False,
                output_d_mu_return_array=False, output_error_estimate_return_vector=False,
                *, mu=None, input=None, **kwargs):
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
        solution_d_mu
            If not `False`, either `True` to return the derivative of the model's
            internal state w.r.t. all parameter components or a tuple `(parameter, index)`
            to return the derivative of a single parameter component.
        output_d_mu
            If `True`, return the gradient of the model output w.r.t. the |Parameter|.
        solution_error_estimate
            If `True`, return an error estimate for the computed internal state.
        output_error_estimate
            If `True`, return an error estimate for the computed output.
        output_d_mu_return_array
            If `True`, return the output gradient as a |NumPy array|.
            Otherwise, return a dict of gradients for each |Parameter|.
        output_error_estimate_return_vector
            If `True`, return the output estimate as a |NumPy array|,
            where each component corresponds to the respective component
            of the :attr:`output_functional`.
            Otherwise, return the euclidian norm of all components.
        mu
            |Parameter values| for which to compute the values.
        input
            The model input. Either a |NumPy array| of shape `(self.dim_input,)`,
            a |Function| with `dim_domain == 1` and `shape_range == (self.dim_input,)`
            mapping time to input, or a `str` expression whith `t` as variable that
            can be used to instatiate an |ExpressionFunction| of this type.
            Can be `None` if `self.dim_input == 0`.
        kwargs
            Further keyword arguments to select further quantities that should
            be returned or to customize how the values are computed.

        Returns
        -------
        A dict with the computed values.
        """
        # make sure no unknown kwargs are passed
        assert kwargs.keys() <= self._compute_allowed_kwargs
        assert input is not None or self.dim_input == 0

        # parse parameter values
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)

        # parse input and add it to the parameter values
        mu_input = Parameters(input=self.dim_input).parse(input)
        input = mu_input.get_time_dependent_value('input') if mu_input.is_time_dependent('input') else mu_input['input']
        mu = mu.with_(input=input)

        # log output
        # explicitly checking if logging is disabled saves some cpu cycles
        if not self.logging_disabled:
            self.logger.info(f'Solving {self.name} for {mu} ...')

        # first call _compute to give subclasses more control
        data = self._compute(solution=solution, output=output,
                             solution_d_mu=solution_d_mu, output_d_mu=output_d_mu,
                             solution_error_estimate=solution_error_estimate,
                             output_error_estimate=output_error_estimate,
                             mu=mu, **kwargs)

        if (solution or output or solution_error_estimate
            or output_error_estimate or solution_d_mu or output_d_mu) \
           and 'solution' not in data:
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

        if solution_d_mu and 'solution_d_mu' not in data:
            if isinstance(solution_d_mu, tuple):
                retval = self._compute_solution_d_mu_single_direction(
                    solution_d_mu[0], solution_d_mu[1], data['solution'], mu=mu, **kwargs)
            else:
                retval = self._compute_solution_d_mu(data['solution'], mu=mu, **kwargs)
            # retval is always a dict
            if isinstance(retval, dict) and 'solution_d_mu' in retval:
                data.update(retval)
            else:
                data['solution_d_mu'] = retval

        if output_d_mu and 'output_d_mu' not in data:
            # TODO use caching here (requires skipping args in key generation)
            retval = self._compute_output_d_mu(data['solution'], mu=mu,
                                               return_array=output_d_mu_return_array,
                                               **kwargs)
            # retval is always a dict
            if isinstance(retval, dict) and 'output_d_mu' in retval:
                data.update(retval)
            else:
                data['output_d_mu'] = retval

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
            retval = self._compute_output_error_estimate(
                data['solution'], mu=mu,
                return_vector=output_error_estimate_return_vector, **kwargs)
            if isinstance(retval, dict):
                assert 'output_error_estimate' in retval
                data.update(retval)
            else:
                data['output_error_estimate'] = retval

        return data

    def solve(self, mu=None, input=None, return_error_estimate=False, **kwargs):
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
        input
            The model input. Either a |NumPy array| of shape `(self.dim_input,)`,
            a |Function| with `dim_domain == 1` and `shape_range == (self.dim_input,)`
            mapping time to input, or a `str` expression whith `t` as variable that
            can be used to instatiate an |ExpressionFunction| of this type.
            Can be `None` if `self.dim_input == 0`.
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
            input=input,
            **kwargs
        )
        if return_error_estimate:
            return data['solution'], data['solution_error_estimate']
        else:
            return data['solution']

    def output(self, mu=None, input=None, return_error_estimate=False,
               return_error_estimate_vector=False, **kwargs):
        """Return the model output for given |parameter values| `mu`.

        This method is a convenience wrapper around :meth:`compute`.

        Parameters
        ----------
        mu
            |Parameter values| for which to compute the output.
        input
            The model input. Either a |NumPy array| of shape `(self.dim_input,)`,
            a |Function| with `dim_domain == 1` and `shape_range == (self.dim_input,)`
            mapping time to input, or a `str` expression whith `t` as variable that
            can be used to instatiate an |ExpressionFunction| of this type.
            Can be `None` if `self.dim_input == 0`.
        return_error_estimate
            If `True`, also return an error estimate for the computed output.
        return_error_estimate_vector
            If `True`, return the output estimate as a |NumPy array|,
            where each component corresponds to the respective component
            of the :attr:`output_functional`.
            Otherwise, return the euclidian norm of all components.
        kwargs
            Additional keyword arguments passed to :meth:`compute` that
            might affect how the solution is computed.

        Returns
        -------
        The computed model output as a 2D |NumPy array|. The dimension
        of axis 1 is :attr:`dim_output`. (For stationary problems, axis 0 has
        dimension 1. For time-dependent problems, the dimension of axis 0
        depends on the number of time steps.)
        When `return_error_estimate` is `True`, the estimate is returned as
        second value.
        """
        data = self.compute(
            output=True,
            output_error_estimate=return_error_estimate,
            output_error_estimate_return_vector=return_error_estimate_vector,
            mu=mu,
            input=input,
            **kwargs
        )
        if return_error_estimate:
            return data['output'], data['output_error_estimate']
        else:
            return data['output']

    def solve_d_mu(self, parameter, index, mu=None, input=None, **kwargs):
        """Solve for the partial derivative of the solution w.r.t. a parameter index

        Parameters
        ----------
        parameter
            parameter for which to compute the sensitivity
        index
            parameter index for which to compute the sensitivity
        mu
            |Parameter value| for which to solve
        input
            The model input. Either a |NumPy array| of shape `(self.dim_input,)`,
            a |Function| with `dim_domain == 1` and `shape_range == (self.dim_input,)`
            mapping time to input, or a `str` expression whith `t` as variable that
            can be used to instatiate an |ExpressionFunction| of this type.
            Can be `None` if `self.dim_input == 0`.

        Returns
        -------
        The sensitivity of the solution as a |VectorArray|.
        """
        data = self.compute(
            solution_d_mu=(parameter, index),
            mu=mu,
            input=input,
            **kwargs
        )
        return data['solution_d_mu']

    def output_d_mu(self, mu=None, input=None, return_array=False, **kwargs):
        """Compute the gradient w.r.t. the parameter of the output functional.

        Parameters
        ----------
        mu
            |Parameter value| for which to compute the gradient
        input
            The model input. Either a |NumPy array| of shape `(self.dim_input,)`,
            a |Function| with `dim_domain == 1` and `shape_range == (self.dim_input,)`
            mapping time to input, or a `str` expression whith `t` as variable that
            can be used to instatiate an |ExpressionFunction| of this type.
            Can be `None` if `self.dim_input == 0`.
        return_array
            if `True`, return the output gradient as a |NumPy array|.
            Otherwise, return a dict of gradients for each |Parameter|.

        Returns
        -------
        The gradient as a |NumPy array| or a dict of |NumPy arrays|.
        """
        data = self.compute(
            output_d_mu=True,
            mu=mu,
            input=input,
            output_d_mu_return_array=return_array,
            **kwargs
        )
        return data['output_d_mu']

    def estimate_error(self, mu=None, input=None, **kwargs):
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
        input
            The model input. Either a |NumPy array| of shape `(self.dim_input,)`,
            a |Function| with `dim_domain == 1` and `shape_range == (self.dim_input,)`
            mapping time to input, or a `str` expression whith `t` as variable that
            can be used to instatiate an |ExpressionFunction| of this type.
            Can be `None` if `self.dim_input == 0`.
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
            input=input,
            **kwargs
        )['solution_error_estimate']

    def estimate_output_error(self, mu=None, input=None, return_vector=False, **kwargs):
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
        input
            The model input. Either a |NumPy array| of shape `(self.dim_input,)`,
            a |Function| with `dim_domain == 1` and `shape_range == (self.dim_input,)`
            mapping time to input, or a `str` expression whith `t` as variable that
            can be used to instatiate an |ExpressionFunction| of this type.
            Can be `None` if `self.dim_input == 0`.
        return_vector
            If `True`, return the output estimate as a |NumPy array|,
            where each component corresponds to the respective component
            of the :attr:`output_functional`.
            Otherwise, return the euclidian norm of all components.
        kwargs
            Additional keyword arguments passed to :meth:`compute` that
            might affect how the error estimate (or the output) is computed.

        Returns
        -------
        The estimated error.
        """
        return self.compute(
            output_error_estimate=True,
            output_error_estimate_return_vector=return_vector,
            mu=mu,
            input=input,
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
            return self.visualizer.visualize(U, **kwargs)
        else:
            raise NotImplementedError('Model has no visualizer.')
