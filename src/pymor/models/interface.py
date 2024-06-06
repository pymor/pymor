# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.cache import CacheableObject
from pymor.operators.constructions import induced_norm
from pymor.parameters.base import Mu, Parameters, ParametricObject
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
            Additional keyword arguments to select further quantities that should
            be computed.

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
        as returned by :meth:`!_compute_solution`. When this is not the case,
        the computation of the output should be implemented in :meth:`!_compute`.

        .. note::

            The default implementation applies the |Operator| given by the
            :attr:`!output_functional` attribute to the given `solution`
            |VectorArray|.

        Parameters
        ----------
        solution
            Internal model state for the given |parameter values|.
        mu
            |Parameter values| for which to compute the output.
        kwargs
            Additional keyword arguments to select further quantities that should
            be computed.

        Returns
        -------
        |NumPy array| with the computed output or a dict which at least
        must contain the key `'output'`.
        """
        return self.output_functional.apply(solution, mu=mu).to_numpy()

    def _compute_solution_d_mu_single_direction(self, parameter, index, solution, mu=None):
        """Compute the solution sensitivity w.r.t. a single parameter.

        Parameters
        ----------
        parameter
            Parameter for which to compute the sensitivity.
        index
            Parameter index for which to compute the sensitivity.
        solution
            Solution of the Model for `mu`.
        mu
            |Parameter value| at which to compute the sensitivity.

        Returns
        -------
        The sensitivity of the solution as a |VectorArray|.
        """
        raise NotImplementedError

    def _compute_solution_d_mu(self, solution, directions, mu=None):
        """Compute solution sensitivities w.r.t. to given parameters.

        Parameters
        ----------
        solution
            Solution of the Model for `mu`.
        directions
            Either `True`, to compute solution sensitivities w.r.t. all parameters
            or a sequence of tuples `(parameter, index)` to compute the solution
            sensitivities for selected parameters.
        mu
            |Parameter value| at which to compute the sensitivities.

        Returns
        -------
        A dict with keys `(parameter, index)` of all computed solution sensitivities.
        """
        sensitivities = {}
        if directions is True:
            directions = ((param, idx) for param, dim in self.parameters.items() for idx in range(dim))
        for (param, idx) in directions:
            sens_for_param = self._compute_solution_d_mu_single_direction(param, idx, solution, mu)
            sensitivities[(param, idx)] = sens_for_param
        return sensitivities

    def _compute_output_d_mu(self, solution, mu=None):
        """Compute the output sensitivites w.r.t. the model's parameters.

        Parameters
        ----------
        solution
            Solution of the Model for `mu`.
        mu
            |Parameter value| at which to compute the sensitivities.

        Returns
        -------
        The output sensitivities as a dict `{(parameter, index): sensitivity}` where
        `sensitivity` is a 2D |NumPy arrays| with axis 0 corresponding to time and axis 1
        corresponding to the output component.
        The returned :class:`OutputDMuResult` object has a `meth`:~OutputDMuResult.to_numpy`
        method to convert it into a single NumPy array, e.g., for use in optimization
        libraries.
        """
        assert self.output_functional is not None
        U_d_mus = self._compute_solution_d_mu(solution, True, mu)
        sensitivities = {}
        for (parameter, size) in self.parameters.items():
            for index in range(size):
                output_d_mu = self.output_functional.d_mu(parameter, index).apply(
                    solution, mu=mu).to_numpy()
                U_d_mu = U_d_mus[parameter, index]
                for t, U in enumerate(U_d_mu):
                    output_d_mu[t] += self.output_functional.jacobian(solution[t], mu).apply(U, mu).to_numpy()[0]
                sensitivities[parameter, index] = output_d_mu
        return OutputDMuResult(sensitivities)

    def _compute_solution_error_estimate(self, solution, mu=None, **kwargs):
        """Compute an error estimate for the computed internal state.

        This method is called by the default implementation of :meth:`compute`
        in :class:`pymor.models.interface.Model`. The assumption is made
        that the error estimate is a derived quantity from the model's internal state
        as returned by :meth:`!_compute_solution`. When this is not the case,
        the computation of the error estimate should be implemented in :meth:`!_compute`.

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
            Additional keyword arguments to select further quantities that should
            be computed.

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
        as returned by :meth:`!_compute_solution`. When this is not the case,
        the computation of the error estimate should be implemented in :meth:`!_compute`.

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
            Additional keyword arguments to select further quantities that should
            be computed.

        Returns
        -------
        The computed error estimate or a dict which at least must contain the key
        `'solution_error_estimate'`.
        """
        if self.error_estimator is None:
            raise ValueError('Model has no error estimator')
        return self.error_estimator.estimate_output_error(solution, mu, self, **kwargs)

    _compute_allowed_kwargs = frozenset()

    def compute(self, solution=False, output=False, solution_d_mu=False, output_d_mu=False,
                solution_error_estimate=False, output_error_estimate=False,
                *, mu=None, input=None, **kwargs):
        """Compute the solution of the model and associated quantities.

        This method computes the output of the model, its internal state,
        and various associated quantities for given |parameter values| `mu`.

        .. note::

            The default implementation defers the actual computations to
            the methods :meth:`!_compute_solution`, :meth:`!_compute_output`,
            :meth:`!_compute_solution_error_estimate` and :meth:`!_compute_output_error_estimate`.
            The call to :meth:`!_compute_solution` is :mod:`cached <pymor.core.cache>`.
            In addition, |Model| implementors may implement :meth:`!_compute` to
            simultaneously compute multiple values in an optimized way. The corresponding
            `_compute_XXX` methods will not be called for values already returned by
            :meth:`!_compute`.

        Parameters
        ----------
        solution
            If `True`, return the model's internal state.
        output
            If `True`, return the model output.
        solution_d_mu
            If not `False`, either `True` to return the sensitivities of the model's
            solution w.r.t. all parameters, or a sequence of tuples `(parameter, index)`
            to compute the solution sensitivities for selected parameters.
        output_d_mu
            If `True`, return the output sensitivities w.r.t. the model's parameters.
        solution_error_estimate
            If `True`, return an error estimate for the computed internal state.
        output_error_estimate
            If `True`, return an error estimate for the computed output.
        mu
            |Parameter values| for which to compute the values.
        input
            The model input. Either a |NumPy array| of shape `(self.dim_input,)`,
            a |Function| with `dim_domain == 1` and `shape_range == (self.dim_input,)`
            mapping time to input, or a `str` expression with `t` as variable that
            can be used to instantiate an |ExpressionFunction| of this type.
            Can be `None` if `self.dim_input == 0`.
        kwargs
            Additional keyword arguments to select further quantities that should
            be computed.

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

        if (solution or solution_error_estimate or solution_d_mu) and 'solution' not in data \
           or (output or output_error_estimate or output_d_mu) and 'output' not in data:
            retval = self.cached_method_call(self._compute_solution, mu=mu, **kwargs)
            if isinstance(retval, dict):
                assert 'solution' in retval
                data.update(retval)
            else:
                data['solution'] = retval

        if output and 'output' not in data:
            # TODO: use caching here (requires skipping args in key generation)
            retval = self._compute_output(data['solution'], mu=mu, **kwargs)
            if isinstance(retval, dict):
                assert 'output' in retval
                data.update(retval)
            else:
                data['output'] = retval

        if solution_d_mu and 'solution_d_mu' not in data:
            retval = self._compute_solution_d_mu(data['solution'], solution_d_mu, mu=mu)
            data['solution_d_mu'] = retval

        if output_d_mu and 'output_d_mu' not in data:
            # TODO: use caching here (requires skipping args in key generation)
            retval = self._compute_output_d_mu(data['solution'], mu=mu)
            data['output_d_mu'] = retval

        if solution_error_estimate and 'solution_error_estimate' not in data:
            # TODO: use caching here (requires skipping args in key generation)
            retval = self._compute_solution_error_estimate(data['solution'], mu=mu, **kwargs)
            if isinstance(retval, dict):
                assert 'solution_error_estimate' in retval
                data.update(retval)
            else:
                data['solution_error_estimate'] = retval

        if output_error_estimate and 'output_error_estimate' not in data:
            # TODO: use caching here (requires skipping args in key generation)
            retval = self._compute_output_error_estimate(
                data['solution'], mu=mu, **kwargs)
            if isinstance(retval, dict):
                assert 'output_error_estimate' in retval
                data.update(retval)
            else:
                data['output_error_estimate'] = retval

        return data

    def solve(self, mu=None, input=None, return_error_estimate=False):
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
            mapping time to input, or a `str` expression with `t` as variable that
            can be used to instantiate an |ExpressionFunction| of this type.
            Can be `None` if `self.dim_input == 0`.
        return_error_estimate
            If `True`, also return an error estimate for the computed solution.

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
        )
        if return_error_estimate:
            return data['solution'], data['solution_error_estimate']
        else:
            return data['solution']

    def output(self, mu=None, input=None, return_error_estimate=False):
        """Return the model output for given |parameter values| `mu`.

        This method is a convenience wrapper around :meth:`compute`.

        Parameters
        ----------
        mu
            |Parameter values| for which to compute the output.
        input
            The model input. Either a |NumPy array| of shape `(self.dim_input,)`,
            a |Function| with `dim_domain == 1` and `shape_range == (self.dim_input,)`
            mapping time to input, or a `str` expression with `t` as variable that
            can be used to instantiate an |ExpressionFunction| of this type.
            Can be `None` if `self.dim_input == 0`.
        return_error_estimate
            If `True`, also return an error estimate for the computed output.

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
            mu=mu,
            input=input,
        )
        if return_error_estimate:
            return data['output'], data['output_error_estimate']
        else:
            return data['output']

    def solve_d_mu(self, parameter, index, mu=None, input=None):
        """Compute the solution sensitivity w.r.t. a single parameter.

        Parameters
        ----------
        parameter
            Parameter for which to compute the sensitivity.
        index
            Parameter index for which to compute the sensitivity.
        mu
            |Parameter value| at which to compute the sensitivity.
        input
            The model input. Either a |NumPy array| of shape `(self.dim_input,)`,
            a |Function| with `dim_domain == 1` and `shape_range == (self.dim_input,)`
            mapping time to input, or a `str` expression with `t` as variable that
            can be used to instantiate an |ExpressionFunction| of this type.
            Can be `None` if `self.dim_input == 0`.

        Returns
        -------
        The sensitivity of the solution as a |VectorArray|.
        """
        data = self.compute(
            solution_d_mu=[(parameter, index)],
            mu=mu,
            input=input,
        )
        return data['solution_d_mu'][parameter, index]

    def output_d_mu(self, mu=None, input=None):
        """Compute the output sensitivites w.r.t. the model's parameters.

        Parameters
        ----------
        mu
            |Parameter value| at which to compute the output sensitivities.
        input
            The model input. Either a |NumPy array| of shape `(self.dim_input,)`,
            a |Function| with `dim_domain == 1` and `shape_range == (self.dim_input,)`
            mapping time to input, or a `str` expression with `t` as variable that
            can be used to instantiate an |ExpressionFunction| of this type.
            Can be `None` if `self.dim_input == 0`.

        Returns
        -------
        The output sensitivities as a dict `{(parameter, index): sensitivity}` where
        `sensitivity` is a 2D |NumPy arrays| with axis 0 corresponding to time and axis 1
        corresponding to the output component.
        The returned :class:`OutputDMuResult` object has a `meth`:~OutputDMuResult.to_numpy`
        method to convert it into a single NumPy array, e.g., for use in optimization
        libraries.
        """
        data = self.compute(
            output_d_mu=True,
            mu=mu,
            input=input,
        )
        return data['output_d_mu']

    def estimate_error(self, mu=None, input=None):
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
            mapping time to input, or a `str` expression with `t` as variable that
            can be used to instantiate an |ExpressionFunction| of this type.
            Can be `None` if `self.dim_input == 0`.

        Returns
        -------
        The estimated :attr:`solution_space` error as a 1D |NumPy array|.
        For stationary problems, the returned array has length 1.
        For time-dependent problems, the length depends on the number of time
        steps. The norm w.r.t. which the error is estimated depends on the given
        problem.
        """
        return self.compute(
            solution_error_estimate=True,
            mu=mu,
            input=input,
        )['solution_error_estimate']

    def estimate_output_error(self, mu=None, input=None):
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
            mapping time to input, or a `str` expression with `t` as variable that
            can be used to instantiate an |ExpressionFunction| of this type.
            Can be `None` if `self.dim_input == 0`.

        Returns
        -------
        The estimated model output as a 2D |NumPy array|. The dimension
        of axis 1 is :attr:`dim_output`. For stationary problems, axis 0 has
        dimension 1. For time-dependent problems, the dimension of axis 0
        depends on the number of time steps. The spatial/temporal norms
        w.r.t. which the error is estimated depend on the given problem.
        """
        return self.compute(
            output_error_estimate=True,
            mu=mu,
            input=input
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
        if self.visualizer is not None:
            return self.visualizer.visualize(U, **kwargs)
        else:
            raise NotImplementedError('Model has no visualizer.')


class OutputDMuResult(FrozenDict):
    """Immutable dict of gradients returned by :meth:`~Model.output_d_mu`."""

    def to_numpy(self):
        """Return gradients as a single 2D NumPy array.

        The array is obtained by stacking the individual arrays along axis 0,
        ordered by alphabetically ordered parameter name.
        """
        return np.vstack([v for k, v in sorted(self.items())])

    def __repr__(self):
        return f'OutputDMuResult({dict(sorted(self.items()))})'
