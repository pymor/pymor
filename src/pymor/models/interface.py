# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from functools import cached_property

import numpy as np

from pymor.core.cache import CacheableObject
from pymor.core.exceptions import CacheKeyGenerationError
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

    @cached_property
    def computable_quantities(self):
        """Set of quantities that can be compute via :meth:`compute`."""
        return (
            {'solution', 'output', 'output_d_mu', 'solution_error_estimate', 'output_error_estimate'}
            | {('solution_d_mu', param, idx) for param, dim in self.parameters.items() for idx in range(dim)}
        )

    def compute(self, data=None, *,
                solution=False, output=False, solution_d_mu=False, output_d_mu=False,
                solution_error_estimate=False, output_error_estimate=False,
                mu=None, input=None, **kwargs):
        """Compute the solution of the model and associated quantities.

        This method computes the output of the model, its internal state,
        and various associated quantities for given |parameter values| `mu`.

        Parameters
        ----------
        data
            If not `None`, a dict of already computed quantities for the given `mu` and
            `input`. If used, newly computed quantities are added to the given dict and
            the same dict is returned.
            Providing a `data` dict can be helpful when some quantities (e.g., `output`)
            depend on already known quantities (e.g, `solution`) and caching has not been
            activated.
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
        # parse parameter values
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu, allow_time_dependent=True)

        # parse input and add it to the parameter values
        if input is not None:
            assert 'input' not in mu
            assert 'input' not in mu.time_dependent_values
            mu_input = Parameters(input=self.dim_input).parse(input)
            input = mu_input.time_dependent_values.get('input') or mu_input['input']
            mu = mu.with_(input=input)
        assert self.dim_input == 0 or 'input' in mu or 'input' in mu.time_dependent_values


        # collect all quantities to be computed
        wanted_quantities = {quantity for quantity, wanted in kwargs.items() if wanted}
        if solution:
            wanted_quantities.add('solution')
        if output:
            wanted_quantities.add('output')
        if solution_d_mu:
            if solution_d_mu is True:
                solution_d_mu = tuple((param, idx) for param, dim in self.parameters.items() for idx in range(dim))
            assert all(0 <= idx < self.parameters[param] for param, idx in solution_d_mu)
            wanted_quantities.update(('solution_d_mu', param, idx) for param, idx in solution_d_mu)
        if output_d_mu:
            wanted_quantities.add('output_d_mu')
        if solution_error_estimate:
            wanted_quantities.add('solution_error_estimate')
        if output_error_estimate:
            wanted_quantities.add('output_error_estimate')

        # make sure no unknown kwargs are passed
        assert wanted_quantities <= self.computable_quantities

        data = data if data is not None else {}
        self._compute_or_retrieve_from_cache(wanted_quantities, data, mu)

        if solution_d_mu:
            data['solution_d_mu'] = {quantity: data[('solution_d_mu',) + quantity] for quantity in solution_d_mu}

        return data

    def _compute_required_quantities(self, quantities, data, mu):
        """Compute additional required properties.

        Parameters
        ----------
        quantities
            Set of additional quantities to compute.
        data
            Dict into which the computed values are inserted.
        mu
            |Parameter values| for which to compute the quantities.

        Returns
        -------
        `None`.
        """
        quantities = {q for q in quantities if q not in data}
        if not quantities:
            return
        self.logger.info('Computing required quantities ...')
        self._compute_or_retrieve_from_cache(quantities, data, mu)

    def _compute_or_retrieve_from_cache(self, quantities, data, mu):
        assert quantities <= self.computable_quantities

        # fetch already computed data from cache and determine which quantities
        # actually need to be computed
        data = data if data is not None else {}
        quantities_to_compute = set()
        for quantity in quantities:
            if quantity in data:
                continue
            if self.cache_region is not None:
                try:
                    data[quantity] = self.get_cached_value((quantity, mu))
                except CacheKeyGenerationError:
                    assert isinstance(quantity, str)
                    self.logger.warning(f'Cannot generate cache key for {mu}. Result will not be cached.')
                    quantities_to_compute.add(quantity)
                except KeyError:
                    quantities_to_compute.add(quantity)
            else:
                quantities_to_compute.add(quantity)

        if quantities_to_compute:
            # log output
            # explicitly checking if logging is disabled saves some cpu cycles
            if not self.logging_disabled:
                quant = ['_'.join(str(qq) for qq in q) if isinstance(q, tuple) else q for q in quantities_to_compute]
                if len(quant) > 10:
                    quant = quant[:10] + ['...']
                with self.logger.block(f'Computing {", ".join(str(q) for q in quant)} of {self.name} for {mu} ...'):
                    # call _compute to actually compute the missing quantities
                    self._compute(quantities_to_compute.copy(), data, mu=mu)
            else:
                # call _compute to actually compute the missing quantities
                self._compute(quantities_to_compute.copy(), data, mu=mu)

        if self.cache_region is not None:
            for quantity in quantities_to_compute:
                try:
                    self.set_cached_value((quantity, mu), data[quantity])
                except CacheKeyGenerationError:
                    pass

        assert all(quantity in data for quantity in quantities_to_compute)


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
        """Compute the output sensitivities w.r.t. the model's parameters.

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

    def _compute(self, quantities, data, mu):
        """Actually compute model quantities.

        Override this method to provide implementations for solving a model,
        computing its output or other :attr:`computable_quantities`.

        `_compute` is passed a :class:`set` of quantities to compute.
        If `_compute` knows how to compute a given quantities, the computed value
        has to be added to the provided `data` dict. After that, the quantity
        should be removed from `quantities` so that a ::

            super()._compute(quantities, data, mu)

        end of the implementation will not cause `NotImplementedErrors`.
        :class:`Model` provides default implementations for the `output`, `output_d_mu`,
        `solution_error_estimate` and `output_error_estimate` quantities.
        The implementations for `output` and `output_d_mu` require the model to possess
        an `output_functional` attribute, which is applied to the solution in order to
        obtain the output. `solution_error_estimate` and `output_error_estimate` defer
        the error estimation to the model's :attr:`error_estimator`.

        In case a requested quantity depends on another quantities, implementations should
        call ::

            self._compute_required_quantities({'quantity_a', 'quantity_b'}, data, mu)

        which will populate `data` with the needed quantities by calling `_compute` again
        or retrieving previously computed values from the cache. Do not compute required
        properties directly in `_compute` as this will break caching.

        Parameters
        ----------
        quantities
            Set of additional quantities to compute.
        data
            Dict into which the computed values are inserted.
        mu
            |Parameter values| for which to compute the quantities.

        Returns
        -------
        `None`.
        """
        if 'solution' in quantities:
            raise NotImplementedError

        if 'output' in quantities:
            # default implementation in case Model has an 'output_functional'
            if not hasattr(self, 'output_functional'):
                raise NotImplementedError
            self._compute_required_quantities({'solution'}, data, mu)

            data['output'] = self.output_functional.apply(data['solution'], mu=mu).to_numpy_TP().T
            quantities.remove('output')

        if 'output_d_mu' in quantities:
            # default implementation in case Model has an 'output_functional'
            if not hasattr(self, 'output_functional'):
                raise NotImplementedError
            self._compute_required_quantities(
                {'solution'} | {('solution_d_mu', param, idx)
                                for param, dim in self.parameters.items() for idx in range(dim)},
                data, mu
            )

            solution = data['solution']
            sensitivities = {}
            for (parameter, size) in self.parameters.items():
                for index in range(size):
                    output_d_mu = self.output_functional.d_mu(parameter, index).apply(
                        solution, mu=mu).to_numpy_TP().T
                    U_d_mu = data['solution_d_mu', parameter, index]
                    for t, U in enumerate(U_d_mu):
                        output_d_mu[t] \
                            += self.output_functional.jacobian(solution[t], mu).apply(U, mu).to_numpy_TP()[:, 0]
                    sensitivities[parameter, index] = output_d_mu
            data['output_d_mu'] = OutputDMuResult(sensitivities)
            quantities.remove('output_d_mu')

        if 'solution_error_estimate' in quantities:
            if self.error_estimator is None:
                raise ValueError('Model has no error estimator')
            self._compute_required_quantities({'solution'}, data, mu)

            data['solution_error_estimate'] = self.error_estimator.estimate_error(data['solution'], mu, self)
            quantities.remove('solution_error_estimate')

        if 'output_error_estimate' in quantities:
            if self.error_estimator is None:
                raise ValueError('Model has no error estimator')
            self._compute_required_quantities({'solution'}, data, mu)

            data['output_error_estimate'] = self.error_estimator.estimate_output_error(data['solution'], mu, self)
            quantities.remove('output_error_estimate')


class OutputDMuResult(FrozenDict):
    """Immutable dict of gradients returned by :meth:`~Model.output_d_mu`."""

    def to_numpy(self):
        """Return gradients as a single 3D NumPy array.

        The array is obtained by stacking the individual arrays along an
        additional axis 0, ordered by alphabetically ordered parameter name.
        """
        return np.array([v for k, v in sorted(self.items())])

    def __repr__(self):
        return f'OutputDMuResult({dict(sorted(self.items()))})'
