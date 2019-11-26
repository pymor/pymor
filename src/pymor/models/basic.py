# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.timestepping import TimeStepperInterface
from pymor.models.interfaces import ModelInterface
from pymor.operators.constructions import VectorOperator, induced_norm
from pymor.tools.formatrepr import indent_value
from pymor.tools.frozendict import FrozenDict
from pymor.vectorarrays.interfaces import VectorArrayInterface


class ModelBase(ModelInterface):
    """Base class for |Models| providing some common functionality."""

    sid_ignore = ModelInterface.sid_ignore | {'visualizer'}

    def __init__(self, products=None, estimator=None, visualizer=None,
                 cache_region=None, name=None, **kwargs):

        products = FrozenDict(products or {})

        if products:
            for k, v in products.items():
                setattr(self, f'{k}_product', v)
                setattr(self, f'{k}_norm', induced_norm(v))

        self.__auto_init(locals())
        self.enable_caching(cache_region)

    def visualize(self, U, **kwargs):
        """Visualize a solution |VectorArray| U.

        Parameters
        ----------
        U
            The |VectorArray| from
            :attr:`~pymor.models.interfaces.ModelInterface.solution_space`
            that shall be visualized.
        kwargs
            See docstring of `self.visualizer.visualize`.
        """
        if self.visualizer is not None:
            return self.visualizer.visualize(U, self, **kwargs)
        else:
            raise NotImplementedError('Model has no visualizer.')

    def estimate(self, U, mu=None):
        if self.estimator is not None:
            return self.estimator.estimate(U, mu=mu, m=self)
        else:
            raise NotImplementedError('Model has no estimator.')


class StationaryModel(ModelBase):
    """Generic class for models of stationary problems.

    This class describes discrete problems given by the equation::

        L(u(μ), μ) = F(μ)

    with a vector-like right-hand side F and a (possibly non-linear) operator L.

    Note that even when solving a variational formulation where F is a
    functional and not a vector, F has to be specified as a vector-like
    |Operator| (mapping scalars to vectors). This ensures that in the complex
    case both L and F are anti-linear in the test variable.

    Parameters
    ----------
    operator
        The |Operator| L.
    rhs
        The vector F. Either a |VectorArray| of length 1 or a vector-like
        |Operator|.
    output_functional
        |Operator| mapping a given solution to the model output. In many applications,
        this will be a |Functional|, i.e. an |Operator| mapping to scalars.
        This is not required, however.
    products
        A dict of inner product |Operators| defined on the discrete space the
        problem is posed on. For each product with key `'x'` a corresponding
        attribute `x_product`, as well as a norm method `x_norm` is added to
        the model.
    parameter_space
        The |ParameterSpace| for which the discrete problem is posed.
    estimator
        An error estimator for the problem. This can be any object with
        an `estimate(U, mu, m)` method. If `estimator` is
        not `None`, an `estimate(U, mu)` method is added to the
        model which will call `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, m, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the model which forwards its arguments to the
        visualizer's `visualize` method.
    cache_region
        `None` or name of the |CacheRegion| to use.
    name
        Name of the model.
    """

    def __init__(self, operator, rhs, output_functional=None, products=None,
                 parameter_space=None, estimator=None, visualizer=None, cache_region=None, name=None):

        if isinstance(rhs, VectorArrayInterface):
            assert rhs in operator.range
            rhs = VectorOperator(rhs, name='rhs')

        assert rhs.range == operator.range and rhs.source.is_scalar and rhs.linear
        assert output_functional is None or output_functional.source == operator.source

        super().__init__(products=products,
                         estimator=estimator, visualizer=visualizer,
                         cache_region=cache_region, name=name)

        self.build_parameter_type(operator, rhs, output_functional)
        self.__auto_init(locals())
        self.solution_space = operator.source
        self.linear = operator.linear and (output_functional is None or output_functional.linear)
        if output_functional is not None:
            self.output_space = output_functional.range

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    {"linear" if self.linear else "non-linear"}\n'
            f'    parameter_space: {indent_value(str(self.parameter_space), len("    parameter_space: "))}\n'
            f'    solution_space:  {self.solution_space}\n'
            f'    output_space:    {self.output_space}\n'
        )

    def _solve(self, mu=None, return_output=False):
        mu = self.parse_parameter(mu)

        # explicitly checking if logging is disabled saves the str(mu) call
        if not self.logging_disabled:
            self.logger.info(f'Solving {self.name} for {mu} ...')

        U = self.operator.apply_inverse(self.rhs.as_range_array(mu), mu=mu)
        if return_output:
            if self.output_functional is None:
                raise ValueError('Model has no output')
            return U, self.output_functional.apply(U, mu=mu)
        else:
            return U


class InstationaryModel(ModelBase):
    """Generic class for models of instationary problems.

    This class describes instationary problems given by the equations::

        M * ∂_t u(t, μ) + L(u(μ), t, μ) = F(t, μ)
                                u(0, μ) = u_0(μ)

    for t in [0,T], where L is a (possibly non-linear) time-dependent
    |Operator|, F is a time-dependent vector-like |Operator|, and u_0 the
    initial data. The mass |Operator| M is assumed to be linear.

    Parameters
    ----------
    T
        The final time T.
    initial_data
        The initial data `u_0`. Either a |VectorArray| of length 1 or
        (for the |Parameter|-dependent case) a vector-like |Operator|
        (i.e. a linear |Operator| with `source.dim == 1`) which
        applied to `NumpyVectorArray(np.array([1]))` will yield the
        initial data for a given |Parameter|.
    operator
        The |Operator| L.
    rhs
        The right-hand side F.
    mass
        The mass |Operator| `M`. If `None`, the identity is assumed.
    time_stepper
        The :class:`time-stepper <pymor.algorithms.timestepping.TimeStepperInterface>`
        to be used by :meth:`~pymor.models.interfaces.ModelInterface.solve`.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each
        intermediate vector that is calculated is returned.
    output_functional
        |Operator| mapping a given solution to the model output. In many applications,
        this will be a |Functional|, i.e. an |Operator| mapping to scalars.
        This is not required, however.
    products
        A dict of product |Operators| defined on the discrete space the
        problem is posed on. For each product with key `'x'` a corresponding
        attribute `x_product`, as well as a norm method `x_norm` is added to
        the model.
    parameter_space
        The |ParameterSpace| for which the discrete problem is posed.
    estimator
        An error estimator for the problem. This can be any object with
        an `estimate(U, mu, m)` method. If `estimator` is
        not `None`, an `estimate(U, mu)` method is added to the
        model which will call `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, m, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the model which forwards its arguments to the
        visualizer's `visualize` method.
    cache_region
        `None` or name of the |CacheRegion| to use.
    name
        Name of the model.
    """

    def __init__(self, T, initial_data, operator, rhs, mass=None, time_stepper=None, num_values=None,
                 output_functional=None, products=None, parameter_space=None, estimator=None, visualizer=None,
                 cache_region=None, name=None):

        if isinstance(rhs, VectorArrayInterface):
            assert rhs in operator.range
            rhs = VectorOperator(rhs, name='rhs')
        if isinstance(initial_data, VectorArrayInterface):
            assert initial_data in operator.source
            initial_data = VectorOperator(initial_data, name='initial_data')

        assert isinstance(time_stepper, TimeStepperInterface)
        assert initial_data.source.is_scalar
        assert operator.source == initial_data.range
        assert rhs is None \
            or rhs.linear and rhs.range == operator.range and rhs.source.is_scalar
        assert mass is None \
            or mass.linear and mass.source == mass.range == operator.source
        assert output_functional is None or output_functional.source == operator.source

        super().__init__(products=products, estimator=estimator,
                         visualizer=visualizer, cache_region=cache_region, name=name)

        self.build_parameter_type(initial_data, operator, rhs, mass, output_functional, provides={'_t': 0})
        self.__auto_init(locals())
        self.solution_space = operator.source
        self.linear = operator.linear and (output_functional is None or output_functional.linear)

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    {"linear" if self.linear else "non-linear"}\n'
            f'    T: {self.T}\n'
            f'    parameter_space: {indent_value(str(self.parameter_space), len("    parameter_space: "))}\n'
            f'    solution_space:  {self.solution_space}\n'
            f'    output_space:    {self.output_space}\n'
        )

    def with_time_stepper(self, **kwargs):
        return self.with_(time_stepper=self.time_stepper.with_(**kwargs))

    def _solve(self, mu=None, return_output=False):
        mu = self.parse_parameter(mu).copy()

        # explicitly checking if logging is disabled saves the expensive str(mu) call
        if not self.logging_disabled:
            self.logger.info(f'Solving {self.name} for {mu} ...')

        mu['_t'] = 0
        U0 = self.initial_data.as_range_array(mu)
        U = self.time_stepper.solve(operator=self.operator, rhs=self.rhs, initial_data=U0, mass=self.mass,
                                    initial_time=0, end_time=self.T, mu=mu, num_values=self.num_values)
        if return_output:
            if self.output_functional is None:
                raise ValueError('Model has no output')
            return U, self.output_functional.apply(U, mu=mu)
        else:
            return U

    def to_lti(self):
        """Convert model to |LTIModel|.

        This method interprets the given model as an |LTIModel|
        in the following way::

            - self.operator        -> A
            self.rhs               -> B
            self.output_functional -> C
            None                   -> D
            self.mass              -> E
        """
        if self.output_functional is None:
            raise ValueError('No output defined.')
        A = - self.operator
        B = self.rhs
        C = self.output_functional
        E = self.mass

        if not all(op.linear for op in [A, B, C, E]):
            raise ValueError('Operators not linear.')

        from pymor.models.iosys import LTIModel
        return LTIModel(A, B, C, E=E, parameter_space=self.parameter_space, visualizer=self.visualizer)
