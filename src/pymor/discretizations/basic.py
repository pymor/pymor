# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.timestepping import TimeStepperInterface
from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.operators.constructions import VectorOperator, induced_norm
from pymor.operators.interfaces import OperatorInterface
from pymor.tools.frozendict import FrozenDict
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.numpy import NumpyVectorSpace


class DiscretizationBase(DiscretizationInterface):
    """Base class for |Discretizations| providing some common functionality."""

    sid_ignore = DiscretizationInterface.sid_ignore | {'visualizer'}

    def __init__(self, operators, estimator=None, visualizer=None,
                 cache_region=None, name=None):
        self.operators = FrozenDict(operators)
        self.linear = all(op is None or op.linear for op in operators.values())
        self.estimator = estimator
        self.visualizer = visualizer
        self.enable_caching(cache_region)
        self.name = name

        for k, v in operators.items():
            if k.endswith('_product'):
                setattr(self, k, v)
                setattr(self, '{}_norm'.format(k[:-len('_product')]), induced_norm(v))

    def visualize(self, U, **kwargs):
        """Visualize a solution |VectorArray| U.

        Parameters
        ----------
        U
            The |VectorArray| from
            :attr:`~pymor.discretizations.interfaces.DiscretizationInterface.solution_space`
            that shall be visualized.
        kwargs
            See docstring of `self.visualizer.visualize`.
        """
        if self.visualizer is not None:
            self.visualizer.visualize(U, self, **kwargs)
        else:
            raise NotImplementedError('Discretization has no visualizer.')

    def estimate(self, U, mu=None):
        if self.estimator is not None:
            return self.estimator.estimate(U, mu=mu, discretization=self)
        else:
            raise NotImplementedError('Discretization has no estimator.')


class StationaryDiscretization(DiscretizationBase):
    """Generic class for discretizations of stationary problems.

    This class describes discrete problems given by the equation::

        L(u(μ), μ) = F(μ)

    with a linear functional F and a (possibly non-linear) operator L.

    Parameters
    ----------
    operator
        The |Operator| L.
    rhs
        The |Functional| F.
    operators
        A dict of |Operators| associated with the discretization.
    parameter_space
        The |ParameterSpace| for which the discrete problem is posed.
    estimator
        An error estimator for the problem. This can be any object with
        an `estimate(U, mu, discretization)` method. If `estimator` is
        not `None`, an `estimate(U, mu)` method is added to the
        discretization which will call `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, discretization, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the discretization which forwards its arguments to the
        visualizer's `visualize` method.
    cache_region
        `None` or name of the |CacheRegion| to use.
    name
        Name of the discretization.

    Attributes
    ----------
    operator
        The |Operator| L. The same as `operators['operator']`.
    rhs
        The |Functional| F. The same as `functionals['rhs']`.
    """

    def __init__(self, operator=None, rhs=None, operators=None,
                 parameter_space=None, estimator=None, visualizer=None, cache_region=None, name=None):
        operators = operators or {}
        operator = operator or operators['operator']
        rhs = rhs or operators['rhs']
        assert isinstance(operator, OperatorInterface)
        assert isinstance(rhs, OperatorInterface) and rhs.linear
        assert operator.source == operator.range == rhs.source
        assert rhs.range.dim == 1
        assert 'operator' not in operators or operator == operators['operator']
        assert 'rhs' not in operators or rhs == operators['rhs']

        operators_with_operators = {'operator': operator, 'rhs': rhs}
        operators_with_operators.update(operators)
        super().__init__(operators=operators_with_operators,
                         estimator=estimator, visualizer=visualizer,
                         cache_region=cache_region, name=name)
        self.solution_space = operator.source
        self.operator = operator
        self.rhs = rhs
        self.build_parameter_type(operator, rhs)
        self.parameter_space = parameter_space

    def with_(self, **kwargs):
        assert set(kwargs.keys()) <= self.with_arguments

        # when 'operators' is not given but 'operator', make sure that
        # we use the old 'operators' dict but with updated 'operator'
        kwargs.setdefault('operators', dict(self.operators,
                                            operator=kwargs.get('operator', self.operator),
                                            rhs=kwargs.get('rhs', self.rhs)))

        # make sure we do not use self.operator (for the case that 'operators' is given)
        kwargs.setdefault('operator', None)
        kwargs.setdefault('rhs', None)

        return super().with_(**kwargs)

    def _solve(self, mu=None):
        mu = self.parse_parameter(mu)

        # explicitly checking if logging is disabled saves the str(mu) call
        if not self.logging_disabled:
            self.logger.info('Solving {} for {} ...'.format(self.name, mu))

        return self.operator.apply_inverse(self.rhs.as_vector(mu), mu=mu)


class InstationaryDiscretization(DiscretizationBase):
    """Generic class for discretizations of instationary problems.

    This class describes instationary problems given by the equations::

        M * ∂_t u(t, μ) + L(u(μ), t, μ) = F(t, μ)
                                u(0, μ) = u_0(μ)

    for t in [0,T], where L is a (possibly non-linear) time-dependent
    |Operator|, F is a time-dependent linear |Functional|, and u_0 the
    initial data. The mass |Operator| M is assumed to be linear,
    time-independent and |Parameter|-independent.

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
        The |Functional| F.
    mass
        The mass |Operator| `M`. If `None`, the identity is assumed.
    time_stepper
        The :class:`time-stepper <pymor.algorithms.timestepping.TimeStepperInterface>`
        to be used by :meth:`~pymor.discretizations.interfaces.DiscretizationInterface.solve`.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each
        intermediate vector that is calculated is returned.
    parameter_space
        The |ParameterSpace| for which the discrete problem is posed.
    estimator
        An error estimator for the problem. This can be any object with
        an `estimate(U, mu, discretization)` method. If `estimator` is
        not `None`, an `estimate(U, mu)` method is added to the
        discretization which will call `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, discretization, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the discretization which forwards its arguments to the
        visualizer's `visualize` method.
    cache_region
        `None` or name of the |CacheRegion| to use.
    name
        Name of the discretization.

    Attributes
    ----------
    T
        The final time T.
    initial_data
        The intial data u_0 given by a vector-like |Operator|. The same
        as `vector_operators['initial_data']`.
    operator
        The |Operator| L. The same as `operators['operator']`.
    rhs
        The |Functional| F. The same as `functionals['rhs']`.
    mass
        The mass operator M. The same as `operators['mass']`.
    time_stepper
        The provided :class:`time-stepper <pymor.algorithms.timestepping.TimeStepperInterface>`.
    """

    def __init__(self, T, initial_data=None, operator=None, rhs=None, mass=None, time_stepper=None, num_values=None,
                 operators=None, parameter_space=None,
                 estimator=None, visualizer=None, cache_region=None, name=None):
        operators = operators or {}
        initial_data = initial_data or operators['initial_data']
        operator = operator or operators['operator']
        rhs = rhs or operators['rhs']
        mass = mass or operators['mass']
        if isinstance(initial_data, VectorArrayInterface):
            initial_data = VectorOperator(initial_data, name='initial_data')

        assert isinstance(initial_data, OperatorInterface)
        assert initial_data.source == NumpyVectorSpace(1)
        assert 'initial_data' not in operators or initial_data == operators['initial_data']

        assert isinstance(operator, OperatorInterface)
        assert operator.source == operator.range == initial_data.range
        assert 'operator' not in operators or operator == operators['operator']

        assert rhs is None or isinstance(rhs, OperatorInterface) and rhs.linear
        assert rhs is None or rhs.source == operator.source and rhs.range.dim == 1
        assert 'rhs' not in operators or rhs == operators['rhs']

        assert mass is None or isinstance(mass, OperatorInterface) and mass.linear
        assert mass is None or mass.source == mass.range == operator.source
        assert 'mass' not in operators or mass == operators['mass']

        assert isinstance(time_stepper, TimeStepperInterface)

        operators_with_operators = {'initial_data': initial_data, 'operator': operator, 'mass': mass, 'rhs': rhs}
        operators_with_operators.update(operators)

        super().__init__(operators=operators_with_operators,
                         estimator=estimator,
                         visualizer=visualizer, cache_region=cache_region, name=name)
        self.T = T
        self.solution_space = operator.source
        self.initial_data = initial_data
        self.operator = operator
        self.rhs = rhs
        self.mass = mass
        self.time_stepper = time_stepper
        self.num_values = num_values
        self.build_parameter_type(initial_data, operator, rhs, mass, provides={'_t': 0})
        self.parameter_space = parameter_space
        if hasattr(time_stepper, 'nt'):
            self.add_with_arguments = self.add_with_arguments | {'time_stepper_nt'}

    def with_(self, **kwargs):
        assert set(kwargs.keys()) <= self.with_arguments
        assert 'time_stepper_nt' not in kwargs or 'time_stepper' not in kwargs

        # when 'operators' is not given but 'operator' or 'mass', make sure that
        # we use the old 'operators' dict but with updated 'operators' and 'mass'
        kwargs.setdefault('operators', dict(self.operators,
                                            operator=kwargs.get('operator', self.operator),
                                            mass=kwargs.get('mass', self.mass),
                                            rhs=kwargs.get('rhs', self.rhs),
                                            initial_data=kwargs.get('initial_data', self.initial_data)))

        # make sure we do not use self.operator (for the case that 'operators' is given)
        kwargs.setdefault('operator', None)
        kwargs.setdefault('mass', None)
        kwargs.setdefault('rhs', None)
        kwargs.setdefault('initial_data', None)

        if 'time_stepper_nt' in kwargs:
            kwargs['time_stepper'] = self.time_stepper.with_(nt=kwargs.pop('time_stepper_nt'))

        return super().with_(**kwargs)

    def _solve(self, mu=None):
        mu = self.parse_parameter(mu).copy()

        # explicitly checking if logging is disabled saves the expensive str(mu) call
        if not self.logging_disabled:
            self.logger.info('Solving {} for {} ...'.format(self.name, mu))

        mu['_t'] = 0
        U0 = self.initial_data.as_vector(mu)
        return self.time_stepper.solve(operator=self.operator, rhs=self.rhs, initial_data=U0, mass=self.mass,
                                       initial_time=0, end_time=self.T, mu=mu, num_values=self.num_values)
