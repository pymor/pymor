# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from itertools import chain

from pymor.algorithms.timestepping import TimeStepperInterface
from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.operators.constructions import VectorOperator, induced_norm
from pymor.operators.interfaces import OperatorInterface
from pymor.tools.frozendict import FrozenDict
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.numpy import scalars


class DiscretizationBase(DiscretizationInterface):
    """Base class for |Discretizations| providing some common functionality."""

    sid_ignore = DiscretizationInterface.sid_ignore | {'visualizer'}
    special_operators = frozenset()
    special_functionals = frozenset()
    special_vector_operators = frozenset()

    def __init__(self, operators, functionals, vector_operators, products=None, estimator=None, visualizer=None,
                 cache_region=None, name=None, **kwargs):

        operators = {} if operators is None else dict(operators)
        functionals = {} if functionals is None else dict(functionals)
        vector_operators = {} if vector_operators is None else dict(vector_operators)

        # handle special operators
        for operator_dict, operator_names in [(operators, self.special_operators),
                                              (functionals, self.special_functionals),
                                              (vector_operators, self.special_vector_operators)]:
            for on in operator_names:
                # special operators may not already exist as attributes
                assert not hasattr(self, on)
                # special operators must be uniquely given
                assert kwargs[on] is None or on not in operator_dict or kwargs[on] == operator_dict[on]

                op = kwargs[on]
                if op is None:
                    op = operator_dict.get(on)

                assert op is None or isinstance(op, OperatorInterface)

                setattr(self, on, op)
                operator_dict[on] = op

        self.operators = FrozenDict(operators)
        self.functionals = FrozenDict(functionals)
        self.vector_operators = FrozenDict(vector_operators)
        self.linear = all(op is None or op.linear for op in chain(operators.values(), functionals.values()))
        self.products = products
        self.estimator = estimator
        self.visualizer = visualizer
        self.enable_caching(cache_region)
        self.name = name

        if products:
            for k, v in products.items():
                setattr(self, '{}_product'.format(k), v)
                setattr(self, '{}_norm'.format(k), induced_norm(v))

    def with_(self, **kwargs):
        assert set(kwargs.keys()) <= self.with_arguments

        for operator_dict, operator_names in [('operators', self.special_operators),
                                              ('functionals', self.special_functionals),
                                              ('vector_operators', self.special_vector_operators)]:

            # when an operator dict is not specified but a special operator contained in the dict,
            # make sure that we use the old operator dict but with updated special operators
            kwargs.setdefault(operator_dict,
                              dict(getattr(self, operator_dict),
                                   **{on: kwargs.get(on, getattr(self, on)) for on in operator_names}))

            # make sure we do not use old special operators in case the corresponding operator dict
            # is specified
            for on in operator_names:
                kwargs.setdefault(on, None)

        return super(DiscretizationBase, self).with_(**kwargs)

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
    products
        A dict of inner product |Operators| defined on the discrete space the
        problem is posed on. For each product a corresponding norm
        is added as a method of the discretization.
    operators
        A dict of |Operators| associated with the discretization.
    functionals
        A dict of (output) |Functionals| associated with the discretization.
    vector_operators
        A dict of vector-like |Operators| associated with the discretization.
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

    special_operators = frozenset({'operator'})
    special_functionals = frozenset({'rhs'})

    def __init__(self, operator=None, rhs=None, products=None, operators=None, functionals=None, vector_operators=None,
                 parameter_space=None, estimator=None, visualizer=None, cache_region=None, name=None):
        super().__init__(operator=operator, rhs=rhs,
                         operators=operators,
                         functionals=functionals,
                         vector_operators=vector_operators,
                         products=products,
                         estimator=estimator, visualizer=visualizer,
                         cache_region=cache_region, name=name)
        self.solution_space = self.operator.source
        self.build_parameter_type(self.operator, self.rhs)
        self.parameter_space = parameter_space

        assert self.operator is not None
        assert self.rhs is not None and self.rhs.linear
        assert self.operator.source == self.operator.range == self.rhs.source
        assert self.rhs.range.dim == 1
        assert all(f.source == self.operator.source for f in self.functionals.values())

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
    products
        A dict of product |Operators| defined on the discrete space the
        problem is posed on. For each product a corresponding norm
        is added as a method of the discretization.
    operators
        A dict of |Operators| associated with the discretization.
    functionals
        A dict of (output) |Functionals| associated with the discretization.
    vector_operators
        A dict of vector-like |Operators| associated with the discretization.
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

    special_operators = frozenset({'operator', 'mass'})
    special_functionals = frozenset({'rhs'})
    special_vector_operators = frozenset({'initial_data'})

    def __init__(self, T, initial_data=None, operator=None, rhs=None, mass=None, time_stepper=None, num_values=None,
                 products=None, operators=None, functionals=None, vector_operators=None, parameter_space=None,
                 estimator=None, visualizer=None, cache_region=None, name=None):

        if isinstance(initial_data, VectorArrayInterface):
            initial_data = VectorOperator(initial_data, name='initial_data')

        super().__init__(initial_data=initial_data, operator=operator,
                         rhs=rhs, mass=mass,
                         operators=operators, functionals=functionals,
                         vector_operators=vector_operators,
                         products=products, estimator=estimator,
                         visualizer=visualizer, cache_region=cache_region, name=name)
        self.T = T
        self.solution_space = self.operator.source
        self.time_stepper = time_stepper
        self.num_values = num_values
        self.build_parameter_type(self.initial_data, self.operator, self.rhs, self.mass, provides={'_t': 0})
        self.parameter_space = parameter_space
        if hasattr(time_stepper, 'nt'):
            self.add_with_arguments = self.add_with_arguments | {'time_stepper_nt'}

        assert isinstance(time_stepper, TimeStepperInterface)
        assert self.initial_data.source == scalars(1)
        assert self.operator.source == self.operator.range == self.initial_data.range
        assert self.rhs is None \
            or self.rhs.linear and self.rhs.source == self.operator.source and self.rhs.range.dim == 1
        assert self.mass is None \
            or self.mass.linear and self.mass.source == self.mass.range == self.operator.source
        assert all(f.source == self.operator.source for f in self.functionals.values() if f)

    def with_(self, **kwargs):
        assert set(kwargs.keys()) <= self.with_arguments
        assert 'time_stepper_nt' not in kwargs or 'time_stepper' not in kwargs
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
