# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import chain

from pymor.algorithms.timestepping import TimeStepperInterface
from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.la.basic import induced_norm
from pymor.la.interfaces import VectorArrayInterface
from pymor.la.numpyvectorarray import NumpyVectorSpace
from pymor.operators.constructions import VectorOperator
from pymor.operators.interfaces import OperatorInterface
from pymor.parameters.base import Parameter
from pymor.tools.arguments import method_arguments
from pymor.tools.frozendict import FrozenDict


class DiscretizationBase(DiscretizationInterface):
    """Base class for |Discretizations| providing some common functionality."""

    def __init__(self, operators, functionals, vector_operators, products=None, estimator=None, visualizer=None,
                 cache_region='disk', name=None):
        self.operators = FrozenDict(operators)
        self.functionals = FrozenDict(functionals)
        self.vector_operators = FrozenDict(vector_operators)
        self.linear = all(op is None or op.linear for op in chain(operators.itervalues(), functionals.itervalues()))
        self.products = products
        self.estimator = estimator
        self.visualizer = visualizer
        self.cache_region = cache_region
        self.name = name

        if products:
            for k, v in products.iteritems():
                setattr(self, '{}_product'.format(k), v)
                setattr(self, '{}_norm'.format(k), induced_norm(v))

    def visualize(self, U, **kwargs):
        """Visualize a solution |VectorArray| U.

        Parameters
        ----------
        U
            The |VectorArray| from :attr:`~DiscretizationInterface.solution_space`
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
        not `None` an `estimate(U, mu)` method is added to the
        discretization which will call `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, discretization, ...)` method. If `visualizer`
        is not `None` a `visualize(U, *args, **kwargs)` method is added
        to the discretization, which forwards its arguments to the
        visualizer's `visualize` method.
    cache_region
        `None` or name of the cache region to use. See
        :mod:`pymor.core.cache`.
    name
        Name of the discretization.

    Attributes
    ----------
    operator
        The |Operator| L. The same as `operators['operator']`.
    rhs
        The |Functional| F. The same as `functionals['rhs']`.
    """

    sid_ignore = ('visualizer', 'cache_region', 'name')

    def __init__(self, operator, rhs, products=None, operators=None, functionals=None, vector_operators=None,
                 parameter_space=None, estimator=None, visualizer=None, cache_region='disk', name=None):
        functionals = functionals or {}
        operators = operators or {}
        vector_operators = vector_operators or {}
        assert isinstance(operator, OperatorInterface)
        assert isinstance(rhs, OperatorInterface) and rhs.linear
        assert operator.source == operator.range == rhs.source
        assert rhs.range.dim == 1
        assert all(f.source == operator.source for f in functionals.values())
        assert 'operator' not in operators or operator == operators['operator']
        assert 'rhs' not in functionals or rhs == functionals['rhs']

        operators_with_operator = {'operator': operator}
        operators_with_operator.update(operators)
        functionals_with_rhs = {'rhs': rhs}
        functionals_with_rhs.update(functionals)
        super(StationaryDiscretization, self).__init__(operators=operators_with_operator,
                                                       functionals=functionals_with_rhs,
                                                       vector_operators=vector_operators, products=products,
                                                       estimator=estimator, visualizer=visualizer,
                                                       cache_region=cache_region, name=name)
        self.solution_space = operator.source
        self.operator = operator
        self.rhs = rhs
        self.build_parameter_type(inherits=(operator, rhs))
        self.parameter_space = parameter_space

    def with_(self, **kwargs):
        assert set(kwargs.keys()) <= self.with_arguments

        if 'operator' not in kwargs and 'operators' in kwargs and 'operator' in kwargs['operators']:
            kwargs['operator'] = kwargs['operators']['operator']
        elif 'operator' in kwargs and 'operators' not in kwargs:
            operators = dict(self.operators)
            operators['operator'] = kwargs['operator']
            kwargs['operators'] = operators

        if 'rhs' not in kwargs and 'functionals' in kwargs and 'rhs' in kwargs['functionals']:
            kwargs['rhs'] = kwargs['functionals']['rhs']
        elif 'rhs' in kwargs and 'functionals' not in kwargs:
            functionals = dict(self.functionals)
            functionals['rhs'] = kwargs['rhs']
            kwargs['functionals'] = functionals

        return self._with_via_init(kwargs)

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
        The end-time T.
    initial_data
        The initial data u_0. Either a |VectorArray| of length 1 or
        (for the |Parameter|-dependent case) a vector-like |Operator|
        (i.e. a linear |Operator| with `source.dim == 1`) which
        applied to `NumpyVectorArray(np.array([1]))` will yield the
        initial data for a given |Parameter|.
    operator
        The |Operator| L.
    rhs
        The |Functional| F.
    mass
        The mass |Operator| `M`. If `None` the identity is assumed.
    time_stepper
        T time-stepper to be used by :meth:`solve`. Has to satisfy
        the :class:`~pymor.algorithms.timestepping.TimeStepperInterface`.
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
        not `None` an `estimate(U, mu)` method is added to the
        discretization which will call `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, discretization, ...)` method. If `visualizer`
        is not `None` a `visualize(U, *args, **kwargs)` method is added
        to the discretization, which forwards its arguments to the
        visualizer's `visualize` method.
    cache_region
        `None` or name of the cache region to use. See
        :mod:`pymor.core.cache`.
    name
        Name of the discretization.

    Attributes
    ----------
    T
        The end-time.
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
        The provided time-stepper.
    """

    sid_ignore = ('visualizer', 'cache_region', 'name')

    def __init__(self, T, initial_data, operator, rhs=None, mass=None, time_stepper=None, num_values=None,
                 products=None, operators=None, functionals=None, vector_operators=None, parameter_space=None,
                 estimator=None, visualizer=None, cache_region='disk', name=None):
        functionals = functionals or {}
        operators = operators or {}
        vector_operators = vector_operators or {}
        if isinstance(initial_data, VectorArrayInterface):
            initial_data = VectorOperator(initial_data, name='initial_data')

        assert isinstance(initial_data, OperatorInterface)
        assert initial_data.source == NumpyVectorSpace(1)
        assert 'initial_data' not in vector_operators or initial_data == vector_operators['initial_data']

        assert isinstance(operator, OperatorInterface)
        assert operator.source == operator.range == initial_data.range
        assert 'operator' not in operators or operator == operators['operator']

        assert rhs is None or isinstance(rhs, OperatorInterface) and rhs.linear
        assert rhs is None or rhs.source == operator.source and rhs.range.dim == 1
        assert 'rhs' not in functionals or rhs == functionals['rhs']

        assert mass is None or isinstance(mass, OperatorInterface) and mass.linear
        assert mass is None or mass.source == mass.range == operator.source
        assert 'mass' not in operators or mass == operators['mass']

        assert isinstance(time_stepper, TimeStepperInterface)
        assert all(f.source == operator.source for f in functionals.values())

        operators_with_operator_mass = {'operator': operator, 'mass': mass}
        operators_with_operator_mass.update(operators)

        functionals_with_rhs = {'rhs': rhs} if rhs else {}
        functionals_with_rhs.update(functionals)

        vector_operators_with_initial_data = {'initial_data': initial_data}
        vector_operators_with_initial_data.update(vector_operators)

        super(InstationaryDiscretization, self).__init__(operators=operators_with_operator_mass,
                                                         functionals=functionals_with_rhs,
                                                         vector_operators=vector_operators_with_initial_data,
                                                         products=products, estimator=estimator,
                                                         visualizer=visualizer, cache_region=cache_region, name=name)
        self.T = T
        self.solution_space = operator.source
        self.initial_data = initial_data
        self.operator = operator
        self.rhs = rhs
        self.mass = mass
        self.time_stepper = time_stepper
        self.num_values = num_values
        self.build_parameter_type(inherits=(initial_data, operator, rhs, mass), provides={'_t': 0})
        self.parameter_space = parameter_space

        if hasattr(time_stepper, 'nt'):
            self.with_arguments = self.with_arguments.union({'time_stepper_nt'})

    with_arguments = frozenset(method_arguments(__init__))  # needed in order to be ably to modify with_arguments
                                                            # during  __init__

    def with_(self, **kwargs):
        assert set(kwargs.keys()) <= self.with_arguments

        if 'operator' not in kwargs and 'operators' in kwargs and 'operator' in kwargs['operators']:
            kwargs['operator'] = kwargs['operators']['operator']
        if 'mass' not in kwargs and 'operators' in kwargs and 'mass' in kwargs['operators']:
            kwargs['mass'] = kwargs['operators']['mass']
        if 'operators' not in kwargs:
            operators = dict(self.operators)
            if 'operator' in kwargs:
                operators['operator'] = kwargs['operator']
            if 'mass' in kwargs:
                operators['mass'] = kwargs['mass']
            kwargs['operators'] = operators

        if 'rhs' not in kwargs and 'functionals' in kwargs and 'rhs' in kwargs['functionals']:
            kwargs['rhs'] = kwargs['functionals']['rhs']
        elif 'rhs' in kwargs and 'functionals' not in kwargs:
            functionals = dict(self.functionals)
            functionals['rhs'] = kwargs['rhs']
            kwargs['functionals'] = functionals

        if ('initial_data' not in kwargs and 'vector_operators' in kwargs
                and 'initial_data' in kwargs['vector_operators']):
            kwargs['initial_data'] = kwargs['vector_operators']['initial_data']
        elif 'initial_data' in kwargs and 'vector_operators' not in kwargs:
            vector_operators = dict(self.vector_operators)
            vector_operators.pop('initial_data', None)
            kwargs['vector_operators'] = vector_operators

        assert 'time_stepper_nt' not in kwargs or 'time_stepper' not in kwargs
        if 'time_stepper_nt' in kwargs:
            kwargs['time_stepper'] = self.time_stepper.with_(nt=kwargs.pop('time_stepper_nt'))

        return self._with_via_init(kwargs)

    def _solve(self, mu=None):
        mu = self.parse_parameter(mu).copy() if self.parametric else Parameter({})

        # explicitly checking if logging is disabled saves the expensive str(mu) call
        if not self.logging_disabled:
            self.logger.info('Solving {} for {} ...'.format(self.name, mu))

        mu['_t'] = 0
        U0 = self.initial_data.as_vector(mu)
        return self.time_stepper.solve(operator=self.operator, rhs=self.rhs, initial_data=U0, mass=self.mass,
                                       initial_time=0, end_time=self.T, mu=mu, num_values=self.num_values)
