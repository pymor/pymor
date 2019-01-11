# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.timestepping import TimeStepperInterface
from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.operators.constructions import VectorOperator, induced_norm
from pymor.operators.interfaces import OperatorInterface
from pymor.tools.frozendict import FrozenDict
from pymor.vectorarrays.interfaces import VectorArrayInterface


class DiscretizationBase(DiscretizationInterface):
    """Base class for |Discretizations| providing some common functionality."""

    sid_ignore = DiscretizationInterface.sid_ignore | {'visualizer'}
    add_with_arguments = DiscretizationInterface.add_with_arguments | {'operators'}
    special_operators = frozenset()

    def __init__(self, operators=None, products=None, estimator=None, visualizer=None,
                 cache_region=None, name=None, **kwargs):

        operators = {} if operators is None else dict(operators)

        # handle special operators
        for on in self.special_operators:
            # special operators must be provided as keyword argument to __init__
            assert on in kwargs
            # special operators may not already exist as attributes
            assert not hasattr(self, on)
            # special operators may not be contained in operators dict
            assert on not in operators

            op = kwargs[on]
            # operators either have to be None or an Operator
            assert op is None \
                or isinstance(op, OperatorInterface) \
                or all(isinstance(o, OperatorInterface) for o in op)
            # set special operator as an attribute
            setattr(self, on, op)
            # add special operator to the operators dict
            operators[on] = op

        self.operators = FrozenDict(operators)
        self.linear = all(op is None or op.linear for op in operators.values())
        self.products = FrozenDict(products or {})
        self.estimator = estimator
        self.visualizer = visualizer
        self.enable_caching(cache_region)
        self.name = name

        if products:
            for k, v in products.items():
                setattr(self, '{}_product'.format(k), v)
                setattr(self, '{}_norm'.format(k), induced_norm(v))

        self.build_parameter_type(*operators.values())

    def with_(self, **kwargs):
        assert set(kwargs.keys()) <= self.with_arguments

        if 'operators' in kwargs:
            # extract special operators from provided operators dict
            operators = kwargs['operators'].copy()
            for on in self.special_operators:
                if on in operators:
                    assert on not in kwargs or kwargs[on] == operators[on]
                    kwargs[on] = operators.pop(on)
            kwargs['operators'] = operators
        else:
            # when an operators dict is not specified make sure that we use the old operators dict
            # but without the special operators
            kwargs['operators'] = {on: op for on, op in self.operators.items()
                                   if on not in self.special_operators}

        # delete empty 'operators' dicts for cases where __init__ does not take
        # an 'operator' argument
        if 'operators' not in self._init_arguments:
            operators = kwargs.pop('operators')
            # in that case, there should not be any operators left in 'operators'
            assert not operators

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
            return self.estimator.estimate(U, mu=mu, d=self)
        else:
            raise NotImplementedError('Discretization has no estimator.')


class StationaryDiscretization(DiscretizationBase):
    """Generic class for discretizations of stationary problems.

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
    products
        A dict of inner product |Operators| defined on the discrete space the
        problem is posed on. For each product a corresponding norm
        is added as a method of the discretization.
    operators
        A dict of additional |Operators| associated with the discretization.
    parameter_space
        The |ParameterSpace| for which the discrete problem is posed.
    estimator
        An error estimator for the problem. This can be any object with
        an `estimate(U, mu, d)` method. If `estimator` is
        not `None`, an `estimate(U, mu)` method is added to the
        discretization which will call `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, d, ...)` method. If `visualizer`
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
        The right-hand side F. The same as `operators['rhs']`.
    operators
        Dict of all |Operators| appearing in the discretization.
    products
        Dict of all product |Operators| associated with the discretization.
    """

    special_operators = frozenset({'operator', 'rhs'})

    def __init__(self, operator, rhs, products=None, operators=None,
                 parameter_space=None, estimator=None, visualizer=None, cache_region=None, name=None):

        if isinstance(rhs, VectorArrayInterface):
            assert rhs in operator.range
            rhs = VectorOperator(rhs, name='rhs')

        assert rhs.range == operator.range and rhs.source.is_scalar and rhs.linear

        super().__init__(operator=operator, rhs=rhs,
                         operators=operators,
                         products=products,
                         estimator=estimator, visualizer=visualizer,
                         cache_region=cache_region, name=name)
        self.solution_space = self.operator.source
        self.parameter_space = parameter_space

    def as_generic_type(self):
        if type(self) is StationaryDiscretization:
            return self
        operators = {k: o for k, o in self.operators.items() if k not in self.special_operators}
        return StationaryDiscretization(
            self.operator, self.rhs, self.products, operators,
            self.parameter_space, self.estimator, self.visualizer, self.cache_region, self.name
        )

    def _solve(self, mu=None):
        mu = self.parse_parameter(mu)

        # explicitly checking if logging is disabled saves the str(mu) call
        if not self.logging_disabled:
            self.logger.info('Solving {} for {} ...'.format(self.name, mu))

        return self.operator.apply_inverse(self.rhs.as_range_array(mu), mu=mu)


class InstationaryDiscretization(DiscretizationBase):
    """Generic class for discretizations of instationary problems.

    This class describes instationary problems given by the equations::

        M * ∂_t u(t, μ) + L(u(μ), t, μ) = F(t, μ)
                                u(0, μ) = u_0(μ)

    for t in [0,T], where L is a (possibly non-linear) time-dependent
    |Operator|, F is a time-dependent vector-like |Operator|, and u_0 the
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
        The right-hand side F.
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
        A dict of additional |Operators| associated with the discretization.
    parameter_space
        The |ParameterSpace| for which the discrete problem is posed.
    estimator
        An error estimator for the problem. This can be any object with
        an `estimate(U, mu, d)` method. If `estimator` is
        not `None`, an `estimate(U, mu)` method is added to the
        discretization which will call `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, d, ...)` method. If `visualizer`
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
        as `operators['initial_data']`.
    operator
        The |Operator| L. The same as `operators['operator']`.
    rhs
        The right-hand side F. The same as `operators['rhs']`.
    mass
        The mass operator M. The same as `operators['mass']`.
    time_stepper
        The provided :class:`time-stepper <pymor.algorithms.timestepping.TimeStepperInterface>`.
    operators
        Dict of all |Operators| appearing in the discretization.
    products
        Dict of all product |Operators| associated with the discretization.
    """

    special_operators = frozenset({'operator', 'mass', 'rhs', 'initial_data'})

    def __init__(self, T, initial_data, operator, rhs, mass=None, time_stepper=None, num_values=None,
                 products=None, operators=None, parameter_space=None, estimator=None, visualizer=None,
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

        super().__init__(initial_data=initial_data, operator=operator, rhs=rhs, mass=mass,
                         operators=operators, products=products, estimator=estimator,
                         visualizer=visualizer, cache_region=cache_region, name=name)
        self.T = T
        self.solution_space = self.operator.source
        self.time_stepper = time_stepper
        self.num_values = num_values
        self.build_parameter_type(self.initial_data, self.operator, self.rhs, self.mass, provides={'_t': 0})
        self.parameter_space = parameter_space
        if hasattr(time_stepper, 'nt'):
            self.add_with_arguments = self.add_with_arguments | {'time_stepper_nt'}

    def as_generic_type(self):
        if type(self) is StationaryDiscretization:
            return self
        operators = {k: o for k, o in self.operators.items() if k not in self.special_operators}
        return InstationaryDiscretization(
            self.T, self.initial_data, self.operator, self.rhs, self.mass, self.time_stepper, self.num_values,
            self.products, operators, self.parameter_space, self.estimator, self.visualizer,
            self.cache_region, self.name
        )

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
        U0 = self.initial_data.as_range_array(mu)
        return self.time_stepper.solve(operator=self.operator, rhs=self.rhs, initial_data=U0, mass=self.mass,
                                       initial_time=0, end_time=self.T, mu=mu, num_values=self.num_values)

    def to_lti(self, output='output_functional'):
        """Convert discretization to |LTISystem|.

        This method interprets the given discretization as an |LTISystem|
        in the following way::

            - self.operator        -> A
            self.rhs               -> B
            self.operators[output] -> C
            None                   -> D
            self.mass              -> E


        Parameters
        ----------
        output
            Key in `self.operators` to use as output functional.
        """
        A = - self.operator
        B = self.rhs
        C = self.operators[output]
        E = self.mass

        if not all(op.linear for op in [A, B, C, E]):
            raise ValueError('Operators not linear.')
        if A.source.id == B.source.id:
            raise ValueError('State space must have different id than input space.')
        if A.source.id == C.range.id:
            raise ValueError('State space must have different id than output space.')

        from pymor.discretizations.iosys import LTISystem
        return LTISystem(A, B, C, E=E, visualizer=self.visualizer)
