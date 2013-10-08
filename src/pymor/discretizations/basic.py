# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.algorithms.timestepping import TimeStepperInterface
from pymor import defaults
from pymor.core import abstractmethod
from pymor.core.cache import CacheableInterface, cached
from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.la import induced_norm, VectorArrayInterface
from pymor.tools import selfless_arguments, FrozenDict
from pymor.operators import OperatorInterface
from pymor.operators.constructions import ConstantOperator
from pymor.parameters import Parametric
from pymor.tools import selfless_arguments


class DiscretizationBase(DiscretizationInterface):

    def __init__(self, operators, products=None, estimator=None, visualizer=None, caching='disk', name=None):
        CacheableInterface.__init__(self, region=caching)
        Parametric.__init__(self)
        self.operators = FrozenDict(operators)
        self.linear = all(op is None or isinstance(op, ConstantOperator) or op.linear for op in operators.itervalues())
        self.products = products
        self.estimator = estimator
        self.visualizer = visualizer
        self.caching = caching
        self.name = name

        if products:
            for k, v in products.iteritems():
                setattr(self, '{}_product'.format(k), v)
                setattr(self, '{}_norm'.format(k), induced_norm(v))
        if estimator is not None:
            self.estimate = self.__estimate
        if visualizer is not None:
            self.visualize = self.__visualize

    @abstractmethod
    def _solve(self, mu=None):
        '''Perform the actual solving.'''
        pass

    @cached
    def solve(self, mu=None):
        '''Solve for a parameter `mu`.

        The result is cached by default.
        '''
        return self._solve(mu)

    def __visualize(self, U, *args, **kwargs):
        self.visualizer.visualize(U, self, *args, **kwargs)

    def __estimate(self, U, mu=None):
        return self.estimator.estimate(U, mu=mu, discretization=self)


class StationaryDiscretization(DiscretizationBase):
    '''Generic class for discretizations of stationary linear problems.

    This class describes discrete problems given by the equation ::

        L_h(μ) ⋅ u_h(μ) = f_h(μ)

    which is to be solved for u_h.

    Parameters
    ----------
    operator
        The operator L_h given as a `LinearOperator`.
    rhs
        The functional f_h given as a `LinearOperator` with `dim_range == 1`.
    visualizer
        A function visualize(U) which visualizes the solution vectors. Can be None,
        in which case no visualization is availabe.
    name
        Name of the discretization.

    Attributes
    ----------
    operator
        The operator L_h. A synonym for operators['operator'].
    operators
        Dictionary of all operators contained in this discretization. The idea is
        that this attribute will be common to all discretizations such that it can
        be used for introspection. Compare the implementation of `reduce_generic_rb`.
        For this class, operators has the keys 'operator' and 'rhs'.
    rhs
        The functional f_h. A synonym for operators['rhs'].
    '''

    sid_ignore = ('visualizer', 'caching', 'name')

    def __init__(self, operator, rhs, products=None, parameter_space=None, estimator=None, visualizer=None,
                 caching='disk', name=None):
        assert isinstance(operator, OperatorInterface) and operator.linear
        assert isinstance(rhs, OperatorInterface) and rhs.linear
        assert operator.dim_source == operator.dim_range == rhs.dim_source
        assert rhs.dim_range == 1

        operators = {'operator': operator, 'rhs': rhs}
        super(StationaryDiscretization, self).__init__(operators=operators, products=products,
                                                       estimator=estimator, visualizer=visualizer,
                                                       caching=caching, name=name)
        self.dim_solution = operator.dim_source
        self.type_solution = operator.type_source
        self.operator = operator
        self.rhs = rhs
        self.operators = operators
        self.build_parameter_type(inherits=(operator, rhs))
        self.parameter_space = parameter_space

    with_arguments = set(selfless_arguments(__init__)).union(['operators'])

    def with_(self, **kwargs):
        assert set(kwargs.keys()) <= self.with_arguments
        assert 'operators' not in kwargs or 'rhs' not in kwargs and 'operator' not in kwargs
        assert 'operators' not in kwargs or set(kwargs['operators'].keys()) <= set(('operator', 'rhs'))

        if 'operators' in kwargs:
            kwargs.update(kwargs.pop('operators'))

        return self._with_via_init(kwargs)

    def _solve(self, mu=None):
        mu = self.parse_parameter(mu)

        # explicitly checking if logging is disabled saves the expensive str(mu) call
        if not self.logging_disabled:
            if self.linear:
                pt = 'sparsity unknown' if getattr(self.operator, 'sparse', None) is None \
                    else ('sparse' if self.operator.sparse else 'dense')
            else:
                pt = 'nonlinear'
            self.logger.info('Solving {} ({}) for {} ...'.format(self.name, pt, mu))

        return self.operator.apply_inverse(self.rhs.as_vector(mu), mu=mu)


class InstationaryDiscretization(DiscretizationBase):

    sid_ignore = ('visualizer', 'caching', 'name')

    def __init__(self, T, initial_data, operator, rhs=None, mass=None, time_stepper=None, products=None,
                 parameter_space=None, estimator=None, visualizer=None, caching='disk', name=None):
        assert isinstance(initial_data, (VectorArrayInterface, OperatorInterface))
        assert not isinstance(initial_data, OperatorInterface) or initial_data.dim_source == 0
        assert isinstance(operator, OperatorInterface)
        assert rhs is None or isinstance(rhs, OperatorInterface) and rhs.linear
        assert mass is None or isinstance(mass, OperatorInterface) and mass.linear
        if isinstance(initial_data, VectorArrayInterface):
            initial_data = ConstantOperator(initial_data, name='initial_data')
        assert isinstance(time_stepper, TimeStepperInterface)
        assert operator.dim_source == operator.dim_range == initial_data.dim_range
        assert rhs is None or rhs.dim_source == operator.dim_source and rhs.dim_range == 1
        assert mass is None or mass.dim_source == mass.dim_range == operator.dim_source

        operators = {'initial_data': initial_data, 'operator': operator, 'rhs': rhs, 'mass': mass}
        super(InstationaryDiscretization, self).__init__(operators=operators, products=products,
                                                         estimator=estimator, visualizer=visualizer,
                                                         caching=caching, name=name)
        self.T = T
        self.dim_solution = operator.dim_source
        self.type_solution = operator.type_source
        self.initial_data = initial_data
        self.operator = operator
        self.rhs = rhs
        self.mass = mass
        self.time_stepper = time_stepper
        self.build_parameter_type(inherits=(initial_data, operator, rhs, mass), provides={'_t': 0})
        self.parameter_space = parameter_space

        if hasattr(time_stepper, 'nt'):
            self.with_arguments.add('time_stepper_nt')

    with_arguments = set(selfless_arguments(__init__)).union(['operators'])

    def with_(self, **kwargs):
        assert set(kwargs.keys()) <= self.with_arguments
        assert 'operators' not in kwargs or not ('rhs' in kwargs or 'operator' in kwargs or 'initial_data' in kwargs or
                                                 'mass' in kwargs)
        assert 'operators' not in kwargs or kwargs['operators'].viewkeys() <= set(('operator', 'rhs', 'initial_data',
                                                                                   'mass'))
        assert 'time_stepper_nt' not in kwargs or 'time_stepper' not in kwargs

        if 'time_stepper_nt' in kwargs:
            kwargs['time_stepper'] = self.time_stepper.with_(nt=kwargs.pop('time_stepper_nt'))

        if 'operators' in kwargs:
            kwargs.update(kwargs.pop('operators'))

        return self._with_via_init(kwargs)

    def _solve(self, mu=None):
        mu = self.parse_parameter(mu).copy()

        # explicitly checking if logging is disabled saves the expensive str(mu) call
        if not self.logging_disabled:
            if self.linear:
                pt = 'sparsity unknown' if getattr(self.operator, 'sparse', None) is None \
                    else ('sparse' if self.operator.sparse else 'dense')
            else:
                pt = 'nonlinear'
            self.logger.info('Solving {} ({}) for {} ...'.format(self.name, pt, mu))

        mu['_t'] = 0
        U0 = self.initial_data.apply(self.initial_data.type_source.zeros(0), mu=mu)
        return self.time_stepper.solve(operator=self.operator, rhs=self.rhs, initial_data=U0, mass=self.mass,
                                       initial_time=0, end_time=self.T, mu=mu)
