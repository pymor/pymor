# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.algorithms.timestepping import TimeStepperInterface
from pymor.la.interfaces import VectorArrayInterface
from pymor.tools import selfless_arguments
from pymor.operators import OperatorInterface, ConstantOperator
from pymor.discretizations.interfaces import DiscretizationInterface


class InstationaryNonlinearDiscretization(DiscretizationInterface):

    def __init__(self, operator, rhs, initial_data, T, time_stepper=None, products=None, parameter_space=None,
                 estimator=None, visualizer=None, caching='disk', name=None):
        assert isinstance(operator, OperatorInterface)
        assert rhs is None or isinstance(rhs, OperatorInterface) and rhs.linear
        assert isinstance(initial_data, (VectorArrayInterface, OperatorInterface))
        assert not isinstance(initial_data, OperatorInterface) or initial_data.dim_source == 0
        if isinstance(initial_data, VectorArrayInterface):
            initial_data = ConstantOperator(initial_data, name='initial_data')
        assert isinstance(time_stepper, TimeStepperInterface)
        assert operator.dim_source == operator.dim_range == initial_data.dim_range
        assert rhs is None or rhs.dim_source == operator.dim_source and rhs.dim_range == 1

        operators = {'operator': operator, 'rhs': rhs, 'initial_data': initial_data}
        super(InstationaryNonlinearDiscretization, self).__init__(operators=operators, products=products,
                                                                  estimator=estimator, visualizer=visualizer,
                                                                  caching=caching, name=name)
        self.operator = operator
        self.rhs = rhs
        self.initial_data = initial_data
        self.T = T
        self.time_stepper = time_stepper
        self.solution_dim = operator.dim_range
        self.build_parameter_type(inherits=(operator, rhs, initial_data), provides={'_t': 0})
        self.parameter_space = parameter_space

        if hasattr(time_stepper, 'nt'):
            self.with_arguments.add('time_stepper_nt')
        self.lock()

    with_arguments = set(selfless_arguments(__init__)).union(['operators'])

    def with_(self, **kwargs):
        assert set(kwargs.keys()) <= self.with_arguments
        assert 'operators' not in kwargs or not ('rhs' in kwargs or 'operator' in kwargs or 'initial_data' in kwargs)
        assert 'operators' not in kwargs or set(kwargs['operators'].keys()) <= set(('operator', 'rhs', 'initial_data'))
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
            self.logger.info('Solving {} for {} ...'.format(self.name, mu))

        mu['_t'] = 0
        U0 = self.initial_data.apply(0, mu=mu)
        return self.time_stepper.solve(operator=self.operator, rhs=self.rhs, initial_data=U0, initial_time=0,
                                       end_time=self.T, mu=mu)
