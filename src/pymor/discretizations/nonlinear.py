# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.algorithms.timestepping import explicit_euler
from pymor.la.interfaces import VectorArrayInterface
from pymor.tools import selfless_arguments
from pymor.operators import OperatorInterface, LinearOperatorInterface, ConstantOperator
from pymor.discretizations.interfaces import DiscretizationInterface


class InstationaryNonlinearDiscretization(DiscretizationInterface):

    def __init__(self, operator, rhs, initial_data, T, nt, parameter_space=None, estimator=None, visualizer=None,
                 caching='disk', name=None):
        assert isinstance(operator, OperatorInterface)
        assert isinstance(rhs, LinearOperatorInterface)
        assert isinstance(initial_data, (VectorArrayInterface, OperatorInterface))
        assert not isinstance(initial_data, OperatorInterface) or initial_data.dim_source == 0
        if isinstance(initial_data, VectorArrayInterface):
            initial_data = ConstantOperator(initial_data, name='initial_data')
        assert operator.dim_source == operator.dim_range == rhs.dim_source == initial_data.dim_range
        assert rhs.dim_range == 1

        operators = {'operator': operator, 'rhs': rhs, 'initial_data': initial_data}
        super(InstationaryNonlinearDiscretization, self).__init__(operators=operators, estimator=estimator,
                                                                  visualizer=visualizer, caching=caching, name=name)
        self.operator = operator
        self.rhs = rhs
        self.initial_data = initial_data
        self.T = T
        self.nt = nt
        self.solution_dim = operator.dim_range
        self.build_parameter_type(inherits={'operator': operator, 'rhs': rhs, 'initial_data': initial_data},
                                  provides={'_t': 0})
        self.parameter_space = parameter_space
        self.lock()

    with_arguments = set(selfless_arguments(__init__)).union(['operators'])

    def with_(self, **kwargs):
        assert 'operators' not in kwargs or not ('rhs' in kwargs or 'operator' in kwargs or 'initial_data' in kwargs)
        assert 'operators' not in kwargs or set(kwargs['operators'].keys()) <= set(('operator', 'rhs', 'initial_data'))

        if 'operators' in kwargs:
            kwargs.update(kwargs.pop('operators'))

        return self._with_via_init(kwargs)

    def _solve(self, mu=None):
        self.logger.info('Solving {} for {} ...'.format(self.name, mu))
        mu_A = self.map_parameter(mu, 'operator', provide={'_t': np.array(0)})
        mu_F = self.map_parameter(mu, 'rhs', provide={'_t': np.array(0)})
        U0 = self.initial_data.apply(0, self.map_parameter(mu, 'initial_data'))
        return explicit_euler(self.operator, self.rhs, U0, 0, self.T, self.nt, mu_A, mu_F)
