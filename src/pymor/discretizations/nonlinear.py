# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.algorithms.timestepping import explicit_euler
from pymor.la.interfaces import VectorArrayInterface
from pymor.tools import dict_property
from pymor.operators import OperatorInterface, LinearOperatorInterface, ConstantOperator
from pymor.discretizations.interfaces import DiscretizationInterface


class InstationaryNonlinearDiscretization(DiscretizationInterface):

    disable_logging = False
    operator = dict_property('operators', 'operator')
    rhs = dict_property('operators', 'rhs')
    initial_data = dict_property('operators', 'initial_data')

    def __init__(self, operator, rhs, initial_data, T, nt, visualizer=None, name=None):
        assert isinstance(operator, OperatorInterface)
        assert isinstance(rhs, LinearOperatorInterface)
        assert isinstance(initial_data, (VectorArrayInterface, OperatorInterface))
        assert not isinstance(initial_data, OperatorInterface) or initial_data.dim_source == 0
        if isinstance(initial_data, VectorArrayInterface):
            initial_data = ConstantOperator(initial_data, name='initial_data')
        assert operator.dim_source == operator.dim_range == rhs.dim_source == initial_data.dim_range
        assert rhs.dim_range == 1

        super(InstationaryNonlinearDiscretization, self).__init__()
        self.operators = {'operator': operator, 'rhs': rhs, 'initial_data': initial_data}
        self.build_parameter_type(inherits={'operator': operator, 'rhs': rhs, 'initial_data': initial_data},
                                  provides={'_t': 0})
        self.T = T
        self.nt = nt

        if visualizer is not None:
            self.visualize = visualizer

        self.solution_dim = operator.dim_range
        self.name = name

    def with_projected_operators(self, operators, name=None):
        assert set(operators.keys()) == {'operator', 'rhs', 'initial_data'}
        return InstationaryNonlinearDiscretization(T=self.T, nt=self.nt, name=name, **operators)

    with_operators = with_projected_operators

    def _solve(self, mu=None):
        if not self.disable_logging:
            self.logger.info('Solving {} for {} ...'.format(self.name, mu))
        mu_A = self.map_parameter(mu, 'operator', provide={'_t': np.array(0)})
        mu_F = self.map_parameter(mu, 'rhs', provide={'_t': np.array(0)})
        U0 = self.initial_data.apply(0, self.map_parameter(mu, 'initial_data'))
        return explicit_euler(self.operator, self.rhs, U0, 0, self.T, self.nt, mu_A, mu_F)
