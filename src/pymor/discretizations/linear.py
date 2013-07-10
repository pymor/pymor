# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core import defaults
from pymor.la import NumpyVectorArray
from pymor.tools import dict_property
from pymor.operators import LinearOperatorInterface
from pymor.operators.solvers import solve_linear
from pymor.discretizations.interfaces import DiscretizationInterface


class StationaryLinearDiscretization(DiscretizationInterface):
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
    solver
        A function solver(A, RHS), which solves the matrix equation A*x = RHS.
        If None, `pymor.operators.solvers.solve_linear()` is chosen.
    visualizer
        A function visualize(U) which visualizes the solution vectors. Can be None,
        in which case no visualization is availabe.
    name
        Name of the discretization.

    Attributes
    ----------
    disable_logging
        If True, no log message is displayed when calling solve. This is useful if
        we want to log solves of detailed discretization but not of reduced ones.
        In the future, this should be a feature of BasicInterface.
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

    operator = dict_property('operators', 'operator')
    rhs = dict_property('operators', 'rhs')

    _logging_disabled = False
    @property
    def logging_disabled(self):
        return self._logging_disabled

    def __init__(self, operator, rhs, solver=None, visualizer=None, parameter_space=None, name=None):
        assert isinstance(operator, LinearOperatorInterface)
        assert isinstance(rhs, LinearOperatorInterface)
        assert operator.dim_source == operator.dim_range == rhs.dim_source
        assert rhs.dim_range == 1

        super(StationaryLinearDiscretization, self).__init__()
        self.operators = {'operator': operator, 'rhs': rhs}
        self.build_parameter_type(inherits={'operator': operator, 'rhs': rhs})
        self.parameter_space = parameter_space

        self.solver = solver or solve_linear

        if visualizer is not None:
            self.visualize = visualizer

        self.solution_dim = operator.dim_range
        self.name = name
        self.lock()

    _with_arguments = set(('operators', 'operator', 'rhs', 'solver', 'visualizer', 'parameter_space', 'name'))

    def with_(self, **kwargs):
        assert 'operators' not in kwargs or 'rhs' not in kwargs and 'operator' not in kwargs
        assert 'operators' not in kwargs or set(kwargs['operators'].keys()) <= set(('operator', 'rhs'))

        if not 'visualizer' in kwargs:
            kwargs['visualizer'] = self.visualize if hasattr(self, 'visualize') else None
        if 'operators' in kwargs:
            kwargs.update(kwargs.pop('operators'))

        return self._with_via_init(kwargs)

    def _solve(self, mu=None):
        A = self.operator.assemble(self.map_parameter(mu, 'operator'))
        RHS = self.rhs.assemble(self.map_parameter(mu, 'rhs')).as_vector_array()

        if not self.logging_disabled:
            sparse = 'sparsity unknown' if A.sparse is None else ('sparse' if A.sparse else 'dense')
            self.logger.info('Solving {} ({}) for {} ...'.format(self.name, sparse, mu))

        return self.solver(A, RHS)

    def disable_logging(self, doit=True):
        self._logging_disabled = doit

    def enable_logging(self, doit=True):
        self._logging_disabled = not doit
