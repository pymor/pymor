from __future__ import absolute_import, division, print_function

import copy

import numpy as np
from scipy.sparse.linalg import bicg
from scipy.sparse import issparse

from pymor.core import BasicInterface, defaults
from pymor.core.cache import Cachable, cached, DEFAULT_DISK_CONFIG
from pymor.tools import dict_property, Named
from pymor.domaindescriptions import BoundaryType
from pymor.parameters import Parametric
from pymor.discreteoperators import LinearDiscreteOperatorInterface


class StationaryLinearDiscretization(BasicInterface, Parametric, Cachable, Named):

    disable_logging = False
    operator = dict_property('operators', 'operator')
    rhs = dict_property('operators', 'rhs')

    def __init__(self, operator, rhs, solver=None, visualizer=None, name=None):
        assert isinstance(operator, LinearDiscreteOperatorInterface)
        assert isinstance(rhs, LinearDiscreteOperatorInterface)
        assert operator.source_dim == operator.range_dim == rhs.source_dim
        assert rhs.range_dim == 1

        Cachable.__init__(self, config=DEFAULT_DISK_CONFIG)
        self.operators = {'operator': operator, 'rhs':rhs}
        self.build_parameter_type(inherits={'operator':operator, 'rhs':rhs})

        def default_solver(A, RHS):
            if issparse(A):
                U, _ = bicg(A, RHS, tol=defaults.bicg_tol, maxiter=defaults.bicg_maxiter)
            else:
                U = np.linalg.solve(A, RHS)
            return U
        self.solver = solver or default_solver

        if visualizer is not None:
            self.visualize = visualizer

        self.solution_dim = operator.range_dim
        self.name = name

    def copy(self):
        c = copy.copy(self)
        c.operators = c.operators.copy()
        Cachable.__init__(c)
        return c

    @cached
    def solve(self, mu={}):
        mu = self.parse_parameter(mu)
        A = self.operator.matrix(self.map_parameter(mu, 'operator'))

        if not self.disable_logging:
            sparse = 'sparse' if issparse(A) else 'dense'
            self.logger.info('Solving {} ({}) for {} ...'.format(self.name, sparse, mu))

        if A.size == 0:
            return np.zeros(0)
        RHS = np.squeeze(self.rhs.matrix(self.map_parameter(mu, 'rhs')))
        if RHS.ndim == 0:
            RHS = RHS[np.newaxis]

        return self.solver(A, RHS)

