from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.sparse.linalg import bicg

import pymor.core as core
from pymor.common import BoundaryType
from pymor.common.domaindescription import Rect
from pymor.common.function.nonparametric import Constant as ConstantFunc


class Elliptic(object):

    def __init__(self, operator, rhs, parameter_dim=None, parameter_map=None, solver=None, visualizer=None):
        self.operator = operator
        self.operators = {operator.name: operator}
        self.rhs = rhs
        if parameter_dim is None:
            self.parameter_dim = operator.parameter_dim + rhs.parameter_dim
        else:
            self.parameter_dim = parameter_dim

        def default_parameter_map(mu):
            return mu[:operator.parameter_dim], mu[operator.parameter_dim:]
        self.parameter_map = parameter_map or default_parameter_map

        def default_solver(A, RHS):
            U, info = bicg(A, RHS)
            return U
        self.solver = solver or default_solver

        if visualizer is not None:
            self.visualize = visualizer


    def solve(self, mu=np.array([])):
        assert mu.size == self.parameter_dim
        mu_operator, mu_rhs = self.parameter_map(mu)
        A = self.operator.matrix(mu_operator)
        RHS = self.rhs.matrix(mu_rhs)
        return self.solver(A, RHS)

