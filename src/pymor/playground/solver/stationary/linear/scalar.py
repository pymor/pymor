#!/usr/bin/env python

# scipy
from scipy.sparse.linalg import cg as la_solver

# pymor
from pymor.core import interfaces


class Interface(interfaces.BasicInterface):

    id = 'solver.stationary.linear.scalar'

    def __str__(self):
        return id

    @interfaces.abstractmethod
    def solve(self):
        pass


class Scipy(Interface):

    id = Interface.id + 'scipy'

    def __init__(self):
        pass

    def solve(self, operator, functional):
        assert operator.size[1] == functional.size
        system_matrix = operator.matrix
        right_hand_side = functional.vector
        result = la_solver(system_matrix,
                           right_hand_side,
                           tol=1.e-12,
                           maxiter=10000)
        return result[0]
