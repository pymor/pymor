#!/usr/bin/env python

# scipy
from scipy.sparse.linalg import bicgstab

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
        result = bicgstab(system_matrix, right_hand_side)
        return result[0]
