#!/usr/bin/env python
# pymor (http://www.pymor.org)
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

# pymor
from pymor.core import interfaces


class Interface(interfaces.BasicInterface):

    id = 'discretization.stationary.detailed'

    def __str__(self):
        return id

    @interfaces.abstractmethod
    def solve(self):
        pass

    @interfaces.abstractmethod
    def visualize(self, vector):
        pass


class Scalar(Interface):

    id = Interface.id + '.scalar'

    def __init__(self, problem, solver, discretizer):
        self.problem = problem
        self.solver = solver
        self._discretizer = discretizer

    def solve(self):
        dof_vector = self.solver.solve(self.problem.operator,
                                       self.problem.functional)
        return self._discretizer.create_discrete_function(dof_vector, 'solution')

    def visualize(self, discrete_function):
        discrete_function.visualize()
