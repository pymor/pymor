# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import ImmutableObject


class Solver(ImmutableObject):

    least_squares = False
    jacobian_solver = None

    @property
    def adjoint_solver(self):
        return self

    def solve(self, operator, V, mu=None, initial_guess=None):
        assert self.least_squares or operator.source.dim == operator.range.dim
        assert V in operator.range
        assert initial_guess is None or initial_guess in operator.source and len(initial_guess) == len(V)

        # always treat handle the zero dimensional case here to avoid having to do this for every
        # linear solver implementation
        if operator.linear and operator.source.dim == operator.range.dim == 0:
            return operator.source.zeros(len(V))

        return self._solve(operator, V, mu=mu, initial_guess=initial_guess)

    def solve_adjoint(self, operator, U, mu=None, initial_guess=None):
        assert operator.linear
        assert self.least_squares or operator.source.dim == operator.range.dim
        assert U in operator.range
        assert initial_guess is None or initial_guess in operator.range and len(initial_guess) == len(U)

        # always treat handle the zero dimensional case here to avoid having to do this for every
        # linear solver implementation
        if operator.source.dim == operator.range.dim == 0:
            return self.range.zeros(len(U))

        return self._solve(operator.H, U, mu=mu, initial_guess=initial_guess)

    def _solve(self, operator, V, mu, initial_guess):
        raise NotImplementedError


class SolverWithAdjointImpl(Solver):

    @property
    def adjoint_solver(self):
        return AdjointSolver(self)

    def solve_adjoint(self, operator, U, mu=None, initial_guess=None):
        assert operator.linear
        assert self.least_squares or operator.source.dim == operator.range.dim
        assert U in operator.range
        assert initial_guess is None or initial_guess in operator.range and len(initial_guess) == len(U)

        # always treat handle the zero dimensional case here to avoid having to do this for every
        # linear solver implementation
        if operator.source.dim == operator.range.dim == 0:
            return self.range.zeros(len(U))

        return self._solve_adjoint(operator, U, mu=mu, initial_guess=initial_guess)

    def _solve_adjoint(self, operator, U, mu, initial_guess):
        raise NotImplementedError


class AdjointSolver(Solver):

    def __init__(self, solver):
        self.solver = solver

    @property
    def adjoint_solver(self):
        return self.solver

    def _solve(self, operator, V, mu, initial_guess):
        return self.solver.solve_adjoint(operator.H, V, mu=mu, initial_guess=initial_guess)

    def _solve_adjoint(self, operator, U, mu, initial_guess):
        return self.solver.solve(operator.H, U, mu=mu, initial_guess=initial_guess)
