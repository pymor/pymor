# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import ImmutableObject
from pymor.core.exceptions import LinAlgError


class Solver(ImmutableObject):
    r"""Equation solver.

    Solves operator equations of the form

    .. math::
        A(U; \mu) = V

    The operator :math:`A` can be linear or non-linear.
    When :math:`A` is linear, a solver can also be used to
    solve the adjoint equation

    .. math::
        A^H(V; \mu) = U

    for :math:`U`.

    When :attr:`~pymor.solvers.interface.Solver.least_squares` is `True`,
    the equations are solved in a least-squares sense:

    .. math::
        \operatorname{argmin}_{U} \|A(U; \mu) - V\|^2 \quad\text{or}\quad
        \operatorname{argmin}_{V} \|A^H(V; \mu) - U\|^2

    Solvers will typically only work for certain classes of |Operators|.
    In most cases, solvers are invoked by the
    :meth:`~pymor.operators.interface.Operator.apply_inverse` and
    :meth:`~pymor.operators.interface.Operator.apply_inverse_adjoint`
    methods of |Operators|. If an |Operator| has no associated solver,
    :class:`~pymor.solvers.default.DefaultSolver` is used.
    """

    least_squares = False
    """If `True`, the solver solves least-squares problems as defined above."""

    jacobian_solver = None
    """If not `None`, a |Solver| for solving linearized equations.

    Used by :class:`~pymor.solvers.newton.NewtonSolver`.
    If `op` is an |Operator| with `op.solver` not `None`, then
    `op.jacobian(U, mu)` will inhert the `jacobian_solver` of
    `op.solver`.
    """

    @property
    def adjoint_solver(self):
        """'Adjoint' solver with `solve` and `solve_adjoint` swapped.

        If `op` is an |Operator| with `op.solver` not `None`, then
        `op.H` will have `op.solver.adjoint_solver` as solver to ensure
        that `op.apply_inverse_adjoint` and `op.H.apply_inverse` are the
        same algorithms.
        """
        return self if type(self)._solve_adjoint is Solver._solve_adjoint else AdjointSolver(self)

    def solve(self, operator, V, mu=None, initial_guess=None, return_info=False):
        """Solve operator equation.

        Parameters
        ----------
        operator
            The |Operator| :math:`A`.
        V
            The right-hand side |VectorArray| :math:`V`.
        mu
            The |parameter values| for which to solve the equation.
        initial_guess
            |VectorArray| with the same length as `V` containing initial guesses
            for the solution.  Some solvers ignore this parameter.
            If `None`, a solver-dependent default is used.
        return_info
            If `True`, return a dict with additional information on the solution
            process (runtime, iterations, residuals, etc.) as a second return value.

        Returns
        -------
        U
            |VectorArray| containing the solutions.
        info
            Dict with additional information. Only returned when `return_info` is
            `True`.

        Raises
        ------
        InversionError
            The equation could not be solved.
        """
        assert self.least_squares or operator.source.dim == operator.range.dim
        assert V in operator.range
        assert initial_guess is None or initial_guess in operator.source and len(initial_guess) == len(V)

        # always treat the zero-dimensional case here to avoid having to do this for every
        # linear solver implementation
        if operator.linear and operator.source.dim == operator.range.dim == 0:
            return operator.source.zeros(len(V))

        result = self._solve(operator, V, mu=mu, initial_guess=initial_guess)
        assert len(result) == 2
        return result if return_info else result[0]

    def solve_adjoint(self, operator, U, mu=None, initial_guess=None, return_info=False):
        """Solve adjoint operator equation.

        Parameters
        ----------
        operator
            The |Operator| :math:`A`.
        U
            The right-hand side |VectorArray| :math:`U`.
        mu
            The |parameter values| for which to solve the equation.
        initial_guess
            |VectorArray| with the same length as `U` containing initial guesses
            for the solution.  Some solvers ignore this parameter.
            If `None`, a solver-dependent default is used.
        return_info
            If `True`, return a dict with additional information on the solution
            process (runtime, iterations, residuals, etc.) as a second return value.

        Returns
        -------
        V
            |VectorArray| containing the solutions.
        info
            Dict with additional information. Only returned when `return_info` is
            `True`.

        Raises
        ------
        InversionError
            The equation could not be solved.
        """
        assert self.least_squares or operator.source.dim == operator.range.dim
        assert U in operator.source
        assert initial_guess is None or initial_guess in operator.range and len(initial_guess) == len(U)
        if not operator.linear:
            raise LinAlgError('Operator not linear.')

        # always treat the zero-dimensional case here to avoid having to do this for every
        # linear solver implementation
        if operator.source.dim == operator.range.dim == 0:
            return operator.range.zeros(len(U))

        result = self._solve_adjoint(operator, U, mu=mu, initial_guess=initial_guess)
        assert len(result) == 2
        return result if return_info else result[0]

    def _solve(self, operator, V, mu, initial_guess):
        raise NotImplementedError

    def _solve_adjoint(self, operator, U, mu, initial_guess):
        return self._solve(operator.H, U, mu=mu, initial_guess=initial_guess)


class AdjointSolver(Solver):

    def __init__(self, solver):
        self.solver = solver

    @property
    def adjoint_solver(self):
        return self.solver

    def _solve(self, operator, V, mu, initial_guess):
        return self.solver._solve_adjoint(operator.H, V, mu=mu, initial_guess=initial_guess)

    def _solve_adjoint(self, operator, U, mu, initial_guess):
        return self.solver._solve(operator.H, U, mu=mu, initial_guess=initial_guess)
