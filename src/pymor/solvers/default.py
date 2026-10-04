# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.defaults import defaults
from pymor.core.exceptions import InversionError
from pymor.solvers.interface import Solver
from pymor.tools.weakrefcache import WeakRefCache

_matrix_operators = WeakRefCache()


class DefaultSolver(Solver):
    """Default |Solver|.

    This |Solver| is used by :meth:`~pymor.operators.interface.Operator.apply_inverse`
    and :meth:`~pymor.operators.interface.Operator.apply_inverse_adjoint` when the
    |Operator| has no :attr:`~pymor.operators.interface.Operator.solver` and no solver
    is specified in the method call.

    `DefaultSolver` uses the following strategy:

    1. Try to call `operator._apply_inverse`.
    2. Assemble `operator`. Try `assembled_operator._apply_inverse`.
    3. a) If `operator` is linear, try to convert `assembled_operator` to a
          |NumpyMatrixOperator| using :func:`~pymor.algorithms.to_matrix.to_matrix`
          and call `apply_inverse` on the converted operator.
       b) If `operator` is non-linear, use :class:`~pymor.solvers.newton.NewtonSolver`.

    Parameters
    ----------
    try_to_matrix
        If `False`, do not try to convert linear operators to |NumpyMatrixOperator|.
        Fail instead.
    """

    @defaults('try_to_matrix')
    def __init__(self, try_to_matrix=True):
        self.try_to_matrix = try_to_matrix

    def _solve(self, operator, V, mu, initial_guess):
        # if the operator implement it's own _apply_inverse, use it
        try:
            return operator._apply_inverse(V, mu=mu, initial_guess=initial_guess)
        except NotImplementedError:
            pass

        # see if assembling the operator helps
        from pymor.operators.constructions import FixedParameterOperator
        assembled_op = operator.assemble(mu)
        if (assembled_op is not operator
                and (not isinstance(assembled_op, FixedParameterOperator) or assembled_op.operator is not operator)):
            if assembled_op.solver:
                return assembled_op.solver.solve(assembled_op, V, initial_guess=initial_guess, return_info=True)
            try:
                return assembled_op._apply_inverse(V, mu=None, initial_guess=initial_guess)
            except NotImplementedError:
                pass

        # if the operator is linear, try converting to a matrix as a last resort
        if operator.linear:
            if not self.try_to_matrix:
                raise InversionError(f'{operator!r} has no solver.')
            mat_op = _convert_to_matrix_and_cache(operator, assembled_op, logger=self.logger)
            v = mat_op.range.from_numpy(V.to_numpy())
            i = None if initial_guess is None else mat_op.source.from_numpy(initial_guess.to_numpy())
            u, info = mat_op.apply_inverse(v, initial_guess=i, return_info=True)
            U = operator.source.from_numpy(u.to_numpy())
            return U, info

        with self.logger.block('Solving nonlinear problem using newton algorithm ...'):
            from pymor.solvers.newton import NewtonSolver
            solver = NewtonSolver()
            return solver.solve(operator, V, initial_guess=initial_guess, mu=mu, return_info=True)

    def _solve_adjoint(self, operator, U, mu, initial_guess):
        # if the operator implement it's own _apply_inverse, use it
        try:
            return operator._apply_inverse_adjoint(U, mu=mu, initial_guess=initial_guess)
        except NotImplementedError:
            pass

        # see if assembling the operator helps
        from pymor.operators.constructions import FixedParameterOperator
        assembled_op = operator.assemble(mu)
        if (assembled_op is not operator
                and (not isinstance(assembled_op, FixedParameterOperator) or assembled_op.operator is not operator)):
            if assembled_op.solver is not None:
                return assembled_op.solver.solve_adjoint(assembled_op, U, initial_guess=initial_guess,
                                                         return_info=True)
            try:
                return assembled_op._apply_inverse_adjoint(U, mu=None, initial_guess=initial_guess)
            except NotImplementedError:
                pass

        # try converting to a matrix as a last resort
        mat_op = _convert_to_matrix_and_cache(operator, assembled_op, logger=self.logger)
        u = mat_op.source.from_numpy(U.to_numpy())
        i = None if initial_guess is None else mat_op.range.from_numpy(initial_guess.to_numpy())
        v, info = mat_op.apply_inverse_adjoint(u, initial_guess=i, return_info=True)
        V = operator.range.from_numpy(v.to_numpy())
        return V, info


def _convert_to_matrix_and_cache(op, assembled_op, logger=None):
    """Convert an assembled operator, caching the result for non-parametric operators."""
    cache_key = op
    if not op.parametric:
        try:
            return _matrix_operators.get(cache_key)
        except KeyError:
            pass

    if logger is not None:
        logger.warning(f'No specialized linear solver available for {op}.')
        logger.warning('Trying to solve by converting to NumPy/SciPy matrix.')

    from pymor.algorithms.rules import NoMatchingRuleError
    from pymor.algorithms.to_matrix import to_matrix
    from pymor.operators.numpy import NumpyMatrixOperator

    try:
        mat_op = NumpyMatrixOperator(to_matrix(assembled_op))
    except (NoMatchingRuleError, NotImplementedError) as e:
        raise InversionError(f'{op!r} has no solver, and to_matrix failed.') from e
    if not op.parametric:
        _matrix_operators.set(cache_key, mat_op)
    return mat_op
