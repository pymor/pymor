# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import ImmutableObject
from pymor.solvers.matrix.equations import LyapunovEquation, PositiveRiccatiEquation, RiccatiEquation


class LyapunovSolver(ImmutableObject):
    r"""Interface for solvers computing the dense solution of a |LyapunovEquation|.

    Concrete solvers implement :meth:`_solve`.  When no solver is passed to
    :meth:`~pymor.solvers.matrix.equations.LyapunovEquation.solve`,
    :class:`~pymor.solvers.matrix.default.DefaultLyapunovSolver` is used.
    """

    def solve(self, equation):
        """Solve a |LyapunovEquation|.

        Parameters
        ----------
        equation
            The |LyapunovEquation| to solve.

        Returns
        -------
        X
            |LyapunovEquation| solution as a |NumPy array|.
        """
        assert isinstance(equation, LyapunovEquation)
        return self._solve(equation)

    def _solve(self, equation):
        raise NotImplementedError


class LyapunovSolverLRCF(ImmutableObject):
    """Interface for solvers computing a low-rank Cholesky factor of a |LyapunovEquation|."""

    def solve(self, equation):
        """Solve a |LyapunovEquation|.

        Parameters
        ----------
        equation
            The |LyapunovEquation| to solve.

        Returns
        -------
        Z
            Low-rank Cholesky factor of the solution, |VectorArray| from `A.source`.
        """
        assert isinstance(equation, LyapunovEquation)
        return self._solve(equation)

    def _solve(self, equation):
        raise NotImplementedError


class RiccatiSolver(ImmutableObject):
    r"""Interface for solvers computing the dense solution of a |RiccatiEquation|."""

    def solve(self, equation):
        """Solve a |RiccatiEquation|.

        Parameters
        ----------
        equation
            The |RiccatiEquation| to solve.

        Returns
        -------
        X
            |RiccatiEquation| solution as a |NumPy array|.
        """
        assert isinstance(equation, RiccatiEquation)
        return self._solve(equation)

    def _solve(self, equation):
        raise NotImplementedError


class RiccatiSolverLRCF(ImmutableObject):
    r"""Interface for solvers computing a low-rank Cholesky factor of a |RiccatiEquation|."""

    def solve(self, equation):
        """Solve a |RiccatiEquation|.

        Parameters
        ----------
        equation
            The |RiccatiEquation| to solve.

        Returns
        -------
        Z
            Low-rank Cholesky factor of the solution, |VectorArray| from `A.source`.
        """
        assert isinstance(equation, RiccatiEquation)
        return self._solve(equation)

    def _solve(self, equation):
        raise NotImplementedError


class PositiveRiccatiSolver(ImmutableObject):
    r"""Interface for solvers computing the dense solution of a |PositiveRiccatiEquation|."""

    def solve(self, equation):
        """Solve a |PositiveRiccatiEquation|.

        Parameters
        ----------
        equation
            The |PositiveRiccatiEquation| to solve.

        Returns
        -------
        X
            |PositiveRiccatiEquation| solution as a |NumPy array|.
        """
        assert isinstance(equation, PositiveRiccatiEquation)
        return self._solve(equation)

    def _solve(self, equation):
        raise NotImplementedError


class PositiveRiccatiSolverLRCF(ImmutableObject):
    r"""Interface for solvers computing a LR Cholesky factor of a |PositiveRiccatiEquation|."""

    def solve(self, equation):
        """Solve a |PositiveRiccatiEquation|.

        Parameters
        ----------
        equation
            The |PositiveRiccatiEquation| to solve.

        Returns
        -------
        Z
            Low-rank Cholesky factor of the solution, |VectorArray| from `A.source`.
        """
        assert isinstance(equation, PositiveRiccatiEquation)
        return self._solve(equation)

    def _solve(self, equation):
        raise NotImplementedError
