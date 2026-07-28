# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import ImmutableObject, abstractmethod
from pymor.solvers.matrix_equations.equations import (
    LyapunovEquation,
    PositiveRiccatiEquation,
    RiccatiEquation,
    SylvesterEquation,
)


class LyapunovSolver(ImmutableObject):
    r"""Interface for solvers computing the dense solution of a |LyapunovEquation|."""

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

    @abstractmethod
    def _solve(self, equation):
        pass


class LyapunovSolverLRCF(ImmutableObject):
    """Interface for solvers computing a LR CF of the solution of a |LyapunovEquation|."""

    def solve(self, equation):
        """Solve a |LyapunovEquation|.

        Parameters
        ----------
        equation
            The |LyapunovEquation| to solve.

        Returns
        -------
        Z
            Low-rank Cholesky factor of the solution, |VectorArray| from `equation.A.source`.
        """
        assert isinstance(equation, LyapunovEquation)
        return self._solve(equation)

    @abstractmethod
    def _solve(self, equation):
        pass


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

    @abstractmethod
    def _solve(self, equation):
        pass


class RiccatiSolverLRCF(ImmutableObject):
    r"""Interface for solvers computing a LR CF factor of the solution of a |RiccatiEquation|."""

    def solve(self, equation):
        """Solve a |RiccatiEquation|.

        Parameters
        ----------
        equation
            The |RiccatiEquation| to solve.

        Returns
        -------
        Z
            Low-rank Cholesky factor of the solution, |VectorArray| from `equation.A.source`.
        """
        assert isinstance(equation, RiccatiEquation)
        return self._solve(equation)

    @abstractmethod
    def _solve(self, equation):
        pass


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

    @abstractmethod
    def _solve(self, equation):
        pass


class PositiveRiccatiSolverLRCF(ImmutableObject):
    r"""Interface for solvers computing a LR CF of the solution of a |PositiveRiccatiEquation|."""

    def solve(self, equation):
        """Solve a |PositiveRiccatiEquation|.

        Parameters
        ----------
        equation
            The |PositiveRiccatiEquation| to solve.

        Returns
        -------
        Z
            Low-rank Cholesky factor of the solution, |VectorArray| from `equation.A.source`.
        """
        assert isinstance(equation, PositiveRiccatiEquation)
        return self._solve(equation)

    @abstractmethod
    def _solve(self, equation):
        pass


class SylvesterSolver(ImmutableObject):
    r"""Interface for solvers computing a solution of a |SylvesterEquation|."""

    def solve(self, equation):
        r"""Solve a |SylvesterEquation|.

        Parameters
        ----------
        equation
            The |SylvesterEquation| to solve.

        Returns
        -------
        V
            Returned if `equation.B` and `equation.Br` are given, |VectorArray|
            from `equation.A.source`.
        W
            Returned if `equation.C` and `equation.Cr` are given, |VectorArray|
            from `equation.A.source`.
        """
        assert isinstance(equation, SylvesterEquation)
        return self._solve(equation)

    @abstractmethod
    def _solve(self, equation):
        pass
