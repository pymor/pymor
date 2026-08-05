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
    r"""Dense solver interface for |LyapunovEquations|."""

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
    """Low-rank solver interface for |LyapunovEquations|.

    Computes a low-rank Cholesky factor of the solution of the |LyapunovEquation|.
    """

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
    r"""Dense solver interface for |RiccatiEquations|."""

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
    r"""Low-rank solver interface for |RiccatiEquations|.

    Computes a low-rank Cholesky factor of the solution of the |RiccatiEquation|.
    """

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
    r"""Dense solver interface for |PositiveRiccatiEquations|."""

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
    r"""Low-rank solver interface for |PositiveRiccatiEquations|.

    Computes a low-rank Cholesky factor of the solution of the |PositiveRiccatiEquation|.
    """

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
    r"""Dense solver interface for |SylvesterEquations|."""

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
