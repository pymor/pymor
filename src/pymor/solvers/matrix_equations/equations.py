# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.sparse as sps

from pymor.core.base import ImmutableObject
from pymor.operators.constructions import IdentityOperator
from pymor.operators.interface import Operator


class LyapunovEquation(ImmutableObject):
    r"""A (generalized) continuous- or discrete-time Lyapunov equation.

    With :math:`E` taken to be the identity if `None`, for `cont_time` `True`:

    - if `trans` is `False`:

      .. math::
          A X E^T + E X A^T + B B^T = 0,

    - if `trans` is `True`:

      .. math::
          A^T X E + E^T X A + B^T B = 0.

    If `cont_time` is `False`, the discrete-time equation is described:

    - if `trans` is `False`:

      .. math::
          A X A^T - E X E^T + B B^T = 0,

    - if `trans` is `True`:

      .. math::
          A^T X A - E^T X E + B^T B = 0.

    Use :meth:`solve` to obtain the dense solution :math:`X` and :meth:`solve_lr`
    to obtain a low-rank factor :math:`Z` with :math:`X \approx Z Z^H`.

    Parameters
    ----------
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    trans
        Whether the first |Operator| in the equation is transposed.
    cont_time
        If `True`, the continuous-time equation is described, otherwise the
        discrete-time equation.
    name
        Name of the equation.
    """

    def __init__(self, A, E, B, trans=False, cont_time=True, name=None):
        assert isinstance(A, Operator)
        assert A.linear
        assert not A.parametric
        assert A.source == A.range
        if E is not None:
            assert isinstance(E, Operator)
            assert E.linear
            assert not E.parametric
            assert E.source == E.range
            assert E.source == A.source
        assert B in A.source
        self.__auto_init(locals())

    @property
    def dim(self):
        r"""Dimension of the unknown :math:`X`."""
        return self.A.source.dim

    def solve(self, solver=None):
        r"""Compute the dense solution :math:`X` as a |NumPy array|."""
        from pymor.solvers.matrix_equations.default import DefaultLyapunovSolver
        from pymor.solvers.matrix_equations.interface import LyapunovSolver
        solver = DefaultLyapunovSolver() if solver is None else solver
        assert isinstance(solver, LyapunovSolver)
        return solver.solve(self)

    def solve_lr(self, solver=None):
        r"""Compute a low-rank factor :math:`Z` as a |VectorArray| from `A.source`."""
        from pymor.solvers.matrix_equations.default import DefaultLyapunovSolverLR
        from pymor.solvers.matrix_equations.interface import LyapunovSolverLR
        solver = DefaultLyapunovSolverLR() if solver is None else solver
        assert isinstance(solver, LyapunovSolverLR)
        return solver.solve(self)

    def to_matrices(self):
        """Return operators as matrices.

        Returns
        -------
        A
            The |NumPy array| or |SciPy spmatrix| A.
        E
            The |NumPy array| or |SciPy spmatrix| E or `None`.
        B
            The |NumPy array| B.
        """
        from pymor.algorithms.to_matrix import to_matrix
        A = to_matrix(self.A, format='dense')
        E = to_matrix(self.E, format='dense') if self.E is not None else None
        B = self.B.to_numpy()

        return A, E, (B.T if self.trans else B)

    @classmethod
    def from_matrices(cls, A, E, B, trans=False, cont_time=True, name=None):
        r"""Create a |LyapunovEquation| from matrices.

        Parameters
        ----------
        A
            The |NumPy array| or |SciPy spmatrix| A.
        E
            The |NumPy array| or |SciPy spmatrix| E or `None`.
        B
            The |NumPy array| B.
        trans
            Whether the first matrix in the equation is transposed.
        cont_time
            If `True`, the continuous-time equation, otherwise the discrete-time one.
        name
            Name of the equation.
        """
        from pymor.bindings.scipy import sparray
        from pymor.operators.numpy import NumpyMatrixOperator

        assert isinstance(A, np.ndarray | sps.spmatrix | sparray)
        assert isinstance(E, np.ndarray | sps.spmatrix | sparray | type(None))
        assert isinstance(B, np.ndarray | sps.spmatrix | sparray)

        A = NumpyMatrixOperator(A)
        E = NumpyMatrixOperator(E) if E is not None else None
        B = A.source.from_numpy(B if not trans else B.T)

        return cls(A, E, B, trans=trans, cont_time=cont_time, name=name)

class RiccatiData(ImmutableObject):
    """Coefficient storage and validation shared by the two Riccati equations.

    Not intended to be used directly.
    """

    def __init__(self, A, E, B, C, R=None, S=None, trans=False, name=None):
        assert isinstance(A, Operator)
        assert A.linear
        assert not A.parametric
        assert A.source == A.range
        if E is not None:
            assert isinstance(E, Operator)
            assert E.linear
            assert not E.parametric
            assert E.source == E.range == A.source
        assert B in A.source
        assert C in A.source
        if R is not None:
            assert isinstance(R, np.ndarray)
            assert R.ndim == 2
            assert R.shape[0] == R.shape[1]
            assert R.shape[0] == (len(C) if not trans else len(B))
        if S is not None:
            assert S in A.source
            assert len(S) == (len(C) if not trans else len(B))
        self.__auto_init(locals())

    @property
    def dim(self):
        r"""Dimension of the unknown :math:`X`."""
        return self.A.source.dim

    def to_matrices(self):
        """Return operators as matrices.

        Returns
        -------
        A
            The |NumPy array| A.
        E
            The |NumPy array| E or `None`.
        B
            The |NumPy array| B.
        C
            The |NumPy array| C.
        R
            The |NumPy array| R or `None`.
        S
            The |NumPy array| S or `None`.
        """
        from pymor.algorithms.to_matrix import to_matrix
        A = to_matrix(self.A, format='dense')
        E = to_matrix(self.E, format='dense') if self.E is not None else None
        B = self.B.to_numpy()
        C = self.C.to_numpy().T
        S = self.S.to_numpy() if self.S is not None else None
        if S is not None and not self.trans:
            S = S.T
        return A, E, B, C, self.R, S

    @classmethod
    def from_matrices(cls, A, E, B, C, R=None, S=None, trans=False, name=None):
        """Create the |RiccatiEquation| or |PositiveRiccatiEquation| from matrices.

        Parameters
        ----------
        A
            The |NumPy array| or |SciPy spmatrix| A.
        E
            The |NumPy array| or |SciPy spmatrix| E or `None`.
        B
            The |NumPy array| B.
        C
            The |NumPy array| C.
        R
            The |NumPy array| R or `None`.
        S
            The |NumPy array| S or `None`.
        trans
            Whether the first matrix in the equation is transposed.
        name
            Name of the equation.
        """
        from pymor.bindings.scipy import sparray
        from pymor.operators.numpy import NumpyMatrixOperator

        assert isinstance(A, np.ndarray | sps.spmatrix | sparray)
        assert isinstance(E, np.ndarray | sps.spmatrix | sparray | type(None))
        assert isinstance(B, np.ndarray)
        assert isinstance(C, np.ndarray)
        assert isinstance(R, np.ndarray | type(None))
        assert isinstance(S, np.ndarray | type(None))

        A = NumpyMatrixOperator(A)
        E = NumpyMatrixOperator(E) if E is not None else None
        B = A.source.from_numpy(B)
        C = A.source.from_numpy(C.T)
        if S is not None:
            S = A.source.from_numpy(S.T if not trans else S)

        return cls(A, E, B, C, R=R, S=S, trans=trans, name=name)


class RiccatiEquation(RiccatiData):
    r"""A (generalized) continuous-time algebraic Riccati equation.

    With :math:`E` taken to be the identity if `None`, :math:`R` the identity if
    `None`, and :math:`S` zero if `None`:

    - if `trans` is `False`:

      .. math::
          A X E^T + E X A^T
          - (E X C^T + S^T) R^{-1} (C X E^T + S)
          + B B^T = 0,

    - if `trans` is `True`:

      .. math::
          A^T X E + E^T X A
          - (E^T X B + S) R^{-1} (B^T X E + S^T)
          + C^T C = 0.

    Only the continuous-time equation is supported.

    Parameters
    ----------
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    C
        The operator C as a |VectorArray| from `A.source`.
    R
        The matrix R as a 2D |NumPy array| or `None`.
    S
        The operator S as a |VectorArray| from `A.source` or `None`.
    trans
        Whether the first |Operator| in the equation is transposed.
    name
        Name of the equation.
    """

    def solve(self, solver=None):
        r"""Compute the dense solution :math:`X` as a |NumPy array|."""
        from pymor.solvers.matrix_equations.default import DefaultRiccatiSolver
        from pymor.solvers.matrix_equations.interface import RiccatiSolver
        solver = DefaultRiccatiSolver() if solver is None else solver
        assert isinstance(solver, RiccatiSolver)
        return solver.solve(self)

    def solve_lr(self, solver=None):
        r"""Compute a low-rank factor :math:`Z` as a |VectorArray| from `A.source`."""
        from pymor.solvers.matrix_equations.default import DefaultRiccatiSolverLR
        from pymor.solvers.matrix_equations.interface import RiccatiSolverLR
        solver = DefaultRiccatiSolverLR() if solver is None else solver
        assert isinstance(solver, RiccatiSolverLR)
        return solver.solve(self)


class PositiveRiccatiEquation(RiccatiData):
    r"""A (generalized) positive continuous-time algebraic Riccati equation.

    Differs from :class:`RiccatiEquation` only in the sign of the quadratic term:

    - if `trans` is `False`:

      .. math::
          A X E^T + E X A^T
          + (E X C^T + S^T) R^{-1} (C X E^T + S)
          + B B^T = 0,

    - if `trans` is `True`:

      .. math::
          A^T X E + E^T X A
          + (E^T X B + S) R^{-1} (B^T X E + S^T)
          + C^T C = 0.

    Parameters
    ----------
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    C
        The operator C as a |VectorArray| from `A.source`.
    R
        The matrix R as a 2D |NumPy array| or `None`.
    S
        The operator S as a |VectorArray| from `A.source` or `None`.
    trans
        Whether the first |Operator| in the equation is transposed.
    name
        Name of the equation.
    """

    def solve(self, solver=None):
        r"""Compute the dense solution :math:`X` as a |NumPy array|."""
        from pymor.solvers.matrix_equations.default import DefaultPositiveRiccatiSolver
        from pymor.solvers.matrix_equations.interface import PositiveRiccatiSolver
        solver = DefaultPositiveRiccatiSolver() if solver is None else solver
        assert isinstance(solver, PositiveRiccatiSolver)
        return solver.solve(self)

    def solve_lr(self, solver=None):
        r"""Compute a low-rank factor :math:`Z` as a |VectorArray| from `A.source`."""
        from pymor.solvers.matrix_equations.default import DefaultPositiveRiccatiSolverLR
        from pymor.solvers.matrix_equations.interface import PositiveRiccatiSolverLR
        solver = DefaultPositiveRiccatiSolverLR() if solver is None else solver
        assert isinstance(solver, PositiveRiccatiSolverLR)
        return solver.solve(self)

class SylvesterEquation(ImmutableObject):
    r"""A |SylvesterEquation|.

    Defines the |SylvesterEquation|

    .. math::
        A V E_r^T + E V A_r^T + B B_r^T = 0

    or

    .. math::
        A^T W E_r + E^T W A_r + C^T C_r = 0

    or both using (in case B, Br, C and Cr) are given.

    Parameters
    ----------
    A
        Real |Operator|.
    Ar
        Real |Operator|.
    E
        Real |Operator| or `None` (then assumed to be the identity).
    Er
        Real |Operator| or `None` (then assumed to be the identity).
    B
        Real |Operator| or `None`.
    Br
        Real |Operator| or `None`.
    C
        Real |Operator| or `None`.
    Cr
        Real |Operator| or `None`.
    """

    def __init__(self, A, Ar, E=None, Er=None, B=None, Br=None, C=None, Cr=None, name=None):
        assert isinstance(A, Operator)
        assert A.linear
        assert A.source == A.range

        assert isinstance(Ar, Operator)
        assert Ar.linear
        assert Ar.source == Ar.range

        assert E is None or isinstance(E, Operator) and E.linear and E.source == E.range == A.source
        if E is None:
            E = IdentityOperator(A.source)

        assert Er is None or isinstance(Er, Operator) and Er.linear and Er.source == Er.range == Ar.source
        if Er is None:
            Er = IdentityOperator(Ar.source)

        assert B is None or isinstance(B, Operator) and B.linear and B.range == A.source
        assert Br is None or isinstance(Br, Operator) and Br.linear and Br.range == Ar.source
        assert (B is None) == (Br is None)
        assert B is None or B.source == Br.source

        assert C is None or isinstance(C, Operator) and C.linear and C.source == A.source
        assert Cr is None or isinstance(Cr, Operator) and Cr.linear and Cr.source == Ar.source
        assert (C is None) == (Cr is None)
        assert C is None or C.range == Cr.range

        assert not (B is None and C is None)

        self.__auto_init(locals())

    @property
    def dim(self):
        """Dimension of the unknown :math:`V` and :math:`W`."""
        return self.A.source.dim

    def solve(self, solver=None):
        r"""Compute the solution :math:`V` or :math:`W` or both as |VectorArrays|."""
        from pymor.solvers.matrix_equations.default import DefaultSylvesterSolver
        from pymor.solvers.matrix_equations.interface import SylvesterSolver
        solver = DefaultSylvesterSolver() if solver is None else solver
        assert isinstance(solver, SylvesterSolver)
        return solver.solve(self)

    def to_matrices(self):
        """Return operators as matrices.

        Returns
        -------
        A
            The The |NumPy array| or |SciPy spmatrix| A or `None`.
        Ar
            The |NumPy array| or |SciPy spmatrix| Ar or `None`.
        E
            The |NumPy array| or |SciPy spmatrix| E or `None`.
        Er
            The |NumPy array| or |SciPy spmatrix| Er or `None`.
        B
            The |NumPy array| or |SciPy spmatrix| B or `None`.
        Br
            The |NumPy array| or |SciPy spmatrix| Br or `None`.
        C
            The |NumPy array| or |SciPy spmatrix| C or `None`.
        Cr
            The |NumPy array| or |SciPy spmatrix| Cr or `None`.
        """
        from pymor.algorithms.to_matrix import to_matrix
        A = to_matrix(self.A, format='dense')
        Ar = to_matrix(self.Ar, format='dense')
        E = None if isinstance(self.E, IdentityOperator) else to_matrix(self.E, format='dense')
        Er = None if isinstance(self.Er, IdentityOperator) else to_matrix(self.Er, format='dense')
        B = None if self.B is None else to_matrix(self.B, format='dense')
        Br = None if self.Br is None else to_matrix(self.Br, format='dense')
        C = None if self.C is None else to_matrix(self.C, format='dense')
        Cr = None if self.Cr is None else to_matrix(self.Cr, format='dense')

        return A, Ar, E, Er, B, Br, C, Cr

    @classmethod
    def from_matrices(cls, A, Ar, E=None, Er=None, B=None, Br=None, C=None, Cr=None, name=None):
        r"""Create a |SylvesterEquation| from matrices.

        Provide `B` and `Br` for the :math:`V` equation, `C` and `Cr` for the
        :math:`W` equation, or all four for both.

        Parameters
        ----------
        A
            The |NumPy array| or |SciPy spmatrix| A.
        Ar
            The |NumPy array| or |SciPy spmatrix| Ar.
        E
            The |NumPy array| or |SciPy spmatrix| E or `None` (then identity).
        Er
            The |NumPy array| or |SciPy spmatrix| Er or `None` (then identity).
        B
            The |NumPy array| or |SciPy spmatrix| B or `None`.
        Br
            The |NumPy array| or |SciPy spmatrix| Br or `None`.
        C
            The |NumPy array| or |SciPy spmatrix| C `None`.
        Cr
            The |NumPy array| or |SciPy spmatrix| Cr or `None`.
        name
            Name of the equation.
        """
        from pymor.bindings.scipy import sparray
        from pymor.operators.numpy import NumpyMatrixOperator

        assert isinstance(A, np.ndarray | sps.spmatrix | sparray)
        assert isinstance(Ar, np.ndarray | sps.spmatrix | sparray)
        assert E is None or isinstance(E, np.ndarray | sps.spmatrix | sparray)
        assert Er is None or isinstance(Er, np.ndarray | sps.spmatrix | sparray)
        assert B is None or isinstance(B, np.ndarray | sps.spmatrix | sparray)
        assert Br is None or isinstance(Br, np.ndarray | sps.spmatrix | sparray)
        assert C is None or isinstance(C, np.ndarray | sps.spmatrix | sparray)
        assert Cr is None or isinstance(Cr, np.ndarray | sps.spmatrix | sparray)

        A = NumpyMatrixOperator(A)
        Ar = NumpyMatrixOperator(Ar)
        E = NumpyMatrixOperator(E) if E is not None else None
        Er = NumpyMatrixOperator(Er) if Er is not None else None
        B = NumpyMatrixOperator(B) if B is not None else None
        Br = NumpyMatrixOperator(Br) if Br is not None else None
        C = NumpyMatrixOperator(C) if C is not None else None
        Cr = NumpyMatrixOperator(Cr) if Cr is not None else None

        return cls(A, Ar, E=E, Er=Er, B=B, Br=Br, C=C, Cr=Cr, name=name)
