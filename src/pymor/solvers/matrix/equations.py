# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.base import ImmutableObject
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

    Use :meth:`solve` to obtain the dense solution :math:`X` and :meth:`solve_lrcf`
    to obtain a low-rank Cholesky factor :math:`Z` with :math:`X \approx Z Z^H`.

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
        from pymor.solvers.matrix.default import DefaultLyapunovSolver
        from pymor.solvers.matrix.interface import LyapunovSolver
        solver = DefaultLyapunovSolver() if solver is None else solver
        assert isinstance(solver, LyapunovSolver)
        return solver.solve(self)

    def solve_lrcf(self, solver=None):
        r"""Compute a low-rank Cholesky factor :math:`Z` as a |VectorArray| from `A.source`."""
        from pymor.solvers.matrix.default import DefaultLyapunovSolverLRCF
        from pymor.solvers.matrix.interface import LyapunovSolverLRCF
        solver = DefaultLyapunovSolverLRCF() if solver is None else solver
        assert isinstance(solver, LyapunovSolverLRCF)
        return solver.solve(self)

    def _dense_args(self):
        from pymor.algorithms.to_matrix import to_matrix
        A = to_matrix(self.A, format='dense')
        E = to_matrix(self.E, format='dense') if self.E is not None else None
        B = self.B.to_numpy()

        _check_lyapunov_dense_args(A, E, B.T if self.trans else B, self.trans)

        return A, E, (B.T if self.trans else B)


class RiccatiEquation(ImmutableObject):
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

    Only the continuous-time equation exists.

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
            if not trans:
                assert R.shape[0] == len(C)
            else:
                assert R.shape[0] == len(B)
        if S is not None:
            assert S in A.source
            if not trans:
                assert len(C) == len(S)
            else:
                assert len(B) == len(S)
        self.__auto_init(locals())

    @property
    def dim(self):
        r"""Dimension of the unknown :math:`X`."""
        return self.A.source.dim

    def solve(self, solver=None):
        r"""Compute the dense solution :math:`X` as a |NumPy array|."""
        from pymor.solvers.matrix.default import DefaultRiccatiSolver
        from pymor.solvers.matrix.interface import RiccatiSolver
        solver = DefaultRiccatiSolver() if solver is None else solver
        assert isinstance(solver, RiccatiSolver)
        return solver.solve(self)

    def solve_lrcf(self, solver=None):
        r"""Compute a low-rank Cholesky factor :math:`Z` as a |VectorArray| from `A.source`."""
        from pymor.solvers.matrix.default import DefaultRiccatiSolverLRCF
        from pymor.solvers.matrix.interface import RiccatiSolverLRCF
        solver = DefaultRiccatiSolverLRCF() if solver is None else solver
        assert isinstance(solver, RiccatiSolverLRCF)
        return solver.solve(self)

    def _dense_args(self):
        from pymor.algorithms.to_matrix import to_matrix
        A = to_matrix(self.A, format='dense')
        E = to_matrix(self.E, format='dense') if self.E is not None else None
        B = self.B.to_numpy()
        C = self.C.to_numpy().T
        S = self.S.to_numpy() if self.S is not None else None
        if S is not None and not self.trans:
            S = S.T

        _check_riccati_dense_args(A, E, B, C, self.R, S, trans=self.trans)

        return A, E, B, C, self.R, S


class PositiveRiccatiEquation(ImmutableObject):
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
            if not trans:
                assert R.shape[0] == len(C)
            else:
                assert R.shape[0] == len(B)
        if S is not None:
            assert S in A.source
            if not trans:
                assert len(C) == len(S)
            else:
                assert len(B) == len(S)
        self.__auto_init(locals())

    @property
    def dim(self):
        """Dimension of the unknown :math:`X`."""
        return self.A.source.dim

    def solve(self, solver=None):
        r"""Compute the dense solution :math:`X` as a |NumPy array|."""
        from pymor.solvers.matrix.default import DefaultPositiveRiccatiSolver
        from pymor.solvers.matrix.interface import PositiveRiccatiSolver
        solver = DefaultPositiveRiccatiSolver() if solver is None else solver
        assert isinstance(solver, PositiveRiccatiSolver)
        return solver.solve(self)

    def solve_lrcf(self, solver=None):
        r"""Compute a low-rank Cholesky factor :math:`Z` as a |VectorArray| from `A.source`."""
        from pymor.solvers.matrix.default import DefaultPositiveRiccatiSolverLRCF
        from pymor.solvers.matrix.interface import PositiveRiccatiSolverLRCF
        solver = DefaultPositiveRiccatiSolverLRCF() if solver is None else solver
        assert isinstance(solver, PositiveRiccatiSolverLRCF)
        return solver.solve(self)

    def _dense_args(self):
        from pymor.algorithms.to_matrix import to_matrix
        A = to_matrix(self.A, format='dense')
        E = to_matrix(self.E, format='dense') if self.E is not None else None
        B = self.B.to_numpy()
        C = self.C.to_numpy().T
        S = self.S.to_numpy() if self.S is not None else None
        if S is not None and not self.trans:
            S = S.T

        _check_riccati_dense_args(A, E, B, C, self.R, S, trans=self.trans)

        return A, E, B, C, self.R, S


def _check_lyapunov_dense_args(A, E, B, trans):
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    if E is not None:
        assert isinstance(E, np.ndarray)
        assert E.ndim == 2
        assert E.shape[0] == E.shape[1]
        assert E.shape[0] == A.shape[0]
    assert isinstance(B, np.ndarray)
    assert B.ndim == 2
    assert not trans and B.shape[0] == A.shape[0] or trans and B.shape[1] == A.shape[0]


def _check_riccati_dense_args(A, E, B, C, R, S, trans):
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    if E is not None:
        assert isinstance(E, np.ndarray)
        assert E.ndim == 2
        assert E.shape[0] == E.shape[1]
        assert E.shape[0] == A.shape[0]
    assert isinstance(B, np.ndarray)
    assert isinstance(C, np.ndarray)
    assert B.shape[0] == A.shape[0]
    assert C.shape[1] == A.shape[0]
    if R is not None:
        assert isinstance(R, np.ndarray)
        assert R.ndim == 2
        assert R.shape[0] == R.shape[1]
        if not trans:
            assert R.shape[0] == C.shape[0]
        else:
            assert R.shape[0] == B.shape[1]
    if S is not None:
        assert isinstance(S, np.ndarray)
        if not trans:
            assert S.shape[1] == A.shape[0]
            assert S.shape[0] == C.shape[0]
        else:
            assert S.shape[0] == A.shape[0]
            assert S.shape[1] == B.shape[1]
