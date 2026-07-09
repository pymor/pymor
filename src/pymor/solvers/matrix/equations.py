# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import ImmutableObject
from pymor.solvers.matrix.utils import _check_lyapunov_args, _check_riccati_args


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
        _check_lyapunov_args(A, E, B, trans)
        self.__auto_init(locals())

    @property
    def dim(self):
        r"""Dimension of the unknown :math:`X`."""
        return self.A.source.dim

    def solve(self, solver=None):
        r"""Compute the dense solution :math:`X` as a |NumPy array|."""
        from pymor.solvers.matrix.interface import LyapunovSolver
        solver = LyapunovSolver() if solver is None else solver
        assert isinstance(solver, LyapunovSolver)
        return solver.solve(self)

    def solve_lrcf(self, solver=None):
        r"""Compute a low-rank Cholesky factor :math:`Z` as a |VectorArray| from `A.source`."""
        from pymor.solvers.matrix.interface import LyapunovSolverLRCF
        solver = LyapunovSolverLRCF() if solver is None else solver
        assert isinstance(solver, LyapunovSolverLRCF)
        return solver.solve(self)

    def _dense_args(self):
        return _dense_lyapunov_args(self)


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
        The matrix R as a 2D |NumPy array| or `None`.  Not an |Operator|: it is
        :math:`m \times m` or :math:`p \times p`, dense, and passed straight to LAPACK.
    S
        The operator S as a |VectorArray| from `A.source` or `None`.
    trans
        Whether the first |Operator| in the equation is transposed.
    name
        Name of the equation.
    """

    def __init__(self, A, E, B, C, R=None, S=None, trans=False, name=None):
        _check_riccati_args(A, E, B, C, R, S, trans)
        self.__auto_init(locals())

    @property
    def dim(self):
        r"""Dimension of the unknown :math:`X`."""
        return self.A.source.dim

    def solve(self, solver=None):
        r"""Compute the dense solution :math:`X` as a |NumPy array|."""
        from pymor.solvers.matrix.interface import RiccatiSolver
        solver = RiccatiSolver() if solver is None else solver
        assert isinstance(solver, RiccatiSolver)
        return solver.solve(self)

    def solve_lrcf(self, solver=None):
        r"""Compute a low-rank Cholesky factor :math:`Z` as a |VectorArray| from `A.source`."""
        from pymor.solvers.matrix.interface import RiccatiSolverLRCF
        solver = RiccatiSolverLRCF() if solver is None else solver
        assert isinstance(solver, RiccatiSolverLRCF)
        return solver.solve(self)

    def _dense_args(self):
        return _dense_riccati_args(self)


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
        _check_riccati_args(A, E, B, C, R, S, trans)
        self.__auto_init(locals())

    @property
    def dim(self):
        """Dimension of the unknown :math:`X`."""
        return self.A.source.dim

    def solve(self, solver=None):
        r"""Compute the dense solution :math:`X` as a |NumPy array|."""
        from pymor.solvers.matrix.interface import PositiveRiccatiSolver
        solver = PositiveRiccatiSolver() if solver is None else solver
        assert isinstance(solver, PositiveRiccatiSolver)
        return solver.solve(self)

    def solve_lrcf(self, solver=None):
        r"""Compute a low-rank Cholesky factor :math:`Z` as a |VectorArray| from `A.source`."""
        from pymor.solvers.matrix.interface import PositiveRiccatiSolverLRCF
        solver = PositiveRiccatiSolverLRCF() if solver is None else solver
        assert isinstance(solver, PositiveRiccatiSolverLRCF)
        return solver.solve(self)

    def _dense_args(self):
        return _dense_riccati_args(self)


def _dense_lyapunov_args(equation):
    """Materialize a |LyapunovEquation| for the dense backends."""
    from pymor.algorithms.to_matrix import to_matrix
    A = to_matrix(equation.A, format='dense')
    E = to_matrix(equation.E, format='dense') if equation.E is not None else None
    B = equation.B.to_numpy()
    return A, E, (B.T if equation.trans else B)


def _dense_riccati_args(equation):
    """Materialize a |RiccatiEquation| / |PositiveRiccatiEquation| for the dense backends."""
    from pymor.algorithms.to_matrix import to_matrix
    A = to_matrix(equation.A, format='dense')
    E = to_matrix(equation.E, format='dense') if equation.E is not None else None
    B = equation.B.to_numpy()
    C = equation.C.to_numpy().T
    S = equation.S.to_numpy() if equation.S is not None else None
    if S is not None and not equation.trans:
        S = S.T

    return A, E, B, C, equation.R, S
