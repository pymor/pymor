# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.lyapunov import mat_eqn_sparse_min_size
from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.operators.interface import Operator

_DEFAULT_RICC_LRCF_SPARSE_SOLVER_BACKEND = 'lrradi'

_DEFAULT_RICC_LRCF_DENSE_SOLVER_BACKEND = 'slycot' if config.HAVE_SLYCOT else 'scipy'

_DEFAULT_RICC_DENSE_SOLVER_BACKEND = 'slycot' if config.HAVE_SLYCOT else 'scipy'


@defaults('default_sparse_solver_backend', 'default_dense_solver_backend')
def solve_ricc_lrcf(A, E, B, C, R=None, S=None, trans=False, options=None,
                    default_sparse_solver_backend=_DEFAULT_RICC_LRCF_SPARSE_SOLVER_BACKEND,
                    default_dense_solver_backend=_DEFAULT_RICC_LRCF_DENSE_SOLVER_BACKEND):
    """Compute an approximate low-rank solution of a Riccati equation.

    Returns a low-rank Cholesky factor :math:`Z` such that :math:`Z Z^T`
    approximates the solution :math:`X` of a (generalized)
    continuous-time algebraic Riccati equation:

    - if trans is `False`

      .. math::
          A X E^T + E X A^T
          - (E X C^T + S^T) R^{-1} (C X E^T + S)
          + B B^T = 0.

    - if trans is `True`

      .. math::
          A^T X E + E^T X A
          - (E^T X B + S) R^{-1} (B^T X E + S^T)
          + C^T C = 0.

    If E is None, it is taken to be identity, and similarly for R.
    If S is None, it is taken to be zero.

    We assume:

    - A and E are real |Operators|,
    - B, C and S are real |VectorArrays| from `A.source`,
    - R is a real |NumPy array|,
    - E is nonsingular,
    - (E, A, B, C) is stabilizable and detectable,
    - R is symmetric positive definite, and
    - :math:`B B^T - S^T R^{-1} S` (:math:`C^T C - S R^{-1} S^T`) is
      positive semi-definite if trans is `False` (`True`).

    For large-scale problems, we additionally assume that `len(B)` and
    `len(C)` are small.

    If the solver is not specified using the options argument, a solver
    backend is chosen based on availability in the following order:

    - for sparse problems (minimum size specified by
      :func:`~pymor.algorithms.lyapunov.mat_eqn_sparse_min_size`)

      1. `lrradi` (see :func:`pymor.algorithms.lrradi.solve_ricc_lrcf`),

    - for dense problems (smaller than
      :func:`~pymor.algorithms.lyapunov.mat_eqn_sparse_min_size`)

      1. `slycot` (see :func:`pymor.bindings.slycot.solve_ricc_lrcf`),
      2. `scipy` (see :func:`pymor.bindings.scipy.solve_ricc_lrcf`).
      3. `internal` (see :func:`pymor.algorithms.mat_eqn_solvers.care.solve_ricc_lrcf`).

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
        Whether the first |Operator| in the Riccati equation is
        transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.algorithms.mat_eqn_solver.care.ricc_lrcf_solver_options`,
        - :func:`pymor.bindings.scipy.ricc_lrcf_solver_options`,
        - :func:`pymor.bindings.slycot.ricc_lrcf_solver_options`,
        - :func:`pymor.algorithms.lrradi.ricc_lrcf_solver_options`.

    default_sparse_solver_backend
        Default sparse solver backend to use (lrradi).
    default_dense_solver_backend
        Default dense solver backend to use (slycot, scipy).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Riccati equation solution,
        |VectorArray| from `A.source`.
    """
    _solve_ricc_check_args(A, E, B, C, R, S, trans)
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        if A.source.dim >= mat_eqn_sparse_min_size():
            backend = default_sparse_solver_backend
        else:
            backend = default_dense_solver_backend
    if backend == 'internal':
        from pymor.algorithms.mat_eqn_solvers.care import solve_ricc_lrcf as solve_ricc_impl
    elif backend == 'scipy':
        from pymor.bindings.scipy import solve_ricc_lrcf as solve_ricc_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_ricc_lrcf as solve_ricc_impl
    elif backend == 'lrradi':
        from pymor.algorithms.lrradi import solve_ricc_lrcf as solve_ricc_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    return solve_ricc_impl(A, E, B, C, R, S, trans=trans, options=options)


@defaults('default_solver_backend')
def solve_ricc_dense(A, E, B, C, R=None, S=None, trans=False, options=None,
                     default_solver_backend=_DEFAULT_RICC_DENSE_SOLVER_BACKEND):
    """Compute the solution of a Riccati equation.

    Returns the solution :math:`X` of a (generalized) continuous-time
    algebraic Riccati equation:

    - if trans is `False`

      .. math::
          A X E^T + E X A^T
          - (E X C^T + S^T) R^{-1} (C X E^T + S)
          + B B^T = 0.

    - if trans is `True`

      .. math::
          A^T X E + E^T X A
          - (E^T X B + S) R^{-1} (B^T X E + S^T)
          + C^T C = 0.

    If E is None, it is taken to be identity, and similarly for R.
    If S is None, it is taken to be zero.

    We assume:

    - A, E, B, C, R, S are real |NumPy arrays|,
    - E is nonsingular,
    - (E, A, B, C) is stabilizable and detectable,
    - R is symmetric positive definite, and
    - :math:`B B^T - S^T R^{-1} S` (:math:`C^T C - S R^{-1} S^T`) is
      positive semi-definite if trans is `False` (`True`).

    If the solver is not specified using the options argument, a solver
    backend is chosen based on availability in the following order:

    1. `slycot` (see :func:`pymor.bindings.slycot.solve_ricc_dense`),
    2. `scipy` (see :func:`pymor.bindings.scipy.solve_ricc_dense`),
    3. `internal` (see :func:`pymor.algorithms.mat_eqn_solvers.care.solve_ricc_dense`).

    Parameters
    ----------
    A
        The matrix A as a 2D |NumPy array|.
    E
        The matrix E as a 2D |NumPy array| or `None`.
    B
        The matrix B as a 2D |NumPy array|.
    C
        The matrix C as a 2D |NumPy array|.
    R
        The matrix B as a 2D |NumPy array| or `None`.
    S
        The matrix S as a 2D |NumPy array| or `None`.
    trans
        Whether the first matrix in the Riccati equation is
        transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.bindings.slycot.ricc_dense_solver_options`,
        - :func:`pymor.bindings.scipy.ricc_dense_solver_options`,
        - :func:`pymor.algorithms.mat_eqn_solvers.ricc_dense_solver_options`.

    default_solver_backend
        Default solver backend to use (slycot, scipy, internal).

    Returns
    -------
    X
        Riccati equation solution as a |NumPy array|.
    """
    _solve_ricc_dense_check_args(A, E, B, C, R, S, trans)
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        backend = default_solver_backend
    if backend == 'internal':
        from pymor.algorithms.mat_eqn_solvers.care import solve_ricc_dense as solve_ricc_impl
    elif backend == 'scipy':
        from pymor.bindings.scipy import solve_ricc_dense as solve_ricc_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_ricc_dense as solve_ricc_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    return solve_ricc_impl(A, E, B, C, R, S, trans, options=options)


def _solve_ricc_dense_check_args(A, E, B, C, R, S, trans):
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


@defaults('default_dense_solver_backend')
def solve_pos_ricc_lrcf(A, E, B, C, R=None, S=None, trans=False, options=None,
                        default_dense_solver_backend=_DEFAULT_RICC_LRCF_DENSE_SOLVER_BACKEND):
    """Compute an approximate low-rank solution of a positive Riccati equation.

    Returns a low-rank Cholesky factor :math:`Z` such that :math:`Z Z^T`
    approximates the solution :math:`X` of a (generalized) positive
    continuous-time algebraic Riccati equation:

    - if trans is `False`

      .. math::
          A X E^T + E X A^T
          + (E X C^T + S^T) R^{-1} (C X E^T + S)
          + B B^T = 0.

    - if trans is `True`

      .. math::
          A^T X E + E^T X A
          + (E^T X B + S) R^{-1} (B^T X E + S^T)
          + C^T C = 0.

    If E is None, it is taken to be identity, and similarly for R.
    If S is None, it is taken to be zero.

    If the solver is not specified using the options argument, a solver
    backend is chosen based on availability in the following order:

    1. `slycot` (see :func:`pymor.bindings.slycot.solve_pos_ricc_lrcf`),
    2. `scipy` (see :func:`pymor.bindings.scipy.solve_pos_ricc_lrcf`),
    3. `internal` (see :func:`pymor.algorithms.mat_eqn_solvers.pcare.solve_pos_ricc_lrcf`).

    Currently, only dense solvers are provided.

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
        Whether the first |Operator| in the positive Riccati equation is
        transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.bindings.scipy.ricc_lrcf_solver_options`,
        - :func:`pymor.bindings.slycot.ricc_lrcf_solver_options`,
        - :func:`pymor.algorithm.mat_eqn_solvers.pcare.pos_ricc_lrcf_solver_options`.

    default_dense_solver_backend
        Default dense solver backend to use (slycot, scipy, internal).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the positive Riccati equation
        solution, |VectorArray| from `A.source`.
    """
    _solve_ricc_check_args(A, E, B, C, R, S, trans)
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        backend = default_dense_solver_backend
    if backend == 'internal':
        from pymor.algorithms.mat_eqn_solvers.pcare import solve_pos_ricc_lrcf as solve_ricc_impl
    elif backend == 'scipy':
        from pymor.bindings.scipy import solve_pos_ricc_lrcf as solve_ricc_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_pos_ricc_lrcf as solve_ricc_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    return solve_ricc_impl(A, E, B, C, R, S, trans=trans, options=options)


@defaults('default_solver_backend')
def solve_pos_ricc_dense(A, E, B, C, R=None, S=None, trans=False, options=None,
                     default_solver_backend=_DEFAULT_RICC_DENSE_SOLVER_BACKEND):
    """Compute the solution of a Riccati equation.

    Returns the solution :math:`X` of a (generalized) continuous-time
    algebraic Riccati equation:

    - if trans is `False`

      .. math::
          A X E^T + E X A^T
          + (E X C^T + S^T) R^{-1} (C X E^T + S)
          + B B^T = 0.

    - if trans is `True`

      .. math::
          A^T X E + E^T X A
          + (E^T X B + S) R^{-1} (B^T X E + S^T)
          + C^T C = 0.

    If E is None, it is taken to be identity, and similarly for R.
    If S is None, it is taken to be zero.

    We assume:

    - A, E, B, C, R, S are real |NumPy arrays|,
    - E is nonsingular,
    - (E, A, B, C) is stabilizable and detectable,
    - R is symmetric positive definite, and
    - :math:`B B^T - S^T R^{-1} S` (:math:`C^T C - S R^{-1} S^T`) is
      positive semi-definite if trans is `False` (`True`).

    If the solver is not specified using the options argument, a solver
    backend is chosen based on availability in the following order:

    1. `slycot` (see :func:`pymor.bindings.slycot.solve_pos_ricc_dense`)
    2. `scipy` (see :func:`pymor.bindings.scipy.solve_pos_ricc_dense`)
    2. `internal` (see :func:`pymor.algorithms.mat_eqn_solvers.pcare.solve_pos_ricc_dense`)

    Parameters
    ----------
    A
        The matrix A as a 2D |NumPy array|.
    E
        The matrix E as a 2D |NumPy array| or `None`.
    B
        The matrix B as a 2D |NumPy array|.
    C
        The matrix C as a 2D |NumPy array|.
    R
        The matrix B as a 2D |NumPy array| or `None`.
    S
        The matrix S as a 2D |NumPy array| or `None`.
    trans
        Whether the first matrix in the Riccati equation is
        transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.bindings.slycot.ricc_dense_solver_options`,
        - :func:`pymor.bindings.scipy.ricc_dense_solver_options`,
        - :func:`pymor.algorithms.mat_eqn_solvers.pcare.pos_ricc_dense_solver_options`.

    default_solver_backend
        Default solver backend to use (slycot, scipy, internal).

    Returns
    -------
    X
        Riccati equation solution as a |NumPy array|.
    """
    _solve_ricc_dense_check_args(A, E, B, C, R, S, trans)
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        backend = default_solver_backend
    if backend == 'internal':
        from pymor.algorithms.mat_eqn_solvers.pcare import solve_pos_ricc_dense as solve_ricc_impl
    elif backend == 'scipy':
        from pymor.bindings.scipy import solve_pos_ricc_dense as solve_ricc_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_pos_ricc_dense as solve_ricc_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    return solve_ricc_impl(A, E, B, C, R, S, trans, options=options)


def _solve_ricc_check_args(A, E, B, C, R, S, trans):
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
