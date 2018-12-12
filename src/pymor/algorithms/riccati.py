# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.lyapunov import MAT_EQN_SPARSE_MIN_SIZE
from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.operators.interfaces import OperatorInterface

_DEFAULT_RICC_LRCF_SPARSE_SOLVER_BACKEND = 'pymess'

_DEFAULT_RICC_LRCF_DENSE_SOLVER_BACKEND = ('pymess' if config.HAVE_PYMESS else
                                           'slycot' if config.HAVE_SLYCOT else
                                           'scipy')


@defaults('options', 'default_sparse_solver_backend', 'default_dense_solver_backend')
def solve_ricc_lrcf(A, E, B, C, R=None, S=None, trans=False, options=None,
                    default_sparse_solver_backend=_DEFAULT_RICC_LRCF_SPARSE_SOLVER_BACKEND,
                    default_dense_solver_backend=_DEFAULT_RICC_LRCF_DENSE_SOLVER_BACKEND):
    r"""Compute an approximate low-rank solution of a Riccati equation.

    Returns a low-rank Cholesky factor :math:`Z` such that :math:`Z Z^T`
    approximates the solution :math:`X` of a (generalized)
    continuous-time algebraic Riccati equation:

    - if trans is `False`

        .. math::
            A X E^T + E X A^T
            - (E X C^T + S) R^{-1} (E X C^T + S)^T
            + B B^T = 0.

    - if trans is `True`

        .. math::
            A^T X E + E^T X A
            - (E^T X B + S) R^{-1} (E^T X B + S)^T
            + C^T C = 0.

    If E is None, it is taken to be the identity operator, and similarly
    for R.
    If S is None, it is taken to be the zero operator.

    We assume A, E, B, C, R, S are real |Operators|, (E, A, B, C) is
    stabilizable and detectable, R is symmetric positive definite, and

    .. math::
        \begin{bmatrix}
            Q & S \\
            S^T & R
        \end{bmatrix}

    is positive semi-definite, where :math:`Q = B B^T` if trans is
    `False` and :math:`Q = C^T C` if trans is `True` (sufficient
    conditions for existence of a unique positive semi-definite
    stabilizing solution X).
    For large-scale problems, we additionally assume that `B.source.dim`
    and `C.range.dim` are small.

    If the solver is not specified using the options argument, a solver
    backend is chosen based on availability in the following order:

    - for sparse problems (minimum size specified by
      `MAT_EQN_SPARSE_MIN_SIZE`)

        1. `pymess` (see :func:`pymor.bindings.pymess.solve_ricc_lrcf`)

    - for dense problems (smaller than `MAT_EQN_SPARSE_MIN_SIZE`)

        1. `pymess` (see :func:`pymor.bindings.pymess.solve_ricc_lrcf`)
        2. `slycot` (see :func:`pymor.bindings.slycot.solve_ricc_lrcf`)
        3. `scipy` (see :func:`pymor.bindings.scipy.solve_ricc_lrcf`)

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E or `None`.
    B
        The |Operator| B.
    C
        The |Operator| C.
    R
        The |Operator| R or `None`.
    S
        The |Operator| S or `None`.
    trans
        Whether the first |Operator| in the Riccati equation is
        transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.bindings.scipy.ricc_lrcf_solver_options`,
        - :func:`pymor.bindings.slycot.ricc_lrcf_solver_options`,
        - :func:`pymor.bindings.pymess.ricc_lrcf_solver_options`.

    default_sparse_solver_backend
        Default sparse solver backend to use (pymess).
    default_dense_solver_backend
        Default dense solver backend to use (pymess, slycot, scipy).

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
        if A.source.dim >= MAT_EQN_SPARSE_MIN_SIZE:
            backend = default_sparse_solver_backend
        else:
            backend = default_dense_solver_backend
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_ricc_lrcf as solve_ricc_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_ricc_lrcf as solve_ricc_impl
    elif backend == 'pymess':
        from pymor.bindings.pymess import solve_ricc_lrcf as solve_ricc_impl
    else:
        raise ValueError('Unknown solver backend ({}).'.format(backend))
    return solve_ricc_impl(A, E, B, C, R, S, trans=trans, options=options)


_DEFAULT_POS_RICC_LRCF_DENSE_SOLVER_BACKEND = ('pymess' if config.HAVE_PYMESS else
                                               'slycot' if config.HAVE_SLYCOT else
                                               'scipy')


@defaults('options', 'default_dense_solver_backend')
def solve_pos_ricc_lrcf(A, E, B, C, R=None, S=None, trans=False, options=None,
                        default_dense_solver_backend=_DEFAULT_RICC_LRCF_DENSE_SOLVER_BACKEND):
    r"""Compute an approximate low-rank solution of a positive Riccati equation.

    Returns a low-rank Cholesky factor :math:`Z` such that :math:`Z Z^T`
    approximates the solution :math:`X` of a positive (generalized)
    continuous-time algebraic Riccati equation:

    - if trans is `False`

        .. math::
            A X E^T + E X A^T
            + (E X C^T + S) R^{-1} (E X C^T + S)^T
            + B B^T = 0.

    - if trans is `True`

        .. math::
            A^T X E + E^T X A
            + (E^T X B + S) R^{-1} (E^T X B + S)^T
            + C^T C = 0.

    If E is None, it is taken to be the identity operator, and similarly
    for R.
    If S is None, it is taken to be the zero operator.

    If the solver is not specified using the options argument, a solver
    backend is chosen based on availability in the following order:

        1. `pymess` (see
        :func:`pymor.bindings.pymess.solve_pos_ricc_lrcf`)
        2. `slycot` (see
        :func:`pymor.bindings.slycot.solve_pos_ricc_lrcf`)
        3. `scipy` (see
        :func:`pymor.bindings.scipy.solve_pos_ricc_lrcf`)

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E or `None`.
    B
        The |Operator| B.
    C
        The |Operator| C.
    R
        The |Operator| R or `None`.
    S
        The |Operator| S or `None`.
    trans
        Whether the first |Operator| in the positive Riccati equation is
        transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.bindings.scipy.pos_ricc_lrcf_solver_options`,
        - :func:`pymor.bindings.slycot.pos_ricc_lrcf_solver_options`,
        - :func:`pymor.bindings.pymess.pos_ricc_lrcf_solver_options`.

    default_dense_solver_backend
        Default dense solver backend to use (pymess, slycot, scipy).

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
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_pos_ricc_lrcf as solve_ricc_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_pos_ricc_lrcf as solve_ricc_impl
    elif backend == 'pymess':
        from pymor.bindings.pymess import solve_pos_ricc_lrcf as solve_ricc_impl
    else:
        raise ValueError('Unknown solver backend ({}).'.format(backend))
    return solve_ricc_impl(A, E, B, C, R, S, trans=trans, options=options)


def _solve_ricc_check_args(A, E, B, C, R=None, S=None, trans=False):
    assert isinstance(A, OperatorInterface) and A.linear
    assert A.source == A.range
    if E is not None:
        assert isinstance(E, OperatorInterface) and E.linear
        assert E.source == E.range == A.source
    assert isinstance(B, OperatorInterface) and B.linear
    assert B.range == A.source
    assert isinstance(C, OperatorInterface) and C.linear
    assert C.source == A.source
    if not trans:
        if R is not None:
            assert isinstance(R, OperatorInterface) and R.linear
            assert R.source == R.range == C.range
        if S is not None:
            assert isinstance(S, OperatorInterface) and S.linear
            assert S.source == C.range
            assert S.range == A.source
    else:
        if R is not None:
            assert isinstance(R, OperatorInterface) and R.linear
            assert R.source == R.range == B.source
        if S is not None:
            assert isinstance(S, OperatorInterface) and S.linear
            assert S.source == B.source
            assert S.range == A.source
