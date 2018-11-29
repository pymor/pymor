# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.core.config import config
from pymor.operators.interfaces import OperatorInterface

MAT_EQN_SPARSE_MIN_SIZE = 1000  # minimal size for which a sparse solver will be used by default

_DEFAULT_LYAP_LRCF_SPARSE_SOLVER_BACKEND = ('pymess' if config.HAVE_PYMESS else
                                            'lradi')

_DEFAULT_LYAP_LRCF_DENSE_SOLVER_BACKEND = ('pymess' if config.HAVE_PYMESS else
                                           'slycot' if config.HAVE_SLYCOT else
                                           'scipy')

_DEFAULT_LYAP_DENSE_SOLVER_BACKEND = ('pymess' if config.HAVE_PYMESS else
                                      'slycot' if config.HAVE_SLYCOT else
                                      'scipy')


def solve_lyap_lrcf(A, E, B, trans=False, options=None,
                    default_sparse_solver_backend=_DEFAULT_LYAP_LRCF_SPARSE_SOLVER_BACKEND,
                    default_dense_solver_backend=_DEFAULT_LYAP_LRCF_DENSE_SOLVER_BACKEND):
    """Compute an approximate low-rank solution of a Lyapunov equation.

    Returns a low-rank Cholesky factor :math:`Z` such that :math:`Z Z^T`
    approximates the solution :math:`X` of a (generalized)
    continuous-time algebraic Lyapunov equation:

    - if trans is `False` and E is `None`:

        .. math::
            A X + X A^T + B B^T = 0,

    - if trans is `False` and E is an |Operator|:

        .. math::
            A X E^T + E X A^T + B B^T = 0,

    - if trans is `True` and E is `None`:

        .. math::
            A^T X + X A + B^T B = 0,

    - if trans is `True` and E is an |Operator|

        .. math::
            A^T X E + E^T X A + B^T B = 0.

    We assume A and E are real |Operators|, E is invertible, and all the
    eigenvalues of (A, E) all lie in the open left half-plane.
    For large-scale problems, we additionally assume `B.source.dim`
    (`B.range.dim`) is small if trans is `False` (`True`).

    If the solver is not specified using the options argument, a solver
    backend is chosen based on availability in the following order:

    - for sparse problems (minimum size specified by
      `MAT_EQN_SPARSE_MIN_SIZE`)

        1. `pymess` (see :func:`pymor.bindings.pymess.solve_lyap_lrcf`)
        2. `lradi` (see :func:`pymor.algorithms.lradi.solve_lyap_lrcf`)

    - for dense problems (smaller than `MAT_EQN_SPARSE_MIN_SIZE`)

        1. `pymess` (see :func:`pymor.bindings.pymess.solve_lyap_lrcf`)
        2. `slycot` (see :func:`pymor.bindings.slycot.solve_lyap_lrcf`)
        3. `scipy` (see :func:`pymor.bindings.scipy.solve_lyap_lrcf`)

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E or `None`.
    B
        The |Operator| B.
    trans
        Whether the first |Operator| in the Lyapunov equation is
        transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.algorithms.lradi.lyap_lrcf_solver_options`,
        - :func:`pymor.bindings.scipy.lyap_lrcf_solver_options`,
        - :func:`pymor.bindings.slycot.lyap_lrcf_solver_options`,
        - :func:`pymor.bindings.pymess.lyap_lrcf_solver_options`.

    default_sparse_solver_backend
        Default sparse solver backend to use (pymess, lradi).
    default_dense_solver_backend
        Default dense solver backend to use (pymess, slycot, scipy).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Lyapunov equation solution,
        |VectorArray| from `A.source`.
    """

    _solve_lyap_check_args(A, E, B, trans)
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        if A.source.dim >= MAT_EQN_SPARSE_MIN_SIZE:
            backend = default_sparse_solver_backend
        else:
            backend = default_dense_solver_backend
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'pymess':
        from pymor.bindings.pymess import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'lradi':
        from pymor.algorithms.lradi import solve_lyap_lrcf as solve_lyap_impl
    else:
        raise ValueError('Unknown solver backend ({}).'.format(backend))
    return solve_lyap_impl(A, E, B, trans=trans, options=options)


def solve_lyap_dense(A, E, B, trans=False, options=None,
                     default_solver_backend=_DEFAULT_LYAP_DENSE_SOLVER_BACKEND):
    """Compute the solution of a Lyapunov equation.

    Returns the solution :math:`X` of a (generalized) continuous-time
    algebraic Lyapunov equation:

    - if trans is `False` and E is `None`:

        .. math::
            A X + X A^T + B B^T = 0,

    - if trans is `False` and E is an |Operator|:

        .. math::
            A X E^T + E X A^T + B B^T = 0,

    - if trans is `True` and E is `None`:

        .. math::
            A^T X + X A + B^T B = 0,

    - if trans is `True` and E is an |Operator|

        .. math::
            A^T X E + E^T X A + B^T B = 0.

    We assume A and E are real |Operators|, E is invertible, and that no
    two eigenvalues of (A, E) sum to zero (i.e., there exists a unique
    solution X).
    Since the solution X is returned as a |NumPy array|, we assume A, E,
    and B can be converted to |NumPy arrays| using
    :func:`~pymor.algorithms.to_matrix.to_matrix`.

    If the solver is not specified using the options argument, a solver
    backend is chosen based on availability in the following order:

    1. `pymess` (see :func:`pymor.bindings.pymess.solve_lyap_dense`)
    2. `slycot` (see :func:`pymor.bindings.slycot.solve_lyap_dense`)
    3. `scipy` (see :func:`pymor.bindings.scipy.solve_lyap_dense`)

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E or `None`.
    B
        The |Operator| B.
    trans
        Whether the first |Operator| in the Lyapunov equation is
        transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.bindings.scipy.lyap_dense_solver_options`,
        - :func:`pymor.bindings.slycot.lyap_dense_solver_options`,
        - :func:`pymor.bindings.pymess.lyap_dense_solver_options`.

    default_solver_backend
        Default solver backend to use (pymess, slycot, scipy).

    Returns
    -------
    X
        Lyapunov equation solution as a |NumPy array|.
    """

    _solve_lyap_check_args(A, E, B, trans)
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        backend = default_solver_backend
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_lyap_dense as solve_lyap_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_lyap_dense as solve_lyap_impl
    elif backend == 'pymess':
        from pymor.bindings.pymess import solve_lyap_dense as solve_lyap_impl
    else:
        raise ValueError('Unknown solver backend ({}).'.format(backend))
    return solve_lyap_impl(A, E, B, trans, options=options)


def _solve_lyap_check_args(A, E, B, trans=False):
    assert isinstance(A, OperatorInterface) and A.linear
    assert A.source == A.range
    assert isinstance(B, OperatorInterface) and B.linear
    assert not trans and B.range == A.source or trans and B.source == A.source
    assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source


def chol(A):
    """Cholesky decomposition.

    This implementation uses SVD to compute the Cholesky factor (can be
    used for singular matrices).

    Parameters
    ----------
    A
        Symmetric positive semidefinite matrix as a |NumPy array|.

    Returns
    -------
    L
        Cholesky factor of A (in the sense that L * L^T approximates A).
    """

    assert isinstance(A, np.ndarray) and A.ndim == 2
    assert A.shape[0] == A.shape[1]

    U, s, _ = spla.svd(A, lapack_driver='gesvd')
    L = U.dot(np.diag(np.sqrt(s)))
    return L
