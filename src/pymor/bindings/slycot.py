# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
config.require('SLYCOT')


import numpy as np
import scipy.linalg as spla
import slycot

from pymor.algorithms.genericsolvers import _parse_options
from pymor.algorithms.lyapunov import _solve_lyap_lrcf_check_args, _solve_lyap_dense_check_args, _chol
from pymor.algorithms.riccati import _solve_ricc_dense_check_args
from pymor.algorithms.to_matrix import to_matrix
from pymor.bindings.scipy import _solve_ricc_check_args
from pymor.core.logger import getLogger


def lyap_lrcf_solver_options():
    """Return available Lyapunov solvers with default options for the slycot backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'slycot_bartels-stewart': {'type': 'slycot_bartels-stewart'}}


def solve_lyap_lrcf(A, E, B, trans=False, cont_time=True, options=None):
    """Compute an approximate low-rank solution of a Lyapunov equation.

    See

    - :func:`pymor.algorithms.lyapunov.solve_cont_lyap_lrcf`
    - :func:`pymor.algorithms.lyapunov.solve_disc_lyap_lrcf`

    for a general description.

    This function uses `slycot.sb03md` (if `E is None`) and `slycot.sg03ad` (if `E is not None`),
    which are dense solvers based on the Bartels-Stewart algorithm. Therefore, we assume A and E can
    be converted to |NumPy arrays| using :func:`~pymor.algorithms.to_matrix.to_matrix` and that
    `B.to_numpy` is implemented.

    Parameters
    ----------
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    trans
        Whether the first |Operator| in the Lyapunov equation is transposed.
    cont_time
        Whether the continuous- or discrete-time Lyapunov equation is solved.
    options
        The solver options to use (see :func:`lyap_lrcf_solver_options`).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Lyapunov equation solution, |VectorArray| from `A.source`.
    """
    _solve_lyap_lrcf_check_args(A, E, B, trans)
    options = _parse_options(options, lyap_lrcf_solver_options(), 'slycot_bartels-stewart', None, False)

    if options['type'] == 'slycot_bartels-stewart':
        X = solve_lyap_dense(to_matrix(A, format='dense'),
                             to_matrix(E, format='dense') if E else None,
                             B.to_numpy().T if not trans else B.to_numpy(),
                             trans=trans, cont_time=cont_time, options=options)
        Z = _chol(X)
    else:
        raise ValueError(f"Unexpected Lyapunov equation solver ({options['type']}).")

    return A.source.from_numpy(Z.T)


def lyap_dense_solver_options():
    """Return available Lyapunov solvers with default options for the slycot backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'slycot_bartels-stewart': {'type': 'slycot_bartels-stewart'}}


def solve_lyap_dense(A, E, B, trans=False, cont_time=True, options=None):
    """Compute the solution of a Lyapunov equation.

    See

    - :func:`pymor.algorithms.lyapunov.solve_cont_lyap_dense`
    - :func:`pymor.algorithms.lyapunov.solve_disc_lyap_dense`

    for a general description.

    This function uses `slycot.sb03md` (if `E is None`) and `slycot.sg03ad` (if `E is not None`),
    which are based on the Bartels-Stewart algorithm.

    Parameters
    ----------
    A
        The matrix A as a 2D |NumPy array|.
    E
        The matrix E as a 2D |NumPy array| or `None`.
    B
        The matrix B as a 2D |NumPy array|.
    trans
        Whether the first matrix in the Lyapunov equation is transposed.
    cont_time
        Whether the continuous- or discrete-time Lyapunov equation is solved.
    options
        The solver options to use (see :func:`lyap_dense_solver_options`).

    Returns
    -------
    X
        Lyapunov equation solution as a |NumPy array|.
    """
    _solve_lyap_dense_check_args(A, E, B, trans)
    options = _parse_options(options, lyap_dense_solver_options(), 'slycot_bartels-stewart', None, False)

    if options['type'] == 'slycot_bartels-stewart':
        n = A.shape[0]
        C = -B.dot(B.T) if not trans else -B.T.dot(B)
        trana = 'T' if not trans else 'N'
        dico = 'C' if cont_time else 'D'
        job = 'B'
        if E is None:
            ldwork = max(2*n*n, 3*n) if cont_time else 2*n*n+2*n
            # slycot v. 0.4.0 does not set ldwork correctly for dico='D'
            # should be fixed in the next release
            U = np.zeros((n, n))
            X, scale, sep, ferr, _ = slycot.sb03md(n, C, A, U, dico, job=job, trana=trana, ldwork=ldwork)
            _solve_check(A.dtype, 'slycot.sb03md', sep, ferr)
        else:
            fact = 'N'
            uplo = 'L'
            Q = np.zeros((n, n))
            Z = np.zeros((n, n))
            _, _, _, _, X, scale, sep, ferr, _, _, _ = slycot.sg03ad(dico, job, fact, trana, uplo,
                                                                     n, A, E,
                                                                     Q, Z, C)
            _solve_check(A.dtype, 'slycot.sg03ad', sep, ferr)
        X /= scale
    else:
        raise ValueError(f"Unexpected Lyapunov equation solver ({options['type']}).")

    return X


def solve_ricc_dense(A, E, B, C, R=None, trans=False, options=None):
    """Compute the solution of a Riccati equation.

    See :func:`pymor.algorithms.riccati.solve_ricc_dense` for a
    general description.

    This function uses `slycot.sb02md` (if `E is None`) which is based on
    the Schur vector approach and `slycot.sg02ad` (if `E is not None`) which
    is based on the method of deflating subspaces.

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
        The matrix R as a 2D |NumPy array| or `None`.
    trans
        Whether the first matrix in the Riccati equation is
        transposed.
    options
        The solver options to use (see
        :func:`ricc_dense_solver_options`).

    Returns
    -------
    X
        Riccati equation solution as a |NumPy array|.
    """
    _solve_ricc_dense_check_args(A, E, B, C, R, trans)
    options = _parse_options(options, ricc_dense_solver_options(), 'slycot', None, False)

    if options['type'] != 'slycot':
        raise ValueError(f"Unexpected Riccati equation solver ({options['type']}).")

    dico = 'C'
    n = A.shape[0]
    if E is not None:
        jobb = 'B'
        fact = 'C'
        uplo = 'U'
        jobl = 'Z'
        scal = 'N'
        sort = 'S'
        acc = 'R'
        m = C.shape[0] if not trans else B.shape[1]
        p = B.shape[1] if not trans else C.shape[0]
        if R is None:
            R = np.eye(m)
        S = np.empty((n, m))
        if not trans:
            A = A.T
            E = E.T
            B, C = C.T, B.T
        out = slycot.sg02ad(dico, jobb, fact, uplo, jobl, scal, sort, acc,
                            n, m, p,
                            A, E, B, C, R, S)
        X = out[1]
        rcond = out[0]
    else:
        if trans:
            if R is None:
                G = B @ B.T
            else:
                G = B @ spla.solve(R, B.T)
            Q = C.T @ C
            X, rcond = slycot.sb02md(n, A, G, Q, dico)[:2]
        else:
            if R is None:
                G = C.T @ C
            else:
                G = C.T @ spla.solve(R, C)
            Q = B @ B.T
            X, rcond = slycot.sb02md(n, A.T, G, Q, dico)[:2]
    _ricc_rcond_check('slycot.sb02md', rcond)

    return X


def ricc_dense_solver_options():
    """Return available Riccati solvers with default options for the slycot backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'slycot': {'type': 'slycot'}}


def _solve_check(dtype, solver, sep, ferr):
    if ferr > 1e-1:
        logger = getLogger(solver)
        logger.warning(f'Estimated forward relative error bound is large (ferr={ferr:e}, sep={sep:e}). '
                       f'Result may not be accurate.')


def ricc_lrcf_solver_options():
    """Return available Riccati solvers with default options for the slycot backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'slycot': {'type': 'slycot'}}


def solve_ricc_lrcf(A, E, B, C, R=None, trans=False, options=None):
    """Compute an approximate low-rank solution of a Riccati equation.

    See :func:`pymor.algorithms.riccati.solve_ricc_lrcf` for a
    general description.

    This function uses `slycot.sb02md` (if E is `None`) or
    `slycot.sg03ad` (if E is not `None`), which are dense solvers.
    Therefore, we assume all |Operators| and |VectorArrays| can be
    converted to |NumPy arrays| using
    :func:`~pymor.algorithms.to_matrix.to_matrix` and
    :func:`~pymor.vectorarrays.interface.VectorArray.to_numpy`.

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
    trans
        Whether the first |Operator| in the Riccati equation is
        transposed.
    options
        The solver options to use (see
        :func:`ricc_lrcf_solver_options`).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Riccati equation solution,
        |VectorArray| from `A.source`.
    """
    _solve_ricc_check_args(A, E, B, C, R, trans)
    options = _parse_options(options, ricc_lrcf_solver_options(), 'slycot', None, False)
    if options['type'] != 'slycot':
        raise ValueError(f"Unexpected Riccati equation solver ({options['type']}).")

    A_source = A.source
    A = to_matrix(A, format='dense')
    E = to_matrix(E, format='dense') if E else None
    B = B.to_numpy().T
    C = C.to_numpy()

    X = solve_ricc_dense(A, E, B, C, R, trans, options)

    return A_source.from_numpy(_chol(X).T)


def _ricc_rcond_check(solver, rcond):
    if rcond < np.finfo(np.float64).eps:
        logger = getLogger(solver)
        logger.warning(f'Estimated reciprocal condition number is small (rcond={rcond:e}). '
                       f'Result may not be accurate.')


def pos_ricc_lrcf_solver_options():
    """Return available positive Riccati solvers with default options for the slycot backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'slycot': {'type': 'slycot'}}


def solve_pos_ricc_lrcf(A, E, B, C, R=None, trans=False, options=None):
    """Compute an approximate low-rank solution of a positive Riccati equation.

    See :func:`pymor.algorithms.riccati.solve_pos_ricc_lrcf` for a
    general description.

    This function uses `slycot.sb02md` (if E is `None`) or
    `slycot.sg03ad` (if E is not `None`), which are dense solvers.
    Therefore, we assume all |Operators| and |VectorArrays| can be
    converted to |NumPy arrays| using
    :func:`~pymor.algorithms.to_matrix.to_matrix` and
    :func:`~pymor.vectorarrays.interface.VectorArray.to_numpy`.

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
    trans
        Whether the first |Operator| in the positive Riccati
        equation is transposed.
    options
        The solver options to use (see
        :func:`pos_ricc_lrcf_solver_options`).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the positive Riccati equation
        solution, |VectorArray| from `A.source`.
    """
    _solve_ricc_check_args(A, E, B, C, R, trans)
    options = _parse_options(options, pos_ricc_lrcf_solver_options(), 'slycot', None, False)
    if options['type'] != 'slycot':
        raise ValueError(f"Unexpected positive Riccati equation solver ({options['type']}).")

    if R is None:
        R = np.eye(len(C) if not trans else len(B))
    return solve_ricc_lrcf(A, E, B, C, -R, trans, options)
