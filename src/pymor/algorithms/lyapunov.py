# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.svd_va import qr_svd
from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.constructions import IdentityOperator
from pymor.operators.interface import Operator
from pymor.tools.frozendict import FrozenDict
from pymor.vectorarrays.constructions import cat_arrays
from pymor.vectorarrays.interface import VectorArray

_DEFAULT_LYAP_SOLVER_BACKEND = FrozenDict(
    {
        'cont': FrozenDict(
            {
                'sparse': 'lradi',
                'dense': 'slycot' if config.HAVE_SLYCOT else 'scipy',
            }
        ),
        'disc': FrozenDict({'dense': 'slycot' if config.HAVE_SLYCOT else 'scipy'}),
    }
)


@defaults('value')
def mat_eqn_sparse_min_size(value=1000):
    """Returns minimal size for which a sparse solver will be used by default."""
    return value


@defaults('default_sparse_solver_backend', 'default_dense_solver_backend')
def solve_cont_lyap_lrcf(A, E, B, trans=False, options=None,
                         default_sparse_solver_backend=_DEFAULT_LYAP_SOLVER_BACKEND['cont']['sparse'],
                         default_dense_solver_backend=_DEFAULT_LYAP_SOLVER_BACKEND['cont']['dense']):
    """Compute an approximate low-rank solution of a continuous-time Lyapunov equation.

    Returns a low-rank Cholesky factor :math:`Z` such that :math:`Z Z^T` approximates the solution
    :math:`X` of a (generalized) continuous-time algebraic Lyapunov equation:

    - if trans is `False` and E is `None`:

      .. math::
         A X + X A^T + B B^T = 0,

    - if trans is `False` and E is an |Operator|:

      .. math::
          A X E^T + E X A^T + B B^T = 0,

    - if trans is `True` and E is `None`:

      .. math::
          A^T X + X A + B^T B = 0,

    - if trans is `True` and E is an |Operator|:

      .. math::
          A^T X E + E^T X A + B^T B = 0.

    We assume A and E are real |Operators|, E is invertible, and all the eigenvalues of (A, E) all
    lie in the open left half-plane. Operator B needs to be given as a |VectorArray| from
    `A.source`, and for large-scale problems, we assume `len(B)` is small.

    If the solver is not specified using the options argument, a solver backend is chosen based on
    availability in the following order:

    - for sparse problems (minimum size specified by
      :func:`mat_eqn_sparse_min_size`)

      1. `lradi` (see :func:`pymor.algorithms.lradi.solve_lyap_lrcf`),

    - for dense problems (smaller than :func:`mat_eqn_sparse_min_size`)

      1. `slycot` (see :func:`pymor.bindings.slycot.solve_lyap_lrcf`),
      2. `scipy` (see :func:`pymor.bindings.scipy.solve_lyap_lrcf`).

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
    options
        The solver options to use.
        See:

        - :func:`pymor.algorithms.lradi.lyap_lrcf_solver_options`,
        - :func:`pymor.bindings.scipy.lyap_lrcf_solver_options`,
        - :func:`pymor.bindings.slycot.lyap_lrcf_solver_options`,

    default_sparse_solver_backend
        Default sparse solver backend to use (lradi).
    default_dense_solver_backend
        Default dense solver backend to use (slycot, scipy).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Lyapunov equation solution, |VectorArray| from `A.source`.
    """
    _solve_lyap_lrcf_check_args(A, E, B, trans)
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        if A.source.dim >= mat_eqn_sparse_min_size():
            backend = default_sparse_solver_backend
        else:
            backend = default_dense_solver_backend
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'lradi':
        from pymor.algorithms.lradi import solve_lyap_lrcf as solve_lyap_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    return solve_lyap_impl(A, E, B, trans=trans, cont_time=True, options=options)


@defaults('default_dense_solver_backend')
def solve_disc_lyap_lrcf(A, E, B, trans=False, options=None,
                         default_dense_solver_backend=_DEFAULT_LYAP_SOLVER_BACKEND['disc']['dense']):
    """Compute an approximate low-rank solution of a discrete-time Lyapunov equation.

    Returns a low-rank Cholesky factor :math:`Z` such that :math:`Z Z^T` approximates the solution
    :math:`X` of a (generalized) discrete-time algebraic Lyapunov equation:

    - if trans is `False` and E is `None`:

      .. math::
         A X A^T - X + B B^T = 0,

    - if trans is `False` and E is an |Operator|:

      .. math::
          A X A^T - E X E^T + B B^T = 0,

    - if trans is `True` and E is `None`:

      .. math::
          A^T X A - X + B^T B = 0,

    - if trans is `True` and E is an |Operator|:

      .. math::
          A^T X A - E^T X E + B^T B = 0.

    We assume A and E are real |Operators|, E is invertible, and all the eigenvalues of (A, E) all
    lie inside the unit circle. Operator B needs to be given as a |VectorArray| from `A.source`, and
    for large-scale problems, we assume `len(B)` is small.

    If the solver is not specified using the options argument, a solver backend is chosen based on
    availability in the following order:

      1. `slycot` (see :func:`pymor.bindings.slycot.solve_lyap_lrcf`),
      2. `scipy` (see :func:`pymor.bindings.scipy.solve_lyap_lrcf`).

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
    options
        The solver options to use.
        See:

        - :func:`pymor.bindings.scipy.lyap_lrcf_solver_options`,
        - :func:`pymor.bindings.slycot.lyap_lrcf_solver_options`.

    default_dense_solver_backend
        Default dense solver backend to use (slycot, scipy).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Lyapunov equation solution, |VectorArray| from `A.source`.
    """
    _solve_lyap_lrcf_check_args(A, E, B, trans)
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        backend = default_dense_solver_backend
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_lyap_lrcf as solve_lyap_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_lyap_lrcf as solve_lyap_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    return solve_lyap_impl(A, E, B, trans=trans, cont_time=False, options=options)


def _solve_lyap_lrcf_check_args(A, E, B, trans):
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


@defaults('default_solver_backend')
def solve_cont_lyap_dense(A, E, B, trans=False, options=None,
                          default_solver_backend=_DEFAULT_LYAP_SOLVER_BACKEND['cont']['dense']):
    """Compute the solution of a continuous-time Lyapunov equation.

    Returns the solution :math:`X` of a (generalized) continuous-time algebraic Lyapunov equation:

    - if trans is `False` and E is `None`:

      .. math::
          A X + X A^T + B B^T = 0,

    - if trans is `False` and E is a |NumPy array|:

      .. math::
          A X E^T + E X A^T + B B^T = 0,

    - if trans is `True` and E is `None`:

      .. math::
          A^T X + X A + B^T B = 0,

    - if trans is `True` and E is a |NumPy array|:

      .. math::
          A^T X E + E^T X A + B^T B = 0.

    We assume A and E are real |NumPy arrays|, E is invertible, and that no two eigenvalues of
    (A, E) sum to zero (i.e., there exists a unique solution X).

    If the solver is not specified using the options argument, a solver backend is chosen based on
    availability in the following order:

    1. `slycot` (see :func:`pymor.bindings.slycot.solve_lyap_dense`)
    2. `scipy` (see :func:`pymor.bindings.scipy.solve_lyap_dense`)

    Parameters
    ----------
    A
        The matrix A as a 2D |NumPy array|.
    E
        The matrix E as a 2D |NumPy array| or `None`.
    B
        The matrix B as a 2D |NumPy array|.
    trans
        Whether the first operator in the Lyapunov equation is transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.bindings.scipy.lyap_dense_solver_options`,
        - :func:`pymor.bindings.slycot.lyap_dense_solver_options`,

    default_solver_backend
        Default solver backend to use (slycot, scipy).

    Returns
    -------
    X
        Lyapunov equation solution as a |NumPy array|.
    """
    _solve_lyap_dense_check_args(A, E, B, trans)
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        backend = default_solver_backend
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_lyap_dense as solve_lyap_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_lyap_dense as solve_lyap_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    return solve_lyap_impl(A, E, B, trans=trans, cont_time=True, options=options)


@defaults('default_solver_backend')
def solve_disc_lyap_dense(A, E, B, trans=False, options=None,
                          default_solver_backend=_DEFAULT_LYAP_SOLVER_BACKEND['disc']['dense']):
    """Compute the solution of a discrete-time Lyapunov equation.

    Returns the solution :math:`X` of a (generalized) continuous-time algebraic Lyapunov equation:

    - if trans is `False` and E is `None`:

      .. math::
         A X A^T - X + B B^T = 0,

    - if trans is `False` and E is a |NumPy array|:

      .. math::
          A X A^T - E X E^T + B B^T = 0,

    - if trans is `True` and E is `None`:

      .. math::
          A^T X A - X + B^T B = 0,

    - if trans is `True` and E is an |NumPy array|:

      .. math::
          A^T X A - E^T X E + B^T B = 0.

    We assume A and E are real |NumPy arrays|, E is invertible, and that all pairwise products of
    two eigenvalues of (A, E) are not equal to one (i.e., there exists a unique solution X).

    If the solver is not specified using the options argument, a solver backend is chosen based on
    availability in the following order:

    1. `slycot` (see :func:`pymor.bindings.slycot.solve_lyap_dense`)
    2. `scipy` (see :func:`pymor.bindings.scipy.solve_lyap_dense`)

    Parameters
    ----------
    A
        The matrix A as a 2D |NumPy array|.
    E
        The matrix E as a 2D |NumPy array| or `None`.
    B
        The matrix B as a 2D |NumPy array|.
    trans
        Whether the first operator in the Lyapunov equation is
        transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.bindings.scipy.lyap_dense_solver_options`,
        - :func:`pymor.bindings.slycot.lyap_dense_solver_options`.

    default_solver_backend
        Default solver backend to use (slycot, scipy).

    Returns
    -------
    X
        Lyapunov equation solution as a |NumPy array|.
    """
    _solve_lyap_dense_check_args(A, E, B, trans)
    if options:
        solver = options if isinstance(options, str) else options['type']
        backend = solver.split('_')[0]
    else:
        backend = default_solver_backend
    if backend == 'scipy':
        from pymor.bindings.scipy import solve_lyap_dense as solve_lyap_impl
    elif backend == 'slycot':
        from pymor.bindings.slycot import solve_lyap_dense as solve_lyap_impl
    else:
        raise ValueError(f'Unknown solver backend ({backend}).')
    return solve_lyap_impl(A, E, B, trans=trans, cont_time=False, options=options)


def _solve_lyap_dense_check_args(A, E, B, trans):
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    if E is not None:
        assert isinstance(E, np.ndarray)
        assert E.ndim == 2
        assert E.shape[0] == E.shape[1]
        assert E.shape[0] == A.shape[0]
    assert isinstance(B, np.ndarray)
    assert A.ndim == 2
    assert not trans and B.shape[0] == A.shape[0] or trans and B.shape[1] == A.shape[0]


def solve_bilinear_lyap_lrcf(A, E, N, B, trans=False, maxit=2, tol=1e-10):
    r"""Solve a bilinear Lyapunov equation.

    Computes a low-rank approximation to the solution.

    The equation is given by

    .. math::
        A X E^T
        + E X A^T
        + \sum_{k = 1}^m N_i X N_i^T
        + B B^T
        = 0

    if `trans` is `False` or

    .. math::
        A^T X E
        + E^T X A
        + \sum_{k = 1}^m N_i^T X N_i
        + B^T B
        = 0

    if `trans` is `True`.

    Parameters
    ----------
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    N
        The tuple of non-parametric |Operators| N_i.
    B
        The operator B as a |VectorArray| from `A.source`.
    trans
        Whether the first |Operator| in the Lyapunov equation is transposed.
    maxit
        Maximum number of iterations.
    tol
        Relative error tolerance.

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Lyapunov equation solution,
        |VectorArray| from `A.source`.
    """
    assert isinstance(A, Operator)
    assert A.source == A.range

    if E is None:
        E = IdentityOperator(A.source)
    assert isinstance(E, Operator)
    assert E.source == E.range
    assert E.source == A.source

    assert isinstance(N, tuple)
    assert all(isinstance(Ni, Operator) for Ni in N)
    assert all(Ni.source == Ni.range for Ni in N)
    assert all(Ni.source == A.source for Ni in N)

    assert isinstance(B, VectorArray)
    assert B in A.source

    assert maxit > 0
    assert tol >= 0

    logger = getLogger('pymor.algorithms.lyapunov.solve_bilinear_lyap_lrcf')
    B_updated = B
    for i in range(maxit):
        with logger.block(f'Iteration {i + 1}'):
            logger.info('Solving Lyapunov equation ...')
            Z = solve_cont_lyap_lrcf(A, E, B_updated, trans=trans)
            logger.info('Compressing low-rank factor ...')
            Z = _compress(Z)
            if tol > 0:
                logger.info('Computing error ...')
                error = _compute_bilinear_lyap_lrcf_error(A, E, N, B, Z, trans)
                logger.info(f'Error: {error:.3e}')
                if error <= tol:
                    break
            if i < maxit - 1:
                if not trans:
                    B_updated = cat_arrays((B,) + tuple(Ni.apply(Z) for Ni in N))
                else:
                    B_updated = cat_arrays((B,) + tuple(Ni.apply_adjoint(Z) for Ni in N))
                logger.info('Compressing B ...')
                B_updated = _compress(B_updated)
    return Z


def _compute_bilinear_lyap_lrcf_error(A, E, H, N, B, Z, trans):
    if not trans:
        U = cat_arrays((A.apply(Z), E.apply(Z), B) + tuple(Ni.apply(Z) for Ni in N))
    else:
        U = cat_arrays((A.apply_adjoint(Z), E.apply_adjoint(Z), B) + tuple(Ni.apply_adjoint(Z) for Ni in N))
    U, R = gram_schmidt(U, return_R=True)
    R2 = R.copy()
    tmp = R[:, : len(Z)].copy()
    R[:, : len(Z)] = R[:, len(Z) : 2 * len(Z)]
    R[:, len(Z) : 2 * len(Z)] = tmp
    error = spla.norm(R @ R2.T)
    B_norm = spla.norm(B.inner(B))
    return error / B_norm


def solve_bilinear_lyap_dense(A, E, N, B, trans=False, maxit=2, tol=1e-10):
    r"""Solve a quadratic-bilinear controllability Lyapunov equation.

    Returns the solution as a dense matrix.

    The equation is given by

    .. math::
        A X E^T
        + E X A^T
        + \sum_{k = 1}^m N_i X N_i^T
        + B B^T
        = 0

    if `trans` is `False` or

    .. math::
        A^T X E
        + E^T X A
        + \sum_{k = 1}^m N_i^T X N_i
        + B^T B
        = 0

    if `trans` is `True`.

    Parameters
    ----------
    A
        The matrix A as a 2D |NumPy array|.
    E
        The matrix E as a 2D |NumPy array| or `None`.
    N
        The tuple of matrices N_i as 2D |NumPy arrays|.
    B
        The matrix B as a 2D |NumPy array|.
    trans
        Whether the first |Operator| in the Lyapunov equation is transposed.
    maxit
        Maximum number of iterations.
    tol
        Relative error tolerance.

    Returns
    -------
    P
        The Lyapunov equation solution as a 2D |NumPy array|.
    """
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    if E is None:
        E = np.eye(A.shape)
    assert isinstance(E, np.ndarray)
    assert E.ndim == 2
    assert E.shape[0] == E.shape[1]
    assert E.shape[0] == A.shape[0]

    assert isinstance(N, tuple)
    assert all(isinstance(Ni, np.ndarray) for Ni in N)
    assert all(Ni.ndim == 2 for Ni in N)
    assert all(Ni.shape[0] == Ni.shape[1] for Ni in N)

    assert isinstance(B, np.ndarray)
    assert B.ndim == 2
    assert B.shape[0] == A.shape[0]

    assert maxit > 0
    assert tol >= 0

    logger = getLogger('pymor.algorithms.lyapunov.solve_bilinear_lyap_dense')
    B_updated = B
    for i in range(maxit):
        with logger.block(f'Iteration {i + 1}'):
            logger.info('Solving Lyapunov equation ...')
            P = solve_cont_lyap_dense(A, E, B_updated, trans=trans)
            Z = _chol(P)
            if tol > 0:
                logger.info('Computing error ...')
                error = _compute_bilinear_lyap_dense_error(A, E, N, B, Z, trans)
                logger.info(f'Error: {error:.3e}')
                if error <= tol:
                    break
            if i < maxit - 1:
                logger.info('Updating B ...')
                if not trans:
                    B_updated = np.hstack((B,) + tuple(Ni @ Z for Ni in N))
                else:
                    B_updated = np.hstack((B,) + tuple(Ni.T @ Z for Ni in N))
                logger.info('Compressing B ...')
                B_updated = _chol(B_updated @ B_updated.T)
    return P


def _compute_bilinear_lyap_dense_error(A, E, N, B, Z, trans):
    if not trans:
        U = np.hstack((A @ Z, E @ Z, B) + tuple(Ni @ Z for Ni in N))
    else:
        U = np.hstack((A.T @ Z, E.T @ Z, B) + tuple(Ni.T @ Z for Ni in N))
    U, R = spla.qr(U, mode='economic')
    R2 = R.copy()
    tmp = R[:, : len(Z)].copy()
    R[:, : len(Z)] = R[:, len(Z) : 2 * len(Z)]
    R[:, len(Z) : 2 * len(Z)] = tmp
    error = spla.norm(R @ R2.T)
    B_norm = spla.norm(B.T @ B)
    return error / B_norm


def _compress(Z):
    if isinstance(Z, VectorArray):
        U, s, _ = qr_svd(Z, rtol=1e-10)
        U.scal(s)
    else:
        U, s, _ = spla.svd(Z, full_matrices=False, lapack_driver='gesvd')
        U *= s
    return U


def _chol(A):
    """Cholesky decomposition.

    This implementation uses SVD to compute the Cholesky factor (can be used for singular matrices).

    Parameters
    ----------
    A
        Symmetric positive semidefinite matrix as a |NumPy array|.

    Returns
    -------
    L
        Cholesky factor of A (in the sense that L * L^T approximates A).
    """
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    U, s, _ = spla.svd(A, lapack_driver='gesvd')
    L = U * np.sqrt(s)
    return L
