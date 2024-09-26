# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

# This file was originally based upon the MORLAB toolbox
# Copyright (C) 2006-2023 Peter Benner, Jens Saak, and Steffen W. R. Werner
# All rights reserved.
# License: BSD 2-Clause License

from numbers import Integral

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.genericsolvers import _parse_options
from pymor.algorithms.mat_eqn_solvers.lyap_sgn import compress_fac, lyap_sgn_fac
from pymor.algorithms.riccati import _solve_ricc_check_args, _solve_ricc_dense_check_args
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.numpy import NumpyMatrixOperator


@defaults('maxiter', 'atol', 'rtol', 'lyap_maxiter', 'lyap_atol', 'lyap_rtol', 'lyap_ctol')
def ricc_lrcf_solver_options(maxiter=100, atol=0, rtol=None,
                             lyap_maxiter=100, lyap_atol=0, lyap_rtol=None, lyap_ctol=None):
    """Return available Riccati solvers with default options for the internal backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {
        'internal': {
            'type': 'internal',
            'maxiter': maxiter,
            'atol': atol,
            'rtol': rtol,
            'lyap_opts': {
                'maxiter': lyap_maxiter,
                'atol': lyap_atol,
                'rtol': lyap_rtol,
                'ctol': lyap_ctol,
            }
        }
    }


def solve_ricc_lrcf(A, E, B, C, R=None, S=None, trans=False, options=None):
    """Compute an approximate low-rank solution of a Riccati equation.

    See :func:`pymor.algorithms.riccati.solve_ricc_lrcf` for a general
    description.

    This function uses :func:`care_nwt_fac`, which is a dense solver.
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
    S
        The operator S as a |VectorArray| from `A.source` or `None`.
    trans
        Whether the first |Operator| in the Riccati equation is
        transposed.
    options
        The solver options to use (see :func:`ricc_lrcf_solver_options`).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Riccati equation solution,
        |VectorArray| from `A.source`.
    """
    _solve_ricc_check_args(A, E, B, C, R, S, trans)
    options = _parse_options(options, ricc_lrcf_solver_options(), 'internal', None, False)
    if options['type'] != 'internal':
        raise ValueError(f"Unexpected Riccati equation solver ({options['type']}).")

    A_mat = to_matrix(A, format='dense')
    E_mat = to_matrix(E, format='dense') if E else None
    B_mat = B.to_numpy().T
    C_mat = C.to_numpy()
    if R is not None:
        raise NotImplementedError
    if S is not None:
        raise NotImplementedError
    if not trans:
        A_mat = A_mat.T
        B_mat, C_mat = C_mat.T, B_mat.T
        if E:
            E_mat = E_mat.T
    Z, _ = care_nwt_fac(A_mat, B_mat, C_mat, E_mat,
                        maxiter=options['maxiter'],
                        atol=options['atol'],
                        rtol=options['rtol'],
                        lyap_opts=options['lyap_opts'])

    return A.source.from_numpy(Z.T)


@defaults('maxiter', 'atol', 'rtol', 'lyap_maxiter', 'lyap_atol', 'lyap_rtol', 'lyap_ctol')
def ricc_dense_solver_options(maxiter=100, atol=0, rtol=None,
                              lyap_maxiter=100, lyap_atol=0, lyap_rtol=None, lyap_ctol=None):
    """Return available Riccati solvers with default options for the internal backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {
        'internal': {
            'type': 'internal',
            'maxiter': maxiter,
            'atol': atol,
            'rtol': rtol,
            'lyap_opts': {
                'maxiter': lyap_maxiter,
                'atol': lyap_atol,
                'rtol': lyap_rtol,
                'ctol': lyap_ctol,
            }
        }
    }


def solve_ricc_dense(A, E, B, C, R=None, S=None, trans=False, options=None):
    """Compute the solution of a Riccati equation.

    See :func:`pymor.algorithms.riccati.solve_ricc_dense` for a general
    description.

    This function uses :func:`care_nwt_fac`, which is a dense solver.

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
    S
        The matrix S as a 2D |NumPy array| or `None`.
    trans
        Whether the first operator in the Riccati equation is
        transposed.
    options
        The solver options to use (see :func:`ricc_dense_solver_options`).

    Returns
    -------
    X
        Riccati equation solution as a |NumPy array|.
    """
    _solve_ricc_dense_check_args(A, E, B, C, R, S, trans)
    options = _parse_options(options, ricc_dense_solver_options(), 'internal', None, False)
    if options['type'] != 'internal':
        raise ValueError(f"Unexpected Riccati equation solver ({options['type']}).")

    if R is not None:
        raise NotImplementedError
    if S is not None:
        raise NotImplementedError

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E) if E is not None else None
    Bva = Aop.source.from_numpy(B.T)
    Cva = Aop.source.from_numpy(C)
    Zva = solve_ricc_lrcf(Aop, Eop, Bva, Cva, trans=trans, options=options)
    Z = Zva.to_numpy().T
    X = Z @ Z.T
    return X


@defaults('maxiter', 'atol', 'rtol', 'ctol', 'lyap_maxiter', 'lyap_atol', 'lyap_rtol', 'lyap_ctol')
def pos_ricc_lrcf_solver_options(maxiter=100, atol=0, rtol=None, ctol=None,
                                 lyap_maxiter=100, lyap_atol=0, lyap_rtol=None, lyap_ctol=None):
    """Return available Riccati solvers with default options for the internal backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {
        'internal': {
            'type': 'internal',
            'maxiter': maxiter,
            'atol': atol,
            'rtol': rtol,
            'ctol': ctol,
            'lyap_opts': {
                'maxiter': lyap_maxiter,
                'atol': lyap_atol,
                'rtol': lyap_rtol,
                'ctol': lyap_ctol,
            }
        }
    }


def solve_pos_ricc_lrcf(A, E, B, C, R=None, S=None, trans=False, options=None):
    """Compute an approximate low-rank solution of a Riccati equation.

    See :func:`pymor.algorithms.riccati.solve_pos_ricc_lrcf` for a general
    description.

    This function uses :func:`pcare_nwt_fac`, which is a dense solver.
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
    S
        The operator S as a |VectorArray| from `A.source` or `None`.
    trans
        Whether the first |Operator| in the Riccati equation is
        transposed.
    options
        The solver options to use (see :func:`ricc_lrcf_solver_options`).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Riccati equation solution,
        |VectorArray| from `A.source`.
    """
    _solve_ricc_check_args(A, E, B, C, R, S, trans)
    options = _parse_options(options, pos_ricc_lrcf_solver_options(), 'internal', None, False)
    if options['type'] != 'internal':
        raise ValueError(f"Unexpected Riccati equation solver ({options['type']}).")

    A_mat = to_matrix(A, format='dense')
    E_mat = to_matrix(E, format='dense') if E else None
    B_mat = B.to_numpy().T
    C_mat = C.to_numpy()
    if R is not None:
        raise NotImplementedError
    if S is not None:
        raise NotImplementedError
    if not trans:
        A_mat = A_mat.T
        B_mat, C_mat = C_mat.T, B_mat.T
        if E:
            E_mat = E_mat.T
    Z, _ = pcare_nwt_fac(A_mat, B_mat, C_mat, E_mat,
                         maxiter=options['maxiter'],
                         atol=options['atol'],
                         rtol=options['rtol'],
                         ctol=options['ctol'],
                         lyap_opts=options['lyap_opts'])

    return A.source.from_numpy(Z.T)


@defaults('maxiter', 'atol', 'rtol', 'ctol', 'lyap_maxiter', 'lyap_atol', 'lyap_rtol', 'lyap_ctol')
def pos_ricc_dense_solver_options(maxiter=100, atol=0, rtol=None, ctol=None,
                                  lyap_maxiter=100, lyap_atol=0, lyap_rtol=None, lyap_ctol=None):
    """Return available Riccati solvers with default options for the internal backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {
        'internal': {
            'type': 'internal',
            'maxiter': maxiter,
            'atol': atol,
            'rtol': rtol,
            'ctol': ctol,
            'lyap_opts': {
                'maxiter': lyap_maxiter,
                'atol': lyap_atol,
                'rtol': lyap_rtol,
                'ctol': lyap_ctol,
            }
        }
    }


def solve_pos_ricc_dense(A, E, B, C, R=None, S=None, trans=False, options=None):
    """Compute the solution of a Riccati equation.

    See :func:`pymor.algorithms.riccati.solve_ricc_dense` for a general
    description.

    This function uses :func:`pcare_nwt_fac`, which is a dense solver.

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
    S
        The matrix S as a 2D |NumPy array| or `None`.
    trans
        Whether the first operator in the Riccati equation is
        transposed.
    options
        The solver options to use (see :func:`ricc_dense_solver_options`).

    Returns
    -------
    X
        Riccati equation solution as a |NumPy array|.
    """
    _solve_ricc_dense_check_args(A, E, B, C, R, S, trans)
    options = _parse_options(options, pos_ricc_dense_solver_options(), 'internal', None, False)
    if options['type'] != 'internal':
        raise ValueError(f"Unexpected Riccati equation solver ({options['type']}).")

    if R is not None:
        raise NotImplementedError
    if S is not None:
        raise NotImplementedError

    Aop = NumpyMatrixOperator(A)
    Eop = NumpyMatrixOperator(E) if E is not None else None
    Bva = Aop.source.from_numpy(B.T)
    Cva = Aop.source.from_numpy(C)
    Zva = solve_pos_ricc_lrcf(Aop, Eop, Bva, Cva, trans=trans, options=options)
    Z = Zva.to_numpy().T
    X = Z @ Z.T
    return X


def care_nwt_fac(A, B, C, E, maxiter=100, atol=0, rtol=None, K0=None, lyap_opts={}):
    """Solve continuous-time Riccati equation.

    Computes the full-rank solutions of the standard algebraic Riccati
    equation

        A^T*X + X*A - X*B*B^T*X + C^T*C = 0,                            (1)

    or of the generalized Riccati equation

        A^T*X*E + E^T*X*A - E^T*X*B*B^T*X*E + C^T*C = 0,                (2)

    with X = Z*Z^T, using the Newton-Kleinman iteration. It is assumed that
    the eigenvalues of A (or s*E - A) lie in the open left half-plane,
    otherwise a stabilizing initial feedback K0 is given as parameter.
    See :cite:`BS13`.

    Parameters
    ----------
    A
        |NumPy array| with dimensions n x n in (1) or (2).
    B
        |NumPy array| with dimensions n x m in (1) or (2).
    C
        |NumPy array| with dimensions p x n in (1) or (2).
    E
        |NumPy array| with dimensions n x n in (2).
        If `None`, the standard equation (1) is solved.
    maxiter
        Positive integer, maximum number of iteration steps.
    atol
        Nonnegative scalar, tolerance for the absolute error in the last
        iteration step.
    rtol
        Nonnegative scalar, tolerance for the relative error in the last
        iteration step.
        If `None`, the value is `10*n*eps`.
    K0
        |NumPy array| with dimensions m x n, used to stabilize the spectrum of
        s*E - A, such that s*E - (A - BK0) has only stable eigenvalues.
        If `None`, taken to be zero.
    lyap_opts
        Dict containing the optional parameters for the Lyapunov equation solver
        see `lyap_sgn_fac`.

    Returns
    -------
    Z
        Full-rank solution factor of (1) or (2), such that `X = Z*Z^T`,
        as a |NumPy array| with dimensions n x r.
    info
        Dict with the following fields:

        :abs_err:
            Vector, containing the absolute error of the iteration matrix in
            each iteration step.
        :rel_err:
            Vector, containing the relative error of the iteration matrix in
            each iteration step.
        :num_iter:
            Number of performed iteration steps.
        :info_lyap:
            List of dicts, containing information about the used Lyapunov
            equations solver for every iteration step, see `lyap_sgn_fac`.

    See Also
    --------
    icare_ric_fac
    pcare_nwt_fac
    """
    # check input matrices
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    assert isinstance(B, np.ndarray)
    assert B.ndim == 2
    assert B.shape[0] == A.shape[0]

    assert isinstance(C, np.ndarray)
    assert C.ndim == 2
    assert C.shape[1] == A.shape[0]

    assert E is None or isinstance(E, np.ndarray)
    if E is not None:
        assert E.ndim == 2
        assert E.shape[0] == E.shape[1]
        assert E.shape[0] == A.shape[0]

    n = A.shape[0]
    m = B.shape[1]

    # check and assign optional parameters
    assert isinstance(maxiter, Integral)
    assert maxiter >= 1

    assert atol >= 0

    if rtol is None:
        rtol = 10 * n * np.finfo(np.float64).eps
    assert rtol >= 0

    assert isinstance(lyap_opts, dict)

    assert K0 is None or isinstance(K0, np.ndarray)
    if K0 is None:
        K0 = np.zeros((m, n))
    assert K0.ndim == 2
    assert K0.shape[0] == m
    assert K0.shape[1] == n

    # case of empty data
    if n == 0:
        Z = np.empty((0, 0))
        info = {}
        return Z, info

    # initialization
    logger = getLogger('pymor.algorithms.mat_eqn_solver.care.care_nwt_fac')
    niter = 1
    converged = False
    ET = None if E is None else E.T
    K = K0

    abs_err = []
    rel_err = []
    info_lyap = []

    # Newton-Kleinman iteration
    while niter <= maxiter and not converged:
        W = np.vstack((C, K))
        Z, info_lyap_cur = lyap_sgn_fac((A - B @ K).T, W.T, ET, **lyap_opts)

        AZ = A.T @ Z
        ZE = Z.T @ E if E is not None else Z.T
        K = (B.T @ Z) @ ZE

        # information about current iteration step
        abs_err.append(spla.norm(AZ @ ZE + ZE.T @ AZ.T - K.T @ K + C.T @ C))
        rel_err.append(abs_err[-1] / max(spla.norm(Z.T @ Z), 1))
        info_lyap.append(info_lyap_cur)

        logger.info(f'step {niter:4d}, absolute error {abs_err[-1]:e}, relative error {rel_err[-1]:e}')

        # method is converged if absolute or relative errors are small enough
        converged = abs_err[-1] <= atol or rel_err[-1] <= rtol
        niter += 1

    niter -= 1

    # warning if iteration not converged
    if niter == maxiter and not converged:
        logger.warning(
            f'No convergence in {niter:d} iteration steps.\n'
            f'Abs. tolerance: {atol:e}, Abs. error: {abs_err[-1]:e}\n'
            f'Rel. tolerance: {rtol:e}, Rel. error: {rel_err[-1]:e}\n'
            f'Try to increase the tolerances or number of iteration steps.'
        )

    # assign information about iteration
    info = {
        'abs_err': np.array(abs_err),
        'rel_err': np.array(rel_err),
        'num_iter': niter,
        'info_lyap': info_lyap,
    }

    return Z, info


def pcare_nwt_fac(A, B, C, E, maxiter=100, atol=0, rtol=None, ctol=None, lyap_opts={}):
    """Solve positive continuous-time Riccati equation.

    Computes the full-rank solutions of the standard positive Riccati
    equation

        A^T*X + X*A + X*B*B^T*X + C^T*C = 0,                            (1)

    or of the generalized positive Riccati equation

        A^T*X*E + E^T*X*A + E^T*X*B*B^T*X*E + C^T*C = 0,                (2)

    with X = Z*Z^T, using the low-rank Newton iteration. It is assumed that
    the eigenvalues of A (or s*E - A) lie in the open left half-plane and
    that the equation (1) (or (2)) has a solution.
    See :cite:`V95`.

    Parameters
    ----------
    A
        |NumPy array| with dimensions n x n in (1) or (2).
    B
        |NumPy array| with dimensions n x m in (1) or (2).
    C
        |NumPy array| with dimensions p x n in (1) or (2).
    E
        |NumPy array| with dimensions n x n in (2).
        If `None`, the standard equation (1) is solved.
    maxiter
        Positive integer, maximum number of iteration steps.
    atol
        Nonnegative scalar, tolerance for the absolute error in the last
        iteration step.
    rtol
        Nonnegative scalar, tolerance for the relative error in the last
        iteration step.
        If `None`, the value is `10*n*eps`.
    ctol
        Nonnegative scalar, tolerance for the column compression during the
        iteration.
        If `None`, the value is `1e-2*sqrt(n*eps)`.
    lyap_opts
        Dict containing the optional parameters for the Lyapunov equation solver
        see `lyap_sgn_fac`.

    Returns
    -------
    Z
        Full-rank solution factor of (1) or (2), such that `X = Z*Z^T`,
        as a |NumPy array| with dimensions n x r.
    info
        Dict with the following fields:

        :abs_err:
            Vector, containing the absolute error of the iteration matrix in
            each iteration step.
        :rel_err:
            Vector, containing the relative error of the iteration matrix in
            each iteration step.
        :num_iter:
            Number of performed iteration steps.
        :info_lyap:
            List of dicts, containing information about the used Lyapunov
            equations solver for every iteration step, see `lyap_sgn_fac`.

    See Also
    --------
    care_nwt_fac
    icare_ric_fac
    """
    # check input matrices
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    assert isinstance(B, np.ndarray)
    assert B.ndim == 2
    assert B.shape[0] == A.shape[0]

    assert isinstance(C, np.ndarray)
    assert C.ndim == 2
    assert C.shape[1] == A.shape[0]

    assert E is None or isinstance(E, np.ndarray)
    if E is not None:
        assert E.ndim == 2
        assert E.shape[0] == E.shape[1]
        assert E.shape[0] == A.shape[0]

    n = A.shape[0]

    # check and assign optional parameters
    assert isinstance(maxiter, Integral)
    assert maxiter >= 1

    assert atol >= 0

    if rtol is None:
        rtol = 10 * n * np.finfo(np.float64).eps
    assert rtol >= 0

    if ctol is None:
        ctol = 1e-2 * np.sqrt(n * np.finfo(np.float64).eps)
    assert ctol >= 0

    assert isinstance(lyap_opts, dict)

    # case of empty data
    if n == 0:
        Z = np.empty((0, 0))
        info = {}
        return Z, info

    # initialization
    logger = getLogger('pymor.algorithms.mat_eqn_solver.care.pcare_nwt_fac')
    niter = 2
    converged = False
    ET = None if E is None else E.T

    abs_err = []
    rel_err = []
    info_lyap = []

    # initial step
    N, info_lyap_cur = lyap_sgn_fac(A.T, C.T, ET, **lyap_opts)

    Z = N
    K = B.T @ N @ N.T
    if E is not None:
        K = K @ E

    abs_err.append(spla.norm(K @ K.T))
    rel_err.append(abs_err[-1] / max(spla.norm(Z.T @ Z), 1))

    logger.info(f'step {niter:4d}, absolute error {abs_err[-1]:e}, relative error {rel_err[-1]:e}')

    info_lyap.append(info_lyap_cur)

    # method is converged if absolute or relative errors are small enough
    converged = abs_err[-1] <= atol or rel_err[-1] <= rtol

    # Newton-Kleinman iteration
    while niter <= maxiter and not converged:
        if E is not None:
            N, info_lyap_cur = lyap_sgn_fac((A + B @ (B.T @ Z) @ (Z.T @ E)).T, K.T, ET, **lyap_opts)
        else:
            N, info_lyap_cur = lyap_sgn_fac((A + B @ (B.T @ Z) @ Z.T).T, K.T, None, **lyap_opts)
        K = B.T @ N @ N.T
        if E is not None:
            K = K @ E

        # column compression
        Z = compress_fac(np.hstack([Z, N]), ctol)

        # information about current iteration step
        abs_err.append(spla.norm(K @ K.T))
        rel_err.append(abs_err[-1] / max(spla.norm(Z.T @ Z), 1))
        info_lyap.append(info_lyap_cur)

        logger.info(f'step {niter:4d}, absolute error {abs_err[-1]:e}, relative error {rel_err[-1]:e}')

        # method is converged if absolute or relative errors are small enough
        converged = abs_err[-1] <= atol or rel_err[-1] <= rtol
        niter += 1

    niter -= 1

    # warning if iteration not converged
    if niter == maxiter and not converged:
        logger.warning(
            f'No convergence in {niter:d} iteration steps.\n'
            f'Abs. tolerance: {atol:e}, Abs. error: {abs_err[-1]:e}\n'
            f'Rel. tolerance: {rtol:e}, Rel. error: {rel_err[-1]:e}\n'
            f'Try to increase the tolerances or number of iteration steps.'
        )

    # assign information about iteration
    info = {
        'abs_err': np.array(abs_err),
        'rel_err': np.array(rel_err),
        'num_iter': niter,
        'info_lyap': info_lyap,
    }

    return Z, info
