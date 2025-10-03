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
from pymor.algorithms.lyapunov import _solve_lyap_lrcf_check_args
from pymor.algorithms.mat_eqn_solvers.tools import compress_fac
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger


@defaults('maxiter', 'atol', 'rtol', 'ctol')
def lyap_lrcf_solver_options(maxiter=100, atol=0, rtol=None, ctol=None):
    """Return available Lyapunov solvers with default options for the internal backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'internal': {'type': 'internal',
                         'maxiter': maxiter,
                         'atol': atol,
                         'rtol': rtol,
                         'ctol': ctol}}


def solve_lyap_lrcf(A, E, B, trans=False, cont_time=True, options=None):
    _solve_lyap_lrcf_check_args(A, E, B, trans)
    options = _parse_options(options, lyap_lrcf_solver_options(), 'internal', None, False)

    if options['type'] == 'internal':
        space = A.source
        A = to_matrix(A, format='dense')
        E = to_matrix(E, format='dense') if E else None
        B = B.to_numpy().T
        if trans:
            A = A.T
            if E is not None:
                E = E.T
        solve = lyap_sgn_fac if cont_time else dlyap_smith_fac
        Z, _ = solve(A, B, E,
                     maxiter=options['maxiter'],
                     atol=options['atol'],
                     rtol=options['rtol'],
                     ctol=options['ctol'])
        return space.from_numpy(Z.T)
    else:
        raise ValueError(f"Unexpected Lyapunov equation solver ({options['type']}).")



def lyap_sgn_fac(A, B, E, maxiter=100, atol=0, rtol=None, ctol=None):
    """Solve continuous-time Lyapunov equation.

    Computes the full-rank solution of the standard Lyapunov equation

       A*X + X*A^T + B*B^T = 0,                                         (1)

    or of the generalized Lyapunov equation

       A*X*E^T + E*X*A^T + B*B^T = 0,                                   (2)

    with X = Z*Z^T, using the sign function iteration. It is assumed that
    the eigenvalues of A (or s*E - A) lie in the open left half-plane.
    See :cite:`BCQ98`.

    Parameters
    ----------
    A
        |NumPy array| with dimensions n x n in (1) or (2).
    B
        |NumPy array| with dimensions n x m in (1) or (2).
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

    See Also
    --------
    lyapdl_sgn_fac
    lyap_sgn_ldl
    """
    # check inputs
    n, E, hasE, rtol, ctol = _check_lyap_fac_inputs(A, B, E, maxiter, atol, rtol, ctol)

    # case of empty data
    if n == 0:
        Z = np.empty((0, 0))
        info = {}
        return Z, info

    # initialization
    logger = getLogger('pymor.algorithms.mat_eqn_solver.lyap_sgn.lyap_sgn_fac')
    Z = B
    niter = 1
    converged = False

    abs_err = []
    rel_err = []

    normE = spla.norm(E) if hasE else np.sqrt(n)

    # sign function iteration
    while niter <= maxiter and not converged:
        EAinv = spla.solve(A.T, E.T).T
        EAinvE = EAinv @ E if hasE else EAinv

        # scaling factor for convergence acceleration
        if niter == 1 or rel_err[-1] > 1e-2:
            c = np.sqrt(spla.norm(A) / spla.norm(EAinvE))
        else:
            c = 1.0
        c1 = 1.0 / (2.0 * c)

        # construction of next full-rank factor with column compression
        Z = np.sqrt(c1) * compress_fac(np.hstack([Z, c * (EAinv @ Z)]), ctol)

        # update of iteration matrix
        A = c1 * A + (0.5 * c) * EAinvE

        # information about current iteration step
        abs_err.append(spla.norm(A + E))
        rel_err.append(abs_err[-1] / normE)

        logger.info(f'step {niter:4d}, absolute error {abs_err[-1]:e}, relative error {rel_err[-1]:e}')

        # method is converged if absolute or relative errors are small enough
        converged = abs_err[-1] <= atol or rel_err[-1] <= rtol
        niter += 1

    Z = np.sqrt(0.5) * spla.solve(E, Z) if hasE else np.sqrt(0.5) * Z

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
    }

    return Z, info


def dlyap_smith_fac(A, B, E, maxiter=100, atol=0, rtol=None, ctol=None):
    """Solve discrete-time Lyapunov equation.

    Computes the solution matrix of the standard discrete-time Lyapunov
    equation

        A*X*A^T - X + B*B^T = 0,                                          (1)

    or of the generalized Lyapunov equation

        A*X*A^T - E^T*X*E + B*B^T = 0,                                     (2)

    with X = Z*Z^T using the Smith iteration. It is assumed that the
    eigenvalues of A (or s*E - A) lie inside the open unit-circle.
    See :cite:`S16`.

    Parameters
    ----------
    A
        |NumPy array| with dimensions n x n in (1) or (2).
    B
        |NumPy array| with dimensions n x m in (1) or (2).
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

    See Also
    --------
    dlyap_smith
    dlyapdl_smith_fac
    """
    # check inputs
    n, E, hasE, rtol, ctol = _check_lyap_fac_inputs(A, B, E, maxiter, atol, rtol, ctol)

    # case of empty data
    if n == 0:
        Z = np.empty((0, 0))
        info = {}
        return Z, info

    # initialization
    logger = getLogger('pymor.algorithms.mat_eqn_solver.lyap_sgn.dlyap_smith_fac')
    if hasE:
        A = spla.solve(E, A)
        Z = spla.solve(E, B)
    else:
        Z = B
    niter = 1
    converged = False

    abs_err = []
    rel_err = []

    # squared Smith iteration
    while niter <= maxiter and not converged:
        # construction of next solution matrix
        AZ = A @ Z
        Z  = compress_fac(np.hstack([Z, AZ]), ctol)

        # update of iteration matrix
        A = A @ A

        # information about current iteration step
        abs_err.append(spla.norm(AZ))
        rel_err.append(abs_err[-1] / spla.norm(Z))

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
    }

    return Z, info


def _check_lyap_fac_inputs(A, B, E, maxiter, atol, rtol, ctol):
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    assert isinstance(B, np.ndarray)
    assert B.ndim == 2
    assert B.shape[0] == A.shape[0]

    assert E is None or isinstance(E, np.ndarray)
    if E is None:
        E = np.eye(A.shape[0])
        hasE = False
    else:
        assert E.ndim == 2
        assert E.shape[0] == E.shape[1]
        assert E.shape[0] == A.shape[0]
        hasE = True

    n = A.shape[0]

    assert isinstance(maxiter, Integral)
    assert maxiter >= 1

    assert atol >= 0

    if rtol is None:
        rtol = 10 * n * np.finfo(np.float64).eps
    assert rtol >= 0

    if ctol is None:
        ctol = 1e-2 * np.sqrt(n * np.finfo(np.float64).eps)
    assert ctol >= 0

    return n, E, hasE, rtol, ctol
