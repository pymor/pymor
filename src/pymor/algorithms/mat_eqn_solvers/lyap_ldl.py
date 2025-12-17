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

from pymor.algorithms.mat_eqn_solvers.tools import compress_ldl
from pymor.core.logger import getLogger


def lyap_sgn_ldl(A, B, R, E, maxiter=100, atol=0, rtol=None, ctol=None):
    """Solve continuous-time Lyapunov equation.

    Computes the full-rank solution of the standard Lyapunov equation

        A*X + X*A^T + B*R*B^T = 0,                                      (1)

    or of the generalized Lyapunov equation

        A*X*E^T + E*X*A^T + B*R*B^T = 0,                                (2)

    with X = Z*Y*Z^T, using the sign function iteration. It is assumed that
    the eigenvalues of A (or s*E - A) lie in the open left half-plane.

    Parameters
    ----------
    A
        |NumPy array| with dimensions n x n in (1) or (2).
    B
        |NumPy array| with dimensions n x m in (1) or (2).
    R
        Symmetric |NumPy array| with dimensions m x m in (1) or (2),
        If `None`, is is assumed to be the identity.
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
        Full-rank solution factor of (1) or (2), such that X = Z*Y*Z',
        as a |NumPy array| with dimensions n x r.
    Y
        Full-rank solution factor of (1) or (2), such that X = Z*Y*Z',
        as a |NumPy array| with dimensions r x r.
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
    lyapdl_sgn_ldl
    lyap_sgn_fac
    """
    # check inputs
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    n = A.shape[0]

    assert isinstance(B, np.ndarray)
    assert B.ndim == 2
    assert B.shape[0] == A.shape[0]

    m = B.shape[1]

    assert R is None or isinstance(R, np.ndarray)
    if R is None:
        R = np.eye(m)
    assert R.ndim == 2
    assert R.shape[0] == m
    assert R.shape[1] == m
    assert np.all(R == R.T)

    assert E is None or isinstance(E, np.ndarray)
    if E is None:
        E = np.eye(A.shape[0])
        hasE = False
    else:
        assert E.ndim == 2
        assert E.shape[0] == n
        assert E.shape[1] == n
        hasE = True

    # check and assign optional parameters
    assert isinstance(maxiter, Integral)
    assert maxiter >= 1

    assert atol >= 0

    if rtol is None:
        rtol = 10 * n * np.finfo(np.float64).eps
    assert rtol >= 0

    if ctol is None:
        ctol = np.sqrt(n) * np.finfo(np.float64).eps
    assert ctol >= 0

    # case of empty data
    if n == 0:
        Z = np.empty((0, 0))
        Y = np.empty((0, 0))
        info = {}
        return Z, Y, info

    # initialization
    logger = getLogger('pymor.algorithms.mat_eqn_solver.lyap_sgn.lyap_sgn_fac')
    Z = B
    Y = R
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
        c2 = 0.5 * c

        # construction of full-rank factorization with LDL row compression
        [Z, Y] = compress_ldl(np.hstack([Z, EAinv @ Z]), spla.block_diag(c1 * Y, c2 * Y), ctol)

        # update of iteration matrix
        A = c1 * A + c2 * EAinvE

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

    return Z, Y, info
