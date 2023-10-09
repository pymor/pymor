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

from pymor.core.logger import getLogger


def lyap_sgn(A, G, E, atol=0, rtol=None, maxiter=100):
    """Solve continuous-time Lyapunov equation.

    Computes the solution matrix of the standard continuous-time Lyapunov
    equation

        A*X + X*A^T + G = 0,                                             (1)

    or of the generalized Lyapunov equation

        A*X*E^T + E*X*A^T + G = 0,                                        (2)

    using the sign function iteration. It is assumed that the eigenvalues
    of A (or s*E - A) lie in the open left half-plane.

    Parameters
    ----------
    A
        Matrix with dimensions n x n in (1) or (2).
    G
        Matrix with dimensions n x n in (1) or (2).
    E
        Matrix with dimensions n x n in (2).
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

    Returns
    -------
    X
        Solution matrix of (1) or (2).
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

    REFERENCE:
      P. Benner, J. M. Claver, E. S. Quintana-Orti, Efficient solution of
      coupled Lyapunov equations via matrix sign function iteration, in:
      Proc. 3rd Portuguese Conf. on Automatic Control CONTROLO'98, Coimbra,
      1998, pp. 205--210.

    See also lyap_sgn_fac, lyap_sgn_ldl.
    """
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    assert isinstance(G, np.ndarray)
    assert G.ndim == 2
    assert G.shape[0] == G.shape[1]
    assert G.shape[0] == A.shape[0]

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

    # check and assign optional parameters
    assert atol >= 0
    if rtol is None:
        rtol = 10 * n * np.finfo(np.float64).eps
    assert rtol >= 0
    assert isinstance(maxiter, Integral)
    assert maxiter >= 1

    # case of empty data
    if n == 0:
        X = np.empty(0)
        info = {}
        return X, info

    # initialization
    logger = getLogger('pymor.algorithms.mat_eqn_solver.lyap_sgn.lyap_sgn')
    X = G
    niter = 1
    converged = False

    abs_err = []
    rel_err = []

    normE = spla.norm(E) if hasE else np.sqrt(n)

    # sign function iteration
    while niter <= maxiter and not converged:
        EAinv = spla.solve(A.T, E.T, check_finite=False).T
        if hasE:
            EAinvE = EAinv * E
        else:
            EAinvE = EAinv

        # scaling factor for convergence acceleration
        if niter == 1 or rel_err[-1] > 1e-2:
            c = np.sqrt(spla.norm(A) / spla.norm(EAinvE))
        else:
            c = 1.0
        c1 = 1.0 / (2.0 * c)
        c2 = 0.5 * c

        # construction of next solution matrix
        X = c1 * X + c2 * (EAinv @ (X @ EAinv.T))

        # update of iteration matrix
        A = c1 * A + c2 * EAinvE

        # information about current iteration step
        abs_err.append(spla.norm(A + E))
        rel_err.append(abs_err[-1] / normE)

        logger.info(f'step {niter:4d}, absolute error {abs_err[-1]:e}, relative error {rel_err[-1]:e}')

        # method is converged if absolute or relative errors are small enough
        converged = abs_err[-1] <= atol or rel_err[-1] <= rtol
        niter += 1

    if hasE:
        X = 0.5 * spla.solve(E, spla.solve(E, X.T, check_finite=False).T, check_finite=False)
    else:
        X = 0.5 * X

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
