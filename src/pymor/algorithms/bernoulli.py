# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.core.logger import getLogger
from pymor.operators.constructions import IdentityOperator


def solve_bernoulli(A, E, B, trans=False, maxiter=100, after_maxiter=3, tol=1e-8):
    """Compute a solution factor of a Bernoulli equation.

    Returns a matrix :math:`Y` with identical dimensions to the matrix :math:`A` such that
    :math:`X = Y Y^H` is an approximate solution of a (generalized) algebraic Bernoulli equation:

    - if trans is `True`

      .. math::
          A^H X E + E^H X A
          - E^H X B B^H X E = 0.

    - if trans is `False`

      .. math::
          A X E^H + E X A^H
          - E X B^H B X E^H = 0.

    This function is based on :cite:`BBQ07`.

    Parameters
    ----------
    A
        The matrix A as a 2D |NumPy array|.
    E
        The matrix E as a 2D |NumPy array| or `None`.
    B
        The matrix B as a 2D |NumPy array|.
    trans
        Whether to solve transposed or standard Bernoulli equation.
    maxiter
        The maximum amount of iterations.
    after_maxiter
        The number of iterations which are to be performed after tolerance is reached.
        This will improve the quality of the solution in cases where the iterates which are used
        by the stopping criterion stagnate prematurely.
    tol
        Tolerance for stopping criterion based on relative change of iterates.

    Returns
    -------
    Y
        The solution factor as a |NumPy array|.
    """
    logger = getLogger('pymor.algorithms.bernoulli.solve_bernoulli')

    n = len(A)
    after_iter = 0

    assert n != 0

    if E is None:
        E = np.eye(n)

    if not trans:
        E = E.conj().T
        A = A.conj().T
        B = B.conj().T

    for i in range(maxiter):
        Aprev = A
        lu, piv = spla.lu_factor(A.conj().T)
        lndetA = np.sum(np.log(np.abs(np.diag(lu))))
        if E is not None:
            lndetE = np.linalg.slogdet(E)[1]
        else:
            lndetE = 0.
        c = np.exp((1./n)*(lndetA - lndetE))
        AinvTET = spla.lu_solve((lu, piv), E.conj().T)
        if E is not None:
            A = 0.5 * ((1 / c) * A + c * AinvTET.conj().T @ E)
        else:
            A = 0.5 * ((1 / c) * A + c * AinvTET.conj().T)
        BT = (1 / np.sqrt(2 * c)) * np.vstack((B.conj().T, c * B.conj().T @ AinvTET))
        Q, R, perm = spla.qr(BT, mode='economic', pivoting=True)
        B = np.eye(n)[perm].T @ R.conj().T
        if after_iter > after_maxiter:
            break
        rnorm = spla.norm(A - Aprev) / spla.norm(A)
        logger.info(f'Relative change of iterates at step {i}: {rnorm:.5e}')
        if rnorm <= tol:
            after_iter += 1

    if rnorm > tol:
        logger.warning(f'Prescribed tolerance for relative change of iterates not achieved '
                       f'({rnorm:e} > {tol:e}) after ' f'{maxiter} steps.')

    Q, R, _ = spla.qr(E.conj() - A.conj(), pivoting=True)
    nsp_rk = 0
    for r in R:
        if np.allclose(r, np.zeros(r.shape)):
            nsp_rk = nsp_rk + 1
    Q = Q[:, n-nsp_rk:].conj()
    _, R = spla.qr(B.conj().T @ Q, mode='economic')
    Y = spla.solve_triangular(R, np.sqrt(2) * Q.conj().T, trans='C')

    return Y.conj().T


def bernoulli_stabilize(A, E, B, ast_spectrum, trans=False):
    """Compute Bernoulli stabilizing feedback.

    Returns a matrix :math:`K` that stabilizes the spectrum of the matrix pair
    :math:`(A, E)`:

    - if trans is `True` the spectrum of

      .. math::
          (A - B K, E)

      contains the eigenvalues of :math:`(A, E)` where anti-stable eigenvalues have
      been mirrored on the imaginary axis.

    - if trans is `False` the spectrum of

      .. math::
          (A - K B, E)

      contains the eigenvalues of :math:`(A, E)` where anti-stable eigenvalues have
      been mirrored on the imaginary axis.

    See e.g. :cite:`BBQ07`.

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E.
    B
        The operator B as a |VectorArray|.
    ast_spectrum
        Tuple `(lev, ew, rev)` where `ew` contains the anti-stable eigenvalues
        and `lev` and `rev` are |VectorArrays| representing the eigenvectors.
    trans
        Indicates which stabilization to perform.

    Returns
    -------
    K
        The stabilizing feedback as a |VectorArray|.
    """
    if E is None:
        E = IdentityOperator(A.source)

    ast_levs = ast_spectrum[0]
    ast_revs = ast_spectrum[2]

    Mt = E.apply2(ast_levs, ast_revs)
    At = A.apply2(ast_levs, ast_revs)

    if trans:
        Bt = ast_levs.inner(B)
    else:
        Bt = B.inner(ast_revs)

    Yz = solve_bernoulli(At, Mt, Bt, trans=trans)
    Xz = Yz @ Yz.conj().T

    if trans:
        K = E.apply_adjoint(ast_levs.conj()).lincomb(B.inner(ast_levs) @ Xz)
    else:
        K = E.apply_adjoint(ast_revs.conj()).lincomb(B.inner(ast_revs) @ Xz)

    return K.real
