# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.genericsolvers import _parse_options
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.riccati import _solve_ricc_check_args
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.constructions import IdentityOperator, LowRankOperator
from pymor.tools.random import new_rng
from pymor.vectorarrays.constructions import cat_arrays


@defaults('lrradi_tol', 'lrradi_maxiter', 'lrradi_shifts', 'hamiltonian_shifts_init_maxiter',
          'hamiltonian_shifts_subspace_columns')
def ricc_lrcf_solver_options(lrradi_tol=1e-10,
                             lrradi_maxiter=500,
                             lrradi_shifts='hamiltonian_shifts',
                             hamiltonian_shifts_init_maxiter=20,
                             hamiltonian_shifts_subspace_columns=6):
    """Returns available Riccati equation solvers with default solver options.

    Parameters
    ----------
    lrradi_tol
        See :func:`solve_ricc_lrcf`.
    lrradi_maxiter
        See :func:`solve_ricc_lrcf`.
    lrradi_shifts
        See :func:`solve_ricc_lrcf`.
    hamiltonian_shifts_init_maxiter
        See :func:`hamiltonian_shifts_init`.
    hamiltonian_shifts_subspace_columns
        See :func:`hamiltonian_shifts`.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'lrradi': {'type': 'lrradi',
                       'tol': lrradi_tol,
                       'maxiter': lrradi_maxiter,
                       'shifts': lrradi_shifts,
                       'shift_options':
                       {'hamiltonian_shifts': {'type': 'hamiltonian_shifts',
                                               'init_maxiter': hamiltonian_shifts_init_maxiter,
                                               'subspace_columns': hamiltonian_shifts_subspace_columns}}}}


def solve_ricc_lrcf(A, E, B, C, R=None, S=None, trans=False, options=None):
    """Compute an approximate low-rank solution of a Riccati equation.

    See :func:`pymor.algorithms.riccati.solve_ricc_lrcf` for a
    general description.

    This function is an implementation of Algorithm 2 in :cite:`BBKS18`.

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E or `None`.
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
        The solver options to use. (see
        :func:`ricc_lrcf_solver_options`)

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Riccati equation solution,
        |VectorArray| from `A.source`.
    """
    _solve_ricc_check_args(A, E, B, C, R, S, trans)

    if S is None:
        if trans:
            Z_cf = lrradi(A, E, B, C, R, None, S, trans, options)
        else:
            Z_cf = lrradi(A, E, B, C, None, R, S, trans, options)
        return Z_cf
    else:
        if R is not None:
            Rinv = spla.solve(R, np.eye(R.shape[0]))
        else:
            R = Rinv = np.eye(len(B) if trans else len(C))

        if trans:
            BRinvSt = LowRankOperator(B, Rinv, S)

            tA = A - BRinvSt
            tC = cat_arrays([C, S])
            tQ = spla.block_diag(np.eye(len(C)), -Rinv)

            Z_cf = lrradi(tA, E, B, tC, R, tQ, trans, options)
        else:
            CRinvSt = LowRankOperator(C, Rinv, S)

            tA = A - CRinvSt
            tB = cat_arrays([B, S])
            tR = spla.block_diag(np.eye(len(B)), -Rinv)

            Z_cf = lrradi(tA, E, tB, C, tR, R, trans, options)
        return Z_cf

def solve_pos_ricc_lrcf(A, E, B, C, R=None, S=None, trans=False, options=None):
    """Compute an approximate low-rank solution of a positive Riccati equation.

    See :func:`pymor.algorithms.riccati.solve_pos_ricc_lrcf` for a
    general description.

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
        The solver options to use (see
        :func:`ricc_lrcf_solver_options`).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the positive Riccati equation
        solution, |VectorArray| from `A.source`.
    """
    _solve_ricc_check_args(A, E, B, C, R, S, trans)
    options = _parse_options(options, ricc_lrcf_solver_options(), 'lrradi', None, False)
    if options['type'] != 'lrradi':
        raise ValueError(f"Unexpected positive Riccati equation solver ({options['type']}).")

    if R is None:
        R = np.eye(len(C) if not trans else len(B))
    return lrradi(A, E, B, C, -R, None, trans, options)


def lrradi(A, E, B, C, R=None, Q=None, trans=False, options=None):
    """Compute the solution of a Riccati equation.

    Returns the solution :math:`X` of a (generalized) continuous-time
    algebraic Riccati equation:

    - if trans is `False`

      .. math::
          A X E^T + E X A^T - E X C^T Q^{-1} C X E^T + B R B^T = 0.

    - if trans is `True`

      .. math::
          A^T X E + E^T X A - E^T X B R^{-1} B^T X E + C^T Q C = 0.

    If E is None, it is taken to be identity, and similarly for R and Q.

    We assume:

    - A and E are real |Operators|,
    - B and C are real |VectorArrays| from `A.source` and `len(B)` and `len(C)`  are small,
    - R, Q are real |NumPy arrays|,
    - E is nonsingular,
    - (E, A, B, C) is stabilizable and detectable,
    - R is symmetric, Q is symmetric.

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
    Q
        The matrix Q as a 2D |NumPy array| or `None`.
    trans
        Whether the first |Operator| in the Riccati equation is
        transposed.
    options
        The solver options to use.
        See:

        - :func:`pymor.algorithms.lrradi.ricc_lrcf_solver_options`.

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Riccati equation solution,
        |VectorArray| from `A.source`.
    """
    options = _parse_options(options, ricc_lrcf_solver_options(), 'lrradi', None, False)
    logger = getLogger('pymor.algorithms.lrradi.solve_ricc_lrcf')

    shift_options = options['shift_options'][options['shifts']]
    if shift_options['type'] == 'hamiltonian_shifts':
        init_shifts = hamiltonian_shifts_init
        iteration_shifts = hamiltonian_shifts
    else:
        raise ValueError('Unknown lrradi shift strategy.')

    if E is None:
        E = IdentityOperator(A.source)

    if not trans:
        B, C = C, B
        R, Q = Q, R

    if R is not None:
        Rinv = spla.solve(R, np.eye(R.shape[0]))
        Rinv = 0.5 * (Rinv + Rinv.T)
    else:
        R = Rinv = np.eye(len(B))

    Z = A.source.empty(reserve=len(C) * options['maxiter'])
    Y = np.empty((0, 0))

    K = A.source.zeros(len(B))

    RF = C.copy()
    RC = np.eye(len(C)) if Q is None else Q

    j = 0
    j_shift = 0
    shifts = init_shifts(A, E, B, C, Rinv, Q, shift_options)

    if Q is None:
        res = np.linalg.norm(RF.gramian(), ord='fro')
    else:
        res = np.linalg.norm(RF.gramian() @ RC, ord='fro')

    init_res = res
    Ctol = res * options['tol']

    while res > Ctol and j < options['maxiter']:
        s = shifts[j_shift]
        sr = s.real
        si = s.imag
        sa = np.abs(s)
        alpha = np.sqrt(-2.0 * sr)

        if not trans:
            AsE = A + s * E
        else:
            AsE = A + np.conj(s) * E

        BRiK = LowRankOperator(B, Rinv, K)
        AsEBRiK = (AsE - BRiK).assemble() # assemble combines the two low-rank
                                          # updates into a single one if A came
                                          # in as a LowRankUpdatedOperator already
                                          # (avoids recursive Sherman-Morrison-Woodburry)

        if not trans:
            V = AsEBRiK.apply_inverse(RF)
        else:
            V = AsEBRiK.apply_inverse_adjoint(RF)

        V = V.lincomb(RC)

        if np.imag(s) == 0:
            V = V.real
            Z.append(alpha * V)

            VB = V.inner(B)
            Yt = RC + (VB @ Rinv @ VB.T)
            Y = spla.block_diag(Y, Yt)

            if not trans:
                EVYt = E.apply(V).lincomb(spla.inv(Yt))
            else:
                EVYt = E.apply_adjoint(V).lincomb(spla.inv(Yt))
            RF.axpy(-2.0 * sr, EVYt)

            K += EVYt.lincomb(Rinv @ VB.T)
            j += 1
        else:
            V1 = alpha * V.real
            V2 = alpha * V.imag
            Z.append(V1)
            Z.append(V2)
            Vr = V1.inner(B)
            Vi = V2.inner(B)
            F1 = np.vstack((
                -sr/sa * Vr - si/sa * Vi,
                 si/sa * Vr - sr/sa * Vi
            ))
            F2 = np.vstack((
                Vr,
                Vi
            ))
            F3 = np.vstack((
                si/sa * np.eye(len(RF)),
                sr/sa * np.eye(len(RF))
            ))
            Yt = spla.block_diag(RC, 0.5 * RC) \
                - (F1 @ Rinv @ F1.T) / (4 * sr)  \
                - (F2 @ Rinv @ F2.T) / (4 * sr)  \
                - (F3 @ RC @ F3.T) / 2
            Y = spla.block_diag(Y, Yt)
            if not trans:
                EVYt = E.apply(cat_arrays([V1, V2])).lincomb(spla.inv(Yt))
            else:
                EVYt = E.apply_adjoint(cat_arrays([V1, V2])).lincomb(spla.inv(Yt))
            RF.axpy(alpha, EVYt[:len(C)])
            K += EVYt.lincomb(Rinv @ F2.T)
            j += 2
        j_shift += 1
        res = np.linalg.norm(RF.gramian() @ RC, ord='fro')
        logger.info(f'Relative residual at step {j}: {res/init_res:.5e}')
        if j_shift >= shifts.size:
            shifts = iteration_shifts(A, E, B, Rinv, RF, RC, K, Z, shift_options)
            j_shift = 0
    # transform solution to lrcf
    Yinv = spla.inv(Y)
    Yinv = (Yinv + Yinv.T) / 2.0
    Z_cf, S = LDL_T_rank_truncation(Z, Yinv)
    S = np.diag(np.sqrt(np.diag(S)))
    Z_cf = Z_cf.lincomb(S)
    return Z_cf, Z, Yinv


def hamiltonian_shifts_init(A, E, B, C, Rinv, Q, shift_options):
    """Compute initial shift parameters for low-rank RADI iteration.

    Compute Galerkin projection of Hamiltonian matrix on space spanned by :math:`C` and return the
    eigenvalue of the projected Hamiltonian with the most impact on convergence as the next shift
    parameter.

    See :cite:`BBKS18`, pp. 318-321.

    Parameters
    ----------
    A
        The |Operator| A from the corresponding Riccati equation.
    E
        The |Operator| E from the corresponding Riccati equation.
    B
        The |VectorArray| B from the corresponding Riccati equation.
    C
        The |VectorArray| C from the corresponding Riccati equation.
    shift_options
        The shift options to use (see :func:`ricc_lrcf_solver_options`).

    Returns
    -------
    shifts
        A |NumPy array| containing a set of stable shift parameters.
    """
    rng = new_rng(0)

    for _ in range(shift_options['init_maxiter']):
        U = gram_schmidt(C, atol=0, rtol=0)
        Ap = A.apply2(U, U)
        UB = U.inner(B)
        Gp = UB @ (Rinv @ UB.T)
        UR = U.inner(C)
        Rp = (UR @ Q) @ UR.T if Q is not None else UR @ UR.T
        Hp = np.block([
            [Ap, Gp],
            [Rp, -Ap.T]
        ])
        Ep = E.apply2(U, U)
        EEp = spla.block_diag(Ep, Ep.T)
        eigvals, eigvecs = spla.eig(Hp, EEp)
        eigpairs = zip(eigvals, eigvecs)
        # filter stable eigenvalues
        eigpairs = list(filter(lambda e: e[0].real < 0, eigpairs))
        if len(eigpairs) == 0:
            # use random subspace instead of span{C} (with same dimensions)
            with rng:
                C = C.random(len(C), distribution='normal')
            continue
        # find shift with most impact on convergence
        maxval = -1
        maxind = 0
        for i in range(len(eigpairs)):
            eig = eigpairs[i][1]
            y_eig = eig[-len(U):]
            x_eig = eig[:len(U)]
            Ey = Ep.T.dot(y_eig)
            xEy = np.abs(np.dot(x_eig, Ey))
            currval = np.linalg.norm(y_eig)**2 / xEy
            if currval > maxval:
                maxval = currval
                maxind = i
        shift = eigpairs[maxind][0]
        # remove imaginary part if it is relatively small
        if np.abs(shift.imag) / np.abs(shift) < 1e-8:
            shift = shift.real
        return np.array([shift])
    raise RuntimeError('Could not generate initial shifts for low-rank RADI iteration.')


def hamiltonian_shifts(A, E, B, Rinv, RF, RC, K, Z, shift_options):
    """Compute further shift parameters for low-rank RADI iteration.

    Compute Galerkin projection of Hamiltonian matrix on space spanned by last few columns of
    :math:`Z` and return the eigenvalue of the projected Hamiltonian with the most impact on
    convergence as the next shift parameter.

    See :cite:`BBKS18`, pp. 318-321.

    Parameters
    ----------
    A
        The |Operator| A from the corresponding Riccati equation.
    E
        The |Operator| E from the corresponding Riccati equation.
    B
        The |VectorArray| B from the corresponding Riccati equation.
    RF
        A |VectorArray| representing the currently computed residual factor.
    RC
        A |NumPy array| representing the currently computed residual core.
    K
        A |VectorArray| representing the currently computed iterate.
    Z
        A |VectorArray| representing the currently computed solution factor.
    shift_options
        The shift options to use (see :func:`ricc_lrcf_solver_options`).

    Returns
    -------
    shifts
        A |NumPy array| containing a set of stable shift parameters.
    """
    l = shift_options['subspace_columns']
    # always use multiple of len(R) columns
    l = max(1, l // len(RF)) * len(RF)
    if len(Z) < l:
        l = len(Z)

    if RC is None:
        RC = np.eye(len(RF))

    U = gram_schmidt(Z[-l:], atol=0, rtol=0)
    Ap = A.apply2(U, U)
    BKp = U.inner(B) @ (Rinv @ U.inner(K).T)
    AAp = Ap - BKp
    UB = U.inner(B)
    Gp = UB.dot(Rinv @ UB.T)
    UR = U.inner(RF)
    Rp = UR.dot(RC @ UR.T)
    Hp = np.block([
        [AAp, Gp],
        [Rp, -AAp.T]
    ])
    Ep = E.apply2(U, U)
    EEp = spla.block_diag(Ep, Ep.T)
    eigvals, eigvecs = spla.eig(Hp, EEp)
    eigpairs = zip(eigvals, eigvecs)
    # filter stable eigenvalues
    eigpairs = list(filter(lambda e: e[0].real < 0, eigpairs))
    # find shift with most impact on convergence
    maxval = -1
    maxind = 0
    for i in range(len(eigpairs)):
        eig = eigpairs[i][1]
        y_eig = eig[-len(U):]
        x_eig = eig[:len(U)]
        Ey = Ep.T.dot(y_eig)
        xEy = np.abs(np.dot(x_eig, Ey))
        currval = np.linalg.norm(y_eig)**2 / xEy
        if currval > maxval:
            maxval = currval
            maxind = i
    shift = eigpairs[maxind][0]
    # remove imaginary part if it is relatively small
    if np.abs(shift.imag) / np.abs(shift) < 1e-8:
        shift = shift.real
    return np.array([shift])


def LDL_T_rank_truncation(L, D, tol=np.finfo(float).eps):
    """Compute a rank-truncated :math:'LDL^T' factorization.

    Computes the QR factorization :math:'Q R = L' of L followed by an
    eigendecomposition of :math:'RDR^T' and a rank decision on the absolute
    values of the computed eigenvalues. The truncated eigenpairs are dropped.
    The resulting core matrix (replacing D) is the diagonal matrix of preserved
    eigenvalues and the updated |VectorArray| is :math:'Q' times the
    preserved (left) eingenvectors.

    Parameters
    ----------
    L
        The |VectorArray| L from representing the left factor in the
        :math:'LDL^T' facorization.
    D
        The |NumPy array| representing the core factor.
    tol
        A float representing the desired relative truncation tolerance
        on the absolute values of the eigenvalues.
        Defaults to double precision machine epsilon

    Returns
    -------
    hL
        The |VectorArray| hL representing the left factor in the
        :math:'LDL^T' rank-truncated facorization.
    hD
        The |NumPy array| representing the core factor.
    """
    # QR decomposition of left factor
    Q, R = gram_schmidt(L, return_R=True)
    # Solve symmetric eigenvalue problem
    RDRT = R @ D @ R.T
    # ensure numerical symmetry
    RDRT = (RDRT+RDRT.T)/2.0
    S, U = spla.eigh(RDRT)

    # Thresholding based on tolerance
    r = np.abs(S) > tol * np.max(np.abs(S))

    # Filtering columns of V and elements of S based on r
    hL = Q.lincomb((U[:, r]).T)
    hD = np.diag(S[r])
    return hL, hD
