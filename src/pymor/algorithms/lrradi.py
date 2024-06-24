# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.to_matrix import to_matrix
from pymor.operators.numpy import NumpyMatrixOperator
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


def solve_ricc_lrcf(A, E, B, C, R=None, Q=None, S=None, trans=False, options=None):
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
    Q
        The matrix Q as a 2D |NumPy array| or `None`.
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
    _solve_ricc_check_args(A, E, B, C, R, Q, S, trans)

    if S is not None:
        if R is not None:
            Rinv = spla.solve(R, np.eye(R.shape[0]))
        else:
            R = Rinv = np.eye(len(B) if trans else len(C))

        BRinvSt = NumpyMatrixOperator(B.to_numpy().T @ Rinv @ S.to_numpy())
        BRinvSt2 = LowRankOperator(B, Rinv, S)
        RHS = BRinvSt.source.from_numpy(np.eye(BRinvSt.source.dim))
        V1 = BRinvSt.apply(RHS)
        V2 = BRinvSt2.apply(RHS)
        assert np.max((V1-V2).norm()) < 1e-10
        tA = (A - BRinvSt).assemble()
        assert isinstance(tA, NumpyMatrixOperator)
        tC = cat_arrays([C, S])
        if Q is None:
            tQ = spla.block_diag(np.eye(len(C)), -Rinv)
        else:
            tQ = spla.block_diag(Q, -Rinv)
        Z_cf = solve_ricc_lrcf(tA, E, B, tC, R, tQ, None, trans, options)
        return Z_cf

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
        if not trans:
            AsE = A + shifts[j_shift] * E
        else:
            AsE = A + np.conj(shifts[j_shift]) * E
        if j == 0:
            if not trans:
                V = AsE.apply_inverse(RF) * np.sqrt(-2 * shifts[j_shift].real)
            else:
                V = AsE.apply_inverse_adjoint(RF) * np.sqrt(-2 * shifts[j_shift].real)
        else:
            BRiK = LowRankOperator(B, Rinv, K)
            AsEBRiK = AsE - BRiK

            if not trans:
                V = AsEBRiK.apply_inverse(RF) * np.sqrt(-2 * shifts[j_shift].real)
            else:
                V = AsEBRiK.apply_inverse_adjoint(RF) * np.sqrt(-2 * shifts[j_shift].real)

        if np.imag(shifts[j_shift]) == 0:
            V = V.real
            Z.append(V)
            VB = V.inner(B)
            Yt = RC - (VB @ Rinv @ VB.T) / (2 * shifts[j_shift].real)
            Y = spla.block_diag(Y, Yt)
            if not trans:
                EVYt = E.apply(V).lincomb(spla.inv(Yt))
            else:
                EVYt = E.apply_adjoint(V).lincomb(spla.inv(Yt))
            RF.axpy(np.sqrt(-2*shifts[j_shift].real), EVYt)
            K += EVYt.lincomb(VB.T)
            j += 1
        else:
            Z.append(V.real)
            Z.append(V.imag)
            Vr = V.real.inner(B)
            Vi = V.imag.inner(B)
            sa = np.abs(shifts[j_shift])
            F1 = np.vstack((
                -shifts[j_shift].real/sa * Vr - shifts[j_shift].imag/sa * Vi,
                shifts[j_shift].imag/sa * Vr - shifts[j_shift].real/sa * Vi
            ))
            F2 = np.vstack((
                Vr,
                Vi
            ))
            F3 = np.vstack((
                shifts[j_shift].imag/sa * np.eye(len(RF)),
                shifts[j_shift].real/sa * np.eye(len(RF))
            ))
            Yt = spla.block_diag(RC, 0.5 * RC) \
                - (F1 @ Rinv @ F1.T) / (4 * shifts[j_shift].real)  \
                - (F2 @ Rinv @ F2.T) / (4 * shifts[j_shift].real)  \
                - (F3 @ RC @ F3.T) / 2
            Y = spla.block_diag(Y, Yt)
            if not trans:
                EVYt = E.apply(cat_arrays([V.real, V.imag])).lincomb(spla.inv(Yt))
            else:
                EVYt = E.apply_adjoint(cat_arrays([V.real, V.imag])).lincomb(spla.inv(Yt))
            RF.axpy(np.sqrt(-2 * shifts[j_shift].real), EVYt[:len(C)])
            K += EVYt.lincomb(F2.T)
            j += 2
        j_shift += 1
        res = np.linalg.norm(RF.gramian() @ RC, ord='fro')
        logger.info(f'Relative residual at step {j}: {res/init_res:.5e}')
        if j_shift >= shifts.size:
            shifts = iteration_shifts(A, E, B, Rinv, RF, RC, K, Z, shift_options)
            j_shift = 0
    # transform solution to lrcf
    cf = spla.cholesky(Y)
    Z_cf = Z.lincomb(spla.solve_triangular(cf, np.eye(len(Z))).T)
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
    _solve_ricc_check_args(A, E, B, C, R, None, S, trans)
    options = _parse_options(options, ricc_lrcf_solver_options(), 'lrradi', None, False)
    if options['type'] != 'lrradi':
        raise ValueError(f"Unexpected positive Riccati equation solver ({options['type']}).")

    if R is None:
        R = np.eye(len(C) if not trans else len(B))
    return solve_ricc_lrcf(A, E, B, C, -R, None, S, trans, options)


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
    KBp = U.inner(K) @ (Rinv @ U.inner(B).T)
    AAp = Ap - KBp
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
