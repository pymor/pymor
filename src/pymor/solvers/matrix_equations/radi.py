# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.defaults import defaults
from pymor.operators.constructions import IdentityOperator, LowRankOperator
from pymor.solvers.matrix_equations.interface import PositiveRiccatiSolverLR, RiccatiSolverLR
from pymor.tools.random import new_rng
from pymor.vectorarrays.constructions import cat_arrays


class RADIRiccatiSolver(RiccatiSolverLR):
    r"""Compute an approximate low-rank factor of the solution of a |RiccatiEquation|.

    This is an implementation of Algorithm 2 in :cite:`BBKS18`.

    Parameters
    ----------
    radi_tol
        Convergence tolerance for the RADI iteration.
    radi_maxiter
        Maximum number of RADI steps. A real shift counts as one step, a
        complex-conjugate shift pair as two.
    radi_shifts
         Strategy for computing the RADI shift parameters. Currently only
        ``'hamiltonian_shifts'`` is supported.
    shifted_system_solver
        The |Solver| for the shifted systems.
    hamiltonian_shifts_init_maxiter
        Maximum number of attempts to generate stable initial shifts before an error is raised.
        See :meth:`hamiltonian_shifts_init`.
    hamiltonian_shifts_subspace_columns
        Number of trailing columns of the solution factor :math:`Z` used to span the
        Galerkin subspace for the subsequent shifts. See :meth:`hamiltonian_shifts`.
    """

    @defaults('radi_tol', 'radi_maxiter', 'radi_shifts', 'shifted_system_solver',
              'hamiltonian_shifts_init_maxiter', 'hamiltonian_shifts_subspace_columns')
    def __init__(self, radi_tol=1e-10, radi_maxiter=500, radi_shifts='hamiltonian_shifts',
                 shifted_system_solver=None, hamiltonian_shifts_init_maxiter=20,
                 hamiltonian_shifts_subspace_columns=6):

        super().__init__()
        self.radi_tol = radi_tol
        self.radi_maxiter = radi_maxiter
        self.radi_shifts = radi_shifts
        self.shifted_system_solver = shifted_system_solver
        self.hamiltonian_shifts_init_maxiter = hamiltonian_shifts_init_maxiter
        self.hamiltonian_shifts_subspace_columns = hamiltonian_shifts_subspace_columns

    def _solve(self, equation):
        A, E, B, C, R, Q, S = equation.A, equation.E, equation.B, equation.C, equation.R, equation.Q, equation.S
        trans = equation.trans

        if R is None:
            R = np.eye(len(B))
        if Q is None:
            Q = np.eye(len(C))

        if S is None:
            if trans:
                Z_lr = self._solve_impl(A, E, B, C, R, Q, trans)
            else:
                Z_lr = self._solve_impl(A, E, C, B, Q, R, trans)

        else:
            if trans:
                Rinv = spla.solve(R, np.eye(R.shape[0]))
                BRinvSt = LowRankOperator(B, Rinv, S)
                tA = A - BRinvSt
                tC = cat_arrays([C, S])
                tQ = spla.block_diag(Q, -Rinv)

                Z_lr = self._solve_impl(tA, E, B, tC, R, tQ, trans)
            else:
                Qinv = spla.solve(Q, np.eye(Q.shape[0]))
                SQinvCt = LowRankOperator(S, Qinv, C)
                tA = A - SQinvCt
                tB = cat_arrays([B, S])
                tR = spla.block_diag(R, -Qinv)

                Z_lr = self._solve_impl(tA, E, C, tB, Q, tR, trans)

        return Z_lr


    def _solve_impl(self, A, E, B, C, R, Q, trans):
        if self.radi_shifts == 'hamiltonian_shifts':
            init_shifts = self.hamiltonian_shifts_init
            iteration_shifts = self.hamiltonian_shifts
        else:
            raise ValueError('Unknown radi shift strategy.')

        solver = self.shifted_system_solver

        if E is None:
            E = IdentityOperator(A.source)

        if R is not None:
            Rinv = spla.solve(R, np.eye(R.shape[0]))
            Rinv = 0.5 * (Rinv + Rinv.T)
        else:
            R = Rinv = np.eye(len(B))

        Z = A.source.empty(reserve=len(C) * self.radi_maxiter)
        Y = np.empty((0, 0))

        K = A.source.zeros(len(B))
        RF = C.copy()
        RC = Q.copy()

        j = 0
        j_shift = 0
        shifts = init_shifts(A, E, B, C, Rinv, Q)

        res = np.linalg.norm(RF.gramian() @ RC, ord=2)
        init_res = res
        Ctol = res * self.radi_tol

        while res > Ctol and j < self.radi_maxiter:
            s = shifts[j_shift]
            sr = s.real
            si = s.imag
            sa = np.abs(s)
            alpha = np.sqrt(-2.0 * sr)

            if not trans:
                AsE = A + s * E
            else:
                AsE = A + np.conj(s) * E

            Im = np.eye(len(B))
            BRiK = LowRankOperator(B, Im, K) if trans else LowRankOperator(K, Im, B)

            # assemble combines the two low-rank updates into a single one if A came
            # in as a LowRankUpdatedOperator already (avoids recursive Sherman-Morrison-Woodburry)
            AsEBRiK = (AsE - BRiK).assemble()

            if not trans:
                V = AsEBRiK.apply_inverse(RF, solver=solver)
            else:
                V = AsEBRiK.apply_inverse_adjoint(RF, solver=solver)

            V = V.lincomb(RC.T)

            if np.imag(shifts[j_shift]) == 0:
                V = alpha * V
                Z.append(V)
                VB = V.inner(B)
                Yt = RC + (VB @ Rinv @ VB.T) / alpha**2
                Y = spla.block_diag(Y, Yt)
                if not trans:
                    EVYt = E.apply(V).lincomb(spla.inv(Yt).T)
                else:
                    EVYt = E.apply_adjoint(V).lincomb(spla.inv(Yt).T)
                RF.axpy(alpha, EVYt)
                K += EVYt.lincomb(VB @ Rinv)
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
                    EVYt = E.apply(cat_arrays([V1, V2])).lincomb(spla.inv(Yt).T)
                else:
                    EVYt = E.apply_adjoint(cat_arrays([V1, V2])).lincomb(spla.inv(Yt).T)
                RF.axpy(alpha, EVYt[:len(C)])
                K += EVYt.lincomb(F2 @ Rinv)
                j += 2
            j_shift += 1
            res = np.linalg.norm(RF.gramian() @ RC, ord=2)
            self.logger.info(f'Relative residual at step {j}: {res/init_res:.5e}')
            if j_shift >= shifts.size:
                shifts = iteration_shifts(A, E, B, Rinv, RF, RC, K, Z)
                j_shift = 0
        # transform solution to low-rank factor
        Yinv = spla.inv(Y)
        Yinv = (Yinv + Yinv.T) / 2.0
        Z_lr, S = self.LDL_T_rank_truncation(Z, Yinv)
        S = np.diag(np.sqrt(np.diag(S)))
        Z_lr = Z_lr.lincomb(S)
        return Z_lr

    def LDL_T_rank_truncation(self, L, D, tol=np.finfo(float).eps):
        r"""Computes a rank-truncated :math:`LDL^T` factorization.

        Computes the QR factorization of :math:`L = QR` followed by an
        eigendecomposition of :math:'RDR^T' and a rank decision based on the absolute
        values of the computed eigenvalues. The truncated core matrix is the diagonal
        matrix of preserved eigenvalues and the truncated :math:`L` is computed as
        :math:`Q` times the preserved (left) eingenvectors.

        Parameters
        ----------
        L
            The |VectorArray| representing the left factor in the
            :math:`LDL^\top` facorization.
        D
            The |NumPy array| representing the core factor.
        tol
            The relative truncation tolerance.

        Returns
        -------
        hL
            The |VectorArray| hL representing the left factor in the
            :math:`LDL^\top` rank-truncated facorization.
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
        hL = Q.lincomb(U[:, r])
        hD = np.diag(S[r])
        return hL, hD


    def hamiltonian_shifts_init(self, A, E, B, C, Rinv, Q):
        """Compute initial shift parameters for low-rank RADI iteration.

        Compute Galerkin projection of Hamiltonian matrix on space spanned by :math:`C` and return
        the eigenvalue of the projected Hamiltonian with the most impact on convergence as the
        next shift parameter.

        See :cite:`BBKS18`, pp. 318-321.

        Parameters
        ----------
        A
            The |Operator| A from the corresponding |RiccatiEquation|.
        E
            The |Operator| E from the corresponding |RiccatiEquation|.
        B
            The |VectorArray| B from the corresponding |RiccatiEquation|.
        C
            The |VectorArray| C from the corresponding |RiccatiEquation|.
        Rinv
            The matrix :math:`R^{-1}` as a |NumPy array| from the corresponding |RiccatiEquation|.
        Q
            The matrix :math:`Q` as a |NumPy array| from the corresponding |RiccatiEquation|.

        Returns
        -------
        shifts
            A |NumPy array| containing a set of stable shift parameters.
        """
        rng = new_rng(0)
        for _ in range(self.hamiltonian_shifts_init_maxiter):
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
            eigpairs = zip(eigvals, eigvecs.T, strict=True)
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


    def hamiltonian_shifts(self, A, E, B, Rinv, RF, RC, K, Z):
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
        Rinv
            The matrix :math:`R^{-1}` as a |NumPy array| from the corresponding |RiccatiEquation|.
        RF
            A |VectorArray| representing the currently computed residual factor.
        RC
            A |NumPy array| representing the currently computed residual core.
        K
            A |VectorArray| representing the currently computed iterate.
        Z
            A |VectorArray| representing the currently computed solution factor.

        Returns
        -------
        shifts
            A |NumPy array| containing a set of stable shift parameters.
        """
        l = self.hamiltonian_shifts_subspace_columns
        # always use multiple of len(R) columns
        l = max(1, l // len(RF)) * len(RF)
        if len(Z) < l:
            l = len(Z)

        if RC is None:
            RC = np.eye(len(RF))

        U = gram_schmidt(Z[-l:], atol=0, rtol=0)
        Ap = A.apply2(U, U)
        BKp = U.inner(K) @ (U.inner(B).T)
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
        eigpairs = zip(eigvals, eigvecs.T, strict=True)
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


class RADIPositiveRiccatiSolver(PositiveRiccatiSolverLR):
    r"""Compute an approximate low-rank factor of the solution of a |PositiveRiccatiEquation|.

    Calls :class:`pymor.solvers.matrix_equations.radi.RADIRiccatiSolver` with flipped signs in
    the `R` or `Q` terms (depending whether the transposed version is solved or not).

    Parameters
    ----------
    radi_tol
        Convergence tolerance for the RADI iteration.
    radi_maxiter
        Maximum number of RADI steps. A real shift counts as one step, a
        complex-conjugate shift pair as two.
    radi_shifts
            Strategy for computing the RADI shift parameters. Currently only
        ``'hamiltonian_shifts'`` is supported.
    shifted_system_solver
        The |Solver| for the shifted systems.
    hamiltonian_shifts_init_maxiter
        Maximum number of attempts to generate stable initial shifts before an error is raised.
        See :meth:`hamiltonian_shifts_init`.
    hamiltonian_shifts_subspace_columns
        Number of trailing columns of the solution factor :math:`Z` used to span the
        Galerkin subspace for the subsequent shifts. See :meth:`hamiltonian_shifts`.
    """

    @defaults('radi_tol', 'radi_maxiter', 'radi_shifts', 'shifted_system_solver',
                'hamiltonian_shifts_init_maxiter', 'hamiltonian_shifts_subspace_columns')
    def __init__(self, radi_tol=1e-10, radi_maxiter=500, radi_shifts='hamiltonian_shifts',
                shifted_system_solver=None, hamiltonian_shifts_init_maxiter=20,
                hamiltonian_shifts_subspace_columns=6):

        self._radi_solver = RADIRiccatiSolver(radi_tol=radi_tol, radi_maxiter=radi_maxiter, radi_shifts=radi_shifts,
                                              shifted_system_solver=shifted_system_solver,
                                              hamiltonian_shifts_init_maxiter=hamiltonian_shifts_init_maxiter,
                                              hamiltonian_shifts_subspace_columns=hamiltonian_shifts_subspace_columns)
        super().__init__()
        self.radi_tol = radi_tol
        self.radi_maxiter = radi_maxiter
        self.radi_shifts = radi_shifts
        self.shifted_system_solver = shifted_system_solver
        self.hamiltonian_shifts_init_maxiter = hamiltonian_shifts_init_maxiter
        self.hamiltonian_shifts_subspace_columns = hamiltonian_shifts_subspace_columns

    def _solve(self, equation):
        A, E, B, C, R, S, Q = equation.A, equation.E, equation.B, equation.C, equation.R, equation.S, equation.Q
        trans = equation.trans

        if R is None:
            R = np.eye(len(B))
        if Q is None:
            Q = np.eye(len(C))

        if S is None:
            if trans:
                Z_lr = self._radi_solver._solve_impl(A, E, B, C, -R, Q, trans)
            else:
                Z_lr = self._radi_solver._solve_impl(A, E, C, B, -Q, R, trans)

        else:
            if trans:
                Rinv = spla.solve(R, np.eye(R.shape[0]))
                BRinvSt = LowRankOperator(B, Rinv, S)
                tA = A + BRinvSt
                tC = cat_arrays([C, S])
                tQ = spla.block_diag(Q, Rinv)

                Z_lr = self._radi_solver._solve_impl(tA, E, B, tC, -R, tQ, trans)
            else:
                Qinv = spla.solve(Q, np.eye(Q.shape[0]))
                SQinvCt = LowRankOperator(S, Qinv, C)
                tA = A + SQinvCt
                tB = cat_arrays([B, S])
                tR = spla.block_diag(R, Qinv)

                Z_lr = self._radi_solver._solve_impl(tA, E, C, tB, -Q, tR, trans)

        return Z_lr
