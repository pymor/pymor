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

        self.__auto_init(locals())
        super().__init__()

    def _solve(self, equation):
        A, E, B, C, R, S, Q = equation.A, equation.E, equation.B, equation.C, equation.R, equation.S, equation.Q
        trans = equation.trans

        if S is None:
            if trans:
                Z_lr = self._solve_impl(A, E, B, C, R, Q, trans)
            else:
                Z_lr = self._solve_impl(A, E, B, C, Q, R, trans)

        else:
            if R is not None:
                Rinv = spla.solve(R, np.eye(R.shape[0]))
            else:
                R = Rinv = np.eye(len(B) if trans else len(C))

            if trans:
                BRinvSt = LowRankOperator(B, Rinv, S)
                tA = A - BRinvSt
                tC = cat_arrays([C, S])
                tQ = spla.block_diag(Q, -Rinv)

                Z_lr = self._solve_impl(tA, E, B, tC, R, tQ, trans)
            else:
                SRinvCt = LowRankOperator(S, Rinv, C)
                tA = A - SRinvCt
                tB = cat_arrays([B, S])
                tR = spla.block_diag(Q, -Rinv)

                Z_lr = self._solve_impl(tA, E, tB, C, tR, R, trans)

        return Z_lr


    def _solve_impl(self, A, E, B, C, R=None, Q=None, trans=False):

        if self.radi_shifts == 'hamiltonian_shifts':
            init_shifts = self.hamiltonian_shifts_init
            iteration_shifts = self.hamiltonian_shifts
        else:
            raise ValueError('Unknown radi shift strategy.')

        solver = self.shifted_system_solver

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

        Z = A.source.empty(reserve=len(C) * self.radi_maxiter)
        Y = np.empty((0, 0))

        K = A.source.zeros(len(B))
        RF = C.copy()
        RC = np.eye(len(C)) if Q is None else Q

        j = 0
        j_shift = 0
        shifts = init_shifts(A, E, B, C, Rinv, Q)

        if Q is None:
            res = np.linalg.norm(RF.gramian(), ord=2)
        else:
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

            AsEBRiK = (AsE - BRiK).assemble() # assemble combines the two low-rank
                                              # updates into a single one if A came
                                              # in as a LowRankUpdatedOperator already
                                              # (avoids recursive Sherman-Morrison-Woodburry)

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
        cf = spla.cholesky(Y)
        Z_lr = Z.lincomb(spla.solve_triangular(cf, np.eye(len(Z))))
        return Z_lr


    def hamiltonian_shifts_init(self, A, E, B, C, Rinv, Q):
        """Compute initial shift parameters for low-rank RADI iteration.

        Compute Galerkin projection of Hamiltonian matrix on space spanned by :math:`C` and return
        the eigenvalue of the projected Hamiltonian with the most impact on convergence as the
        next shift parameter.

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
        Rinv

        Q

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
            eigpairs = zip(eigvals, eigvecs, strict=True)
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

        RF

        RC

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
        eigpairs = zip(eigvals, eigvecs, strict=True)
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


class RADIPositiveRealRiccatiSolver(PositiveRiccatiSolverLR):
    def _solve(self, equation):
        A, E, B, C, R, S, Q = equation.A, equation.E, equation.B, equation.C, equation.R, equation.S, equation.Q
        trans = equation.trans

        if S is None:
            if trans:
                Z_lr = RADIRiccatiSolver._solve_impl(A, E, B, C, -R, Q, trans)
            else:
                Z_lr = RADIRiccatiSolver._solve_impl(A, E, B, C, Q, -R, trans)

        else:
            if R is not None:
                Rinv = spla.solve(R, np.eye(R.shape[0]))
            else:
                R = Rinv = np.eye(len(B) if trans else len(C))

            if trans:
                BRinvSt = LowRankOperator(B, Rinv, S)
                tA = A + BRinvSt
                tC = cat_arrays([C, S])
                tQ = spla.block_diag(Q, Rinv)

                Z_lr = RADIRiccatiSolver._solve_impl(tA, E, B, tC, -R, tQ, trans)
            else:
                SRinvCt = LowRankOperator(S, Rinv, C)
                tA = A + SRinvCt
                tB = cat_arrays([B, S])
                tR = spla.block_diag(Q, Rinv)

                Z_lr = RADIRiccatiSolver._solve_impl(tA, E, tB, C, tR, -R, trans)

        return Z_lr
