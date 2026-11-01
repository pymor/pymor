# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.solvers.matrix_equations.interface import BernoulliSolver, BernoulliSolverLR


class SignBernoulliSolver(BernoulliSolver):
    r"""Compute the solution of a |BernoulliEquation| using the matrix sign function.

    Computes a low-rank factor :math:`Z` with :class:`SignBernoulliSolverLR` and
    returns :math:`X = Z Z^H`.

    Parameters
    ----------
    maxiter
        The maximum amount of iterations.
    after_maxiter
        The number of iterations which are to be performed after tolerance is reached.
        This will improve the quality of the solution in cases where the iterates which are used
        by the stopping criterion stagnate prematurely.
    tol
        Tolerance for stopping criterion based on relative change of iterates.
    """

    def __init__(self, maxiter=100, after_maxiter=3, tol=1e-8):
        self.maxiter = maxiter
        self.after_maxiter = after_maxiter
        self.tol = tol

    def _solve(self, equation):
        solver = SignBernoulliSolverLR(maxiter=self.maxiter, after_maxiter=self.after_maxiter, tol=self.tol)
        Z = solver.solve(equation).to_numpy()
        return Z @ Z.conj().T


class SignBernoulliSolverLR(BernoulliSolverLR):
    r"""Compute a low-rank factor of the solution of a |BernoulliEquation|.

    Uses the matrix sign function method from :cite:`BBQ07`. It works with dense
    matrices, so it is only suitable for small problems.

    Parameters
    ----------
    maxiter
        The maximum amount of iterations.
    after_maxiter
        The number of iterations which are to be performed after tolerance is reached.
        This will improve the quality of the solution in cases where the iterates which are used
        by the stopping criterion stagnate prematurely.
    tol
        Tolerance for stopping criterion based on relative change of iterates.
    """

    def __init__(self, maxiter=100, after_maxiter=3, tol=1e-8):
        self.maxiter = maxiter
        self.after_maxiter = after_maxiter
        self.tol = tol

    def _solve(self, equation):
        A, E, B = equation.to_matrices()
        trans = equation.trans

        n = len(A)
        after_iter = 0

        assert n != 0

        if E is None:
            E = np.eye(n)

        if not trans:
            E = E.conj().T
            A = A.conj().T
            B = B.conj().T

        for i in range(self.maxiter):
            Aprev = A
            lu, piv = spla.lu_factor(A.conj().T)
            lndetA = np.sum(np.log(np.abs(np.diag(lu))))
            lndetE = np.linalg.slogdet(E)[1]
            c = np.exp((1./n)*(lndetA - lndetE))
            AinvTET = spla.lu_solve((lu, piv), E.conj().T)
            A = 0.5 * ((1 / c) * A + c * AinvTET.conj().T @ E)
            BT = (1 / np.sqrt(2 * c)) * np.vstack((B.conj().T, c * B.conj().T @ AinvTET))
            Q, R, perm = spla.qr(BT, mode='economic', pivoting=True)
            B = np.eye(n)[perm].T @ R.conj().T
            if after_iter > self.after_maxiter:
                break
            rnorm = spla.norm(A - Aprev) / spla.norm(A)
            self.logger.info(f'Relative change of iterates at step {i}: {rnorm:.5e}')
            if rnorm <= self.tol:
                after_iter += 1

        if rnorm > self.tol:
            self.logger.warning(f'Prescribed tolerance for relative change of iterates not achieved '
                                f'({rnorm:e} > {self.tol:e}) after {self.maxiter} steps.')

        Q, R, _ = spla.qr(E.conj() - A.conj(), pivoting=True)
        nsp_rk = 0
        for r in R:
            if np.allclose(r, np.zeros(r.shape)):
                nsp_rk = nsp_rk + 1
        Q = Q[:, n-nsp_rk:].conj()
        _, R = spla.qr(B.conj().T @ Q, mode='economic')
        Y = spla.solve_triangular(R, np.sqrt(2) * Q.conj().T, trans='C')

        return equation.A.source.from_numpy(Y.conj().T)
