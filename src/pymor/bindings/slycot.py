# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('SLYCOT')


import numpy as np
import scipy.linalg as spla
import slycot

from pymor.core.logger import getLogger
from pymor.solvers.matrix.interface import (
    LyapunovSolver,
    LyapunovSolverLRCF,
    PositiveRiccatiSolver,
    PositiveRiccatiSolverLRCF,
    RiccatiSolver,
    RiccatiSolverLRCF,
)
from pymor.solvers.matrix.utils import _chol


class SlycotLyapunovSolver(LyapunovSolver):
    """Compute the dense solution of a |LyapunovEquation| using slycot.

    Uses `slycot.sb03md57` (if E is `None`) or `slycot.sg03ad` (if E is not `None`),
    which are based on the Bartels-Stewart algorithm.

    This solver has no tunable parameters.
    """

    def _solve(self, equation):
        A, E, B = equation._dense_args()
        trans = equation.trans
        cont_time = equation.cont_time

        n = A.shape[0]
        C = -B.dot(B.T) if not trans else -B.T.dot(B)
        trana = 'T' if not trans else 'N'
        dico = 'C' if cont_time else 'D'
        job = 'B'
        if E is None:
            _, _, X, scale, sep, ferr, _ = slycot.sb03md57(A, C=C, dico=dico, job=job, trana=trana)
            _solve_check(A.dtype, 'slycot.sb03md57', sep, ferr)
        else:
            fact = 'N'
            uplo = 'L'
            Q = np.zeros((n, n))
            Z = np.zeros((n, n))
            A, E = A.copy(), E.copy()  # avoid overwriting (see #2168)
            _, _, _, _, X, scale, sep, ferr, _, _, _ = slycot.sg03ad(dico, job, fact, trana, uplo,
                                                                     n, A, E,
                                                                     Q, Z, C)
            _solve_check(A.dtype, 'slycot.sg03ad', sep, ferr)
        X /= scale
        return X


class SlycotLyapunovSolverLRCF(LyapunovSolverLRCF):
    r"""Compute a low-rank Cholesky factor of a |LyapunovEquation| using slycot.

    Computes the dense solution :math:`X` with :class:`SlycotLyapunovSolver` and
    factorizes it.

    This solver has no tunable parameters.
    """

    def _solve(self, equation):
        X = SlycotLyapunovSolver().solve(equation)
        return equation.A.source.from_numpy(_chol(X))


class SlycotRiccatiSolver(RiccatiSolver):
    """Compute the dense solution of a |RiccatiEquation| using slycot.

    Uses `slycot.sb02md` (if E is `None` and S is `None`), which is based on the Schur
    vector approach, or `slycot.sb02od` (if E is `None` and S is not `None`) or
    `slycot.sg02ad` (if E is not `None`), which are both based on the method of
    deflating subspaces.

    This solver has no tunable parameters.
    """

    def _solve(self, equation):
        A, E, B, C, R, S = equation._dense_args()
        trans = equation.trans

        dico = 'C'
        n = A.shape[0]
        if E is not None:
            jobb = 'B'
            fact = 'C'
            uplo = 'U'
            jobl = 'Z' if S is None else 'N'
            scal = 'N'
            sort = 'S'
            acc = 'R'
            m = C.shape[0] if not trans else B.shape[1]
            p = B.shape[1] if not trans else C.shape[0]
            if R is None:
                R = np.eye(m)
            if S is None:
                S = np.empty((n, m))
            elif not trans:
                S = S.T
            if not trans:
                A = A.T
                E = E.T
                B, C = C.T, B.T
            out = slycot.sg02ad(dico, jobb, fact, uplo, jobl, scal, sort, acc,
                                n, m, p,
                                A, E, B, C, R, S)
            X = out[1]
            rcond = out[0]
            _ricc_rcond_check('slycot.sg02ad', rcond)
        elif S is not None:
            m = C.shape[0] if not trans else B.shape[1]
            p = B.shape[1] if not trans else C.shape[0]
            if R is None:
                R = np.eye(m)
            else:
                R = R.copy()  # fix overwrite issue (#2200)
            S = S.copy()  # fix overwrite issue (#2200)
            if trans:
                C = C.copy()  # fix overwrite issue (#2200)
                X, rcond = slycot.sb02od(n, m, A, B, C, R, dico, p=p, L=S, fact='C')[:2]
            else:
                B = B.copy()  # fix overwrite issue (#2200)
                X, rcond = slycot.sb02od(n, m, A.T, C.T, B.T, R, dico, p=p, L=S.T, fact='C')[:2]
            _ricc_rcond_check('slycot.sb02od', rcond)
        else:
            if trans:
                if R is None:
                    G = B @ B.T
                else:
                    G = B @ spla.solve(R, B.T)
                Q = C.T @ C
                X, rcond = slycot.sb02md(n, A, G, Q, dico)[:2]
            else:
                if R is None:
                    G = C.T @ C
                else:
                    G = C.T @ spla.solve(R, C)
                Q = B @ B.T
                X, rcond = slycot.sb02md(n, A.T, G, Q, dico)[:2]
            _ricc_rcond_check('slycot.sb02md', rcond)

        return X


class SlycotRiccatiSolverLRCF(RiccatiSolverLRCF):
    r"""Compute a low-rank Cholesky factor of a |RiccatiEquation| using slycot.

    Computes the dense solution :math:`X` with :class:`SlycotRiccatiSolver` and
    factorizes it.

    This solver has no tunable parameters.
    """

    def _solve(self, equation):
        X = SlycotRiccatiSolver().solve(equation)
        return equation.A.source.from_numpy(_chol(X))


class SlycotPositiveRiccatiSolver(PositiveRiccatiSolver):
    """Compute the dense solution of a |PositiveRiccatiEquation| using slycot.

    The positive Riccati equation differs from the |RiccatiEquation| only in the sign of
    the quadratic term, so it is solved by :class:`SlycotRiccatiSolver` with :math:`R`
    negated.  When :math:`R` is `None` it is materialized as the identity first --
    passing `None` on would let the ordinary Riccati solver default to :math:`+I`.

    This solver has no tunable parameters.
    """

    def _solve(self, equation):
        R = equation.R
        if R is None:
            R = np.eye(len(equation.C) if not equation.trans else len(equation.B))
        temp_equation = equation.with_(R=-R)
        return SlycotRiccatiSolver().solve(temp_equation)


class SlycotPositiveRiccatiSolverLRCF(PositiveRiccatiSolverLRCF):
    r"""Compute a low-rank Cholesky factor of a |PositiveRiccatiEquation| using slycot.

    Computes the dense solution :math:`X` with :class:`SlycotPositiveRiccatiSolver` and
    factorizes it.  The factorization assumes :math:`X \succcurlyeq 0`.

    This solver has no tunable parameters.
    """

    def _solve(self, equation):
        X = SlycotPositiveRiccatiSolver()._solve(equation)
        return equation.A.source.from_numpy(_chol(X))


def _solve_check(dtype, solver, sep, ferr):
    if ferr > 1e-1:
        logger = getLogger(solver)
        logger.warning(f'Estimated forward relative error bound is large (ferr={ferr:e}, sep={sep:e}). '
                       f'Result may not be accurate.')



def _ricc_rcond_check(solver, rcond):
    if rcond < np.finfo(np.float64).eps:
        logger = getLogger(solver)
        logger.warning(f'Estimated reciprocal condition number is small (rcond={rcond:e}). '
                       f'Result may not be accurate.')
