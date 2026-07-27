# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import scipy.linalg as spla

from pymor.algorithms.to_matrix import to_matrix
from pymor.operators.interface import Operator
from pymor.solvers.matrix.interface import SylvesterSolver


class SylvesterSchurSolver(SylvesterSolver):
    r"""Compuates the solution of a |SylvesterEquation|.

    Solves the |SylvesterEquation| by (generalized) Schur decomposition
    (Algorithms 3 and 4 in :cite:`BKS11`), if the necessary parameters are given.

    Parameters
    ----------
    shifted_system_solver
        The |Solver| for the shifted systems.
    """

    def __init__(self, shifted_system_solver=None):
        self.shifted_system_solver = shifted_system_solver

    def _solve(self, equation):
        A = equation.A
        Ar = equation.Ar
        E = equation.E or None
        Er = equation.Er or None
        B = equation.B or None
        Br = equation.Br or None
        C = equation.C or None
        Cr = equation.Cr or None

        compute_V = B is not None and Br is not None
        compute_W = C is not None and Cr is not None

        if not compute_V and not compute_W:
            raise ValueError('Not enough parameters are given to solve a Sylvester equation.')

        if compute_V:
            assert isinstance(B, Operator)
            assert B.linear
            assert B.range == A.source
            assert isinstance(Br, Operator)
            assert Br.linear
            assert Br.range == Ar.source
            assert B.source == Br.source

        if compute_W:
            assert isinstance(C, Operator)
            assert C.linear
            assert C.source == A.source
            assert isinstance(Cr, Operator)
            assert Cr.linear
            assert Cr.source == Ar.source
            assert C.range == Cr.range

        # convert reduced operators
        Ar = to_matrix(Ar, format='dense')
        r = Ar.shape[0]
        if Er is not None:
            Er = to_matrix(Er, format='dense')

        # (Generalized) Schur decomposition
        if Er is None:
            TAr, Z = spla.schur(Ar, output='complex')
            Q = Z
        else:
            TAr, TEr, Q, Z = spla.qz(Ar, Er, output='complex')

        # solve for V, from the last column to the first
        if compute_V:
            V = A.source.empty(reserve=r)

            BrTQ = Br.apply_adjoint(Br.range.from_numpy(Q))
            BBrTQ = B.apply(BrTQ)
            for i in range(-1, -r - 1, -1):
                rhs = -BBrTQ[i].copy()
                if i < -1:
                    if Er is not None:
                        rhs -= A.apply(V.lincomb(TEr[i, :i:-1].conjugate().T))
                    rhs -= E.apply(V.lincomb(TAr[i, :i:-1].conjugate().T))
                TErii = 1 if Er is None else TEr[i, i]
                eAaE = TErii.conjugate() * A + TAr[i, i].conjugate() * E
                V.append(eAaE.apply_inverse(rhs, solver=self.shifted_system_solver))

            V = V.lincomb(Z.conjugate()[:, ::-1].T)
            V = V.real

        # solve for W, from the first column to the last
        if compute_W:
            W = A.source.empty(reserve=r)

            CrZ = Cr.apply(Cr.source.from_numpy(Z))
            CTCrZ = C.apply_adjoint(CrZ)
            for i in range(r):
                rhs = -CTCrZ[i].copy()
                if i > 0:
                    if Er is not None:
                        rhs -= A.apply_adjoint(W.lincomb(TEr[:i, i].T))
                    rhs -= E.apply_adjoint(W.lincomb(TAr[:i, i].T))
                TErii = 1 if Er is None else TEr[i, i]
                eAaE = TErii.conjugate() * A + TAr[i, i].conjugate() * E
                W.append(eAaE.apply_inverse_adjoint(rhs, solver=self.shifted_system_solver))

            W = W.lincomb(Q.conjugate().T)
            W = W.real

        if compute_V and compute_W:
            return V, W
        elif compute_V:
            return V
        else:
            return W
