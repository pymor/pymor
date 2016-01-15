# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse as sps

from pymor.operators.interfaces import OperatorInterface
from pymor.operators.constructions import IdentityOperator, LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorArray

import pymess


class LyapunovEquation(pymess.equation):
    """Lyapunov equation class for pymess

    Represents a Lyapunov equation::

        A * X + X * A^T + RHS * RHS^T = 0

    if E is None, otherwise a generalized Lyapunov equation::

        A * X * E^T + E * X * A^T + RHS * RHS^T = 0.

    For the dual Lyapunov equation, opt.type needs to be pymess.MESS_OP_TRANSPOSE.

    Parameters
    ----------
    opt
        pymess options structure
    A
        The |Operator| A.
    E
        The |Operator| E or None.
    RHS
        The |Operator| RHS.
    """
    def __init__(self, opt, A, E, RHS):
        dim = A.source.dim

        super(LyapunovEquation, self).__init__(name='lyap_eqn', opt=opt, dim=dim)

        if isinstance(RHS, NumpyMatrixOperator):
            if RHS.sparse:
                self.RHS = RHS._matrix.toarray()
            else:
                self.RHS = RHS._matrix
        else:
            if opt.type == pymess.MESS_OP_NONE:
                eye = NumpyVectorArray(sps.eye(RHS.source.dim))
                self.RHS = np.array(RHS.apply(eye).data.T)
            else:
                eye = NumpyVectorArray(sps.eye(RHS.range.dim))
                self.RHS = np.array(RHS.apply_adjoint(eye).data)

        self.A = A
        self.E = E
        self.p = []

    def A_apply(self, op, y):
        y = NumpyVectorArray(np.array(y).T)
        if op == pymess.MESS_OP_NONE:
            x = self.A.apply(y)
        else:
            x = self.A.apply_adjoint(y)
        return np.matrix(x.data).T

    def E_apply(self, op, y):
        if self.E is None:
            return y

        y = NumpyVectorArray(np.array(y).T)
        if op == pymess.MESS_OP_NONE:
            x = self.E.apply(y)
        else:
            x = self.E.apply_adjoint(y)
        return np.matrix(x.data).T

    def As_apply(self, op, y):
        y = NumpyVectorArray(np.array(y).T)
        if op == pymess.MESS_OP_NONE:
            x = self.A.apply_inverse(y)
        else:
            x = self.A.apply_inverse_adjoint(y)
        return np.matrix(x.data).T

    def Es_apply(self, op, y):
        if self.E is None:
            return y

        y = NumpyVectorArray(np.array(y).T)
        if op == pymess.MESS_OP_NONE:
            x = self.E.apply_inverse(y)
        else:
            x = self.E.apply_inverse_adjoint(y)
        return np.matrix(x.data).T

    def ApE_apply(self, op, p, idx_p, y):
        y = NumpyVectorArray(np.array(y).T)
        if op == pymess.MESS_OP_NONE:
            x = self.A.apply(y)
            if self.E is None:
                x += p * y
            else:
                x += p * self.E.apply(y)
        else:
            x = self.A.apply_adjoint(y)
            if self.E is None:
                x += p.conjugate() * y
            else:
                x += p.conjugate() * self.E.apply_adjoint(y)
        return np.matrix(x.data).T

    def ApEs_apply(self, op, p, idx_p, y):
        y = NumpyVectorArray(np.array(y).T)
        if self.E is None:
            E = IdentityOperator(self.A.source)
        else:
            E = self.E

        if p.imag == 0.0:
            ApE = LincombOperator((self.A, E), (1., p.real))
        else:
            ApE = LincombOperator((self.A, E), (1., p))

        if op == pymess.MESS_OP_NONE:
            x = ApE.apply_inverse(y)
        else:
            x = ApE.apply_inverse_adjoint(y)
        return np.matrix(x.data).T


def solve_lyap(A, E, B, trans=False, tol=None):
    """Find a factor of the solution of a Lyapunov equation

    Returns factor Z such that Z * Z^T is approximately the solution X of a Lyapunov equation (if E is None)::

        A * X + X * A^T + B * B^T = 0

    or generalized Lyapunov equation::

        A * X * E^T + E * X * A^T + B * B^T = 0.

    If trans is True, then solve (if E is None)::

        A^T * X + X * A + B^T * B = 0

    or::

        A^T * X * E + E^T * X * A + B^T * B = 0.

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E or None.
    B
        The |Operator| B.
    trans
        If A, E, and B need to be transposed.
    tol
        Tolerance parameter.
    """
    assert isinstance(A, OperatorInterface) and A.linear
    assert A.source == A.range
    assert isinstance(B, OperatorInterface) and B.linear
    assert not trans and B.range == A.source or trans and B.source == A.source
    assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source

    if A.source.dim <= 1000:
        A_matrix = A._matrix
        if A.sparse:
            A_matrix = A_matrix.toarray()
        if E is not None:
            E_matrix = E._matrix
            if E.sparse:
                E_matrix = E_matrix.toarray()
        B_matrix = B._matrix
        if B.sparse:
            B_matrix = B_matrix.toarray()
        if not trans:
            if E is None:
                Z = pymess.lyap(A_matrix, None, B_matrix)
            else:
                Z = pymess.lyap(A_matrix, E_matrix, B_matrix)
        else:
            if E is None:
                Z = pymess.lyap(A_matrix.T, None, B_matrix.T)
            else:
                Z = pymess.lyap(A_matrix.T, E_matrix.T, B_matrix.T)
    else:
        opts = pymess.options()
        if trans:
            opts.type = pymess.MESS_OP_TRANSPOSE
        if tol is not None:
            opts.rel_change_tol = tol
            opts.adi.res2_tol = tol
            opts.adi.res2c_tol = tol
        eqn = LyapunovEquation(opts, A, E, B)
        Z, status = pymess.lradi(eqn, opts)

    Z = NumpyVectorArray(np.array(Z).T)

    return Z
