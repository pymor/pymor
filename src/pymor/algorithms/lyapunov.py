# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse as sps

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
        The |VectorArray| RHS.
    """
    def __init__(self, opt, A, E, RHS):
        dim = A.source.dim

        super(LyapunovEquation, self).__init__(name='lyap_eqn', opt=opt, dim=dim)

        self.A = A
        self.E = E
        self.RHS = np.array(RHS.data.T)
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
            x = self.A.apply_adjoint_inverse(y)
        return np.matrix(x.data).T

    def Es_apply(self, op, y):
        if self.E is None:
            return y

        y = NumpyVectorArray(np.array(y).T)
        if op == pymess.MESS_OP_NONE:
            x = self.E.apply_inverse(y)
        else:
            x = self.E.apply_adjoint_inverse(y)
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
            E = NumpyMatrixOperator(sps.eye(self.dim, format='csc'))
        else:
            E = self.E

        if p.imag == 0.0:
            ApE = self.A.assemble_lincomb((self.A, E), (1, p.real))
        else:
            ApE = self.A.assemble_lincomb((self.A, E), (1, p))

        if op == pymess.MESS_OP_NONE:
            x = ApE.apply_inverse(y)
        else:
            x = ApE.apply_adjoint_inverse(y)
        return np.matrix(x.data).T


def solve_lyap(A, E, B, trans=False):
    """Find a factor of the solution of a Lyapunov equation

    Returns factor Z such that Z * Z^T is approximately the solution X of a Lyapunov equation::

        A * X + X * A^T + B * B^T = 0

    if E is None, otherwise a generalized Lyapunov equation::

        A * X * E^T + E * X * A^T + B * B^T = 0.

    If trans is True, then solve::

        A^T * X + X * A + B^T * B = 0

    if E is None, otherwise::

        A^T * X * E + E^T * X * A + B^T * B = 0.

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E or None.
    B
        The |VectorArray| B.
    """
    if A.source.dim <= 1000:
        if not A.sparse:
            A = A._matrix
        else:
            A = A._matrix.toarray()
        if E is not None:
            if not E.sparse:
                E = E._matrix
            else:
                E = E._matrix.toarray()
        if not trans:
            Z = pymess.lyap(A, E, B.data.T)
        else:
            if E is None:
                Z = pymess.lyap(A.T, None, B.data.T)
            else:
                Z = pymess.lyap(A.T, E.T, B.data.T)
    else:
        opts = pymess.options()
        if trans:
            opts.type = pymess.MESS_OP_TRANSPOSE
        eqn = LyapunovEquation(opts, A, E, B)
        Z, status = pymess.lradi(eqn, opts)

    Z = NumpyVectorArray(np.array(Z).T)

    return Z
