# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from pymor.operators.interfaces import OperatorInterface
from pymor.vectorarrays.numpy import NumpyVectorArray
import pymess


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
        if E is None:
            eqn = pymess.equation_lyap(opts, A._matrix, None, B._matrix)
        else:
            eqn = pymess.equation_lyap(opts, A._matrix, E._matrix, B._matrix)
        Z, status = pymess.lradi(eqn, opts)

    Z = NumpyVectorArray(np.array(Z).T)

    return Z
