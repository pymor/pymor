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


def solve_lyap(A, E, B, trans=False):
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
        The |VectorArray| B.
    trans
        If A, E, and B need to be transposed.
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
        if E is None:
            if not trans:
                eqn = pymess.equation_lyap(opts, A._matrix, None, B.data.T)
            else:
                eqn = pymess.equation_lyap(opts, A._matrix, None, B.data)
        else:
            if not trans:
                eqn = pymess.equation_lyap(opts, A._matrix, E._matrix, B.data.T)
            else:
                eqn = pymess.equation_lyap(opts, A._matrix, E._matrix, B.data)
        Z, status = pymess.lradi(eqn, opts)

    Z = NumpyVectorArray(np.array(Z).T)

    return Z
