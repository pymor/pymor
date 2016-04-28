# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.algorithms.numpy import to_numpy_operator
from pymor.operators.interfaces import OperatorInterface


def operator2matrix(A):
    """Transforms NumpyMatrixOperator to NumPy array

    Parameters
    ----------
    A
        NumpyMatrixOperator or None.

    Returns
    -------
    A_mat
        |NumPy array| or None.
    """
    if A is None:
        A_mat = None
    else:
        A_mat = to_numpy_operator(A)._matrix
    return A_mat


def solve_ricc(A, E=None, B=None, Q=None, C=None, R=None, G=None,
               trans=False, meth=None, tol=None):
    """Find a factor of the solution of a Riccati equation

    Returns factor Z such that Z * Z^T is approximately the solution X of a
    Riccati equation::

        A^T * X * E + E^T * X * A - E^T * X * B * R^{-1} * B^T * X * E + Q = 0.

    If E in None, it is taken to be the identity matrix.
    Q can instead be given as C^T * C. In this case, Q needs to be None, and
    C not None.
    B * R^{-1} B^T can instead be given by G. In this case, B and R need to be
    None, and G not None.
    If R and G are None, then R is taken to be the identity matrix.
    If trans is True, then the dual Riccati equation is solved::

        A * X * E^T + E * X * A^T - E * X * C^T * R^{-1} * C * X * E^T + Q = 0,

    where Q can be replaced by B * B^T and C^T * R^{-1} * C by G.

    Parameters
    ----------
    A
        The |Operator| A.
    B
        The |Operator| B or None.
    E
        The |Operator| E or None.
    Q
        The |Operator| Q or None.
    C
        The |Operator| C or None.
    R
        The |Operator| R or None.
    D
        The |Operator| D or None.
    G
        The |Operator| G or None.
    L
        The |Operator| L or None.
    trans
        If the dual equation needs to be solved.
    meth
        Method to use {'scipy', 'slycot', 'pymess_care'}.
        If meth is None, a solver is chosen automatically.
    tol
        Tolerance parameter.

    Returns
    -------
    Z
        Low-rank factor of the Riccati equation solution, |VectorArray| from `A.source`.
    """
    assert isinstance(A, OperatorInterface) and A.linear
    assert A.source == A.range
    if E is not None:
        assert isinstance(E, OperatorInterface) and E.linear
        assert E.source == E.range == A.source
    if not trans:
        if C is not None:
            assert Q is None
            assert isinstance(C, OperatorInterface) and C.linear
            assert C.source == A.source
        else:
            assert isinstance(Q, OperatorInterface) and Q.linear
            assert Q.source == Q.range == A.source
        if G is not None:
            assert B is None and R is None
            assert isinstance(G, OperatorInterface) and G.linear
            assert G.source == G.range == A.source
        else:
            assert isinstance(B, OperatorInterface) and B.linear
            assert B.range == A.source
            if R is not None:
                assert isinstance(R, OperatorInterface) and R.linear
                assert R.source == R.range == B.source
    else:
        if B is not None:
            assert Q is None
            assert isinstance(B, OperatorInterface) and B.linear
            assert B.range == A.source
        else:
            assert isinstance(Q, OperatorInterface) and Q.linear
            assert Q.source == Q.range == A.source
        if G is not None:
            assert C is None and R is None
            assert isinstance(G, OperatorInterface) and G.linear
            assert G.source == G.range == A.source
        else:
            assert C is not None
            assert isinstance(C, OperatorInterface) and C.linear
            assert C.source == A.source
            if R is not None:
                assert isinstance(R, OperatorInterface) and R.linear
                assert R.source == R.range == C.range
    assert meth is None or meth in {'scipy', 'slycot', 'pymess_care'}

    if meth is None:
        import imp
        try:
            imp.find_module('slycot')
            meth = 'slycot'
        except ImportError:
            try:
                imp.find_module('pymess')
                meth = 'pymess_care'
            except ImportError:
                meth = 'scipy'

    if meth == 'scipy':
        if E is not None or G is not None:
            raise NotImplementedError()
        import scipy.linalg as spla
        A_mat = operator2matrix(A)
        B_mat = operator2matrix(B)
        C_mat = operator2matrix(C)
        Q_mat = operator2matrix(Q)
        R_mat = operator2matrix(R)
        if R is None:
            if not trans:
                R_mat = np.eye(B.source.dim)
            else:
                R_mat = np.eye(C.range.dim)
        if not trans:
            if Q is None:
                Q_mat = C_mat.T.dot(C_mat)
            X = spla.solve_continuous_are(A_mat, B_mat, Q_mat, R_mat)
        else:
            if Q is None:
                Q_mat = B_mat.dot(B_mat.T)
            X = spla.solve_continuous_are(A_mat.T, C_mat.T, Q_mat, R_mat)
        from pymor.algorithms.cholp import cholp
        Z = cholp(X, copy=False)
    elif meth == 'slycot':
        import slycot
        A_mat = operator2matrix(A)
        B_mat = operator2matrix(B)
        C_mat = operator2matrix(C)
        R_mat = operator2matrix(R)
        G_mat = operator2matrix(G)
        Q_mat = operator2matrix(Q)

        n = A_mat.shape[0]
        dico = 'C'

        if E is None:
            if not trans:
                if G is None:
                    if R is None:
                        G_mat = B_mat.dot(B_mat.T)
                    else:
                        G_mat = slycot.sb02mt(n, B_mat.shape[1], B_mat, R_mat)[-1]
                if C is not None:
                    Q_mat = C_mat.T.dot(C_mat)
                X = slycot.sb02md(n, A_mat, G_mat, Q_mat, dico)[0]
            else:
                if G is None:
                    if R is None:
                        G_mat = C_mat.T.dot(C_mat)
                    else:
                        G_mat = slycot.sb02mt(n, C_mat.shape[0], C_mat.T, R_mat)[-1]
                if B is not None:
                    Q_mat = B_mat.dot(B_mat.T)
                X = slycot.sb02md(n, A_mat.T, G_mat, Q_mat, dico)[0]
        else:
            E_mat = operator2matrix(E)
            jobb = 'B' if G is None else 'B'
            fact = 'C' if Q is None else 'N'
            uplo = 'U'
            jobl = 'Z'
            scal = 'N'
            sort = 'S'
            acc = 'R'
            if not trans:
                m = 0 if B is None else B_mat.shape[1]
                p = 0 if C is None else C_mat.shape[0]
                if G is not None:
                    B_mat = G_mat
                    R_mat = np.empty((1, 1))
                elif R is None:
                    R_mat = np.eye(m)
                if Q is None:
                    Q_mat = C_mat
                L_mat = np.empty((n, m))
                ret = slycot.sg02ad(dico, jobb, fact, uplo, jobl, scal, sort, acc, n, m, p,
                                    A_mat, E_mat, B_mat, Q_mat, R_mat, L_mat)
            else:
                m = 0 if C is None else C_mat.shape[0]
                p = 0 if B is None else B_mat.shape[1]
                if G is not None:
                    C_mat = G_mat
                    R_mat = np.empty((1, 1))
                elif R is None:
                    C_mat = C_mat.T
                    R_mat = np.eye(m)
                if Q is None:
                    Q_mat = B_mat.T
                L_mat = np.empty((n, m))
                ret = slycot.sg02ad(dico, jobb, fact, uplo, jobl, scal, sort, acc, n, m, p,
                                    A_mat.T, E_mat.T, C_mat, Q_mat, R_mat, L_mat)
            X = ret[1]
            iwarn = ret[-1]
            if iwarn == 1:
                print('slycot.sg02ad warning: solution may be inaccurate.')
        from pymor.algorithms.cholp import cholp
        Z = cholp(X, copy=False)
    elif meth == 'pymess_care':
        if Q is not None or R is not None or G is not None:
            raise NotImplementedError()
        import pymess
        A_mat = operator2matrix(A)
        E_mat = operator2matrix(E)
        B_mat = operator2matrix(B)
        C_mat = operator2matrix(C)
        if not trans:
            Z = pymess.care(A_mat, E_mat, B_mat, C_mat)
        else:
            if E is None:
                Z = pymess.care(A_mat.T, None, C_mat.T, B_mat.T)
            else:
                Z = pymess.care(A_mat.T, E_mat.T, C_mat.T, B_mat.T)

    Z = A.source.from_data(np.array(Z).T)

    return Z
