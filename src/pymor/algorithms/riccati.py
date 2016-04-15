# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.operators.interfaces import OperatorInterface
from pymor.vectorarrays.numpy import NumpyVectorArray


def solve_ricc(A, B=None, E=None, Q=None, C=None, R=None, D=None, G=None, L=None, trans=False, meth=None, tol=None):
    """Find a factor of the solution of a Riccati equation

    Returns factor Z such that Z * Z^T is approximately the solution X of a Riccati equation::

        A^T * X * E + E^T * X * A - (L + E^T * X * B) * R^{-1} * (L + E^T * X * B)^T + Q = 0.

    If E in None, it is taken to be the identity matrix.
    B * R^-1 B^T can instead be given by G. In this case, B, R, D, L need to be None, and G not a None.
    Q can instead be given as C^T * C. In this case, Q needs to be None, and C not a None.
    R can instead be given as D^T * D. In this case, R needs to be None, and D not a None.
    If R, D, G are None, then R is taken to be the identity matrix.
    If L in None, then it is taken to be the zero matrix.
    If trans is True, then the dual Riccati equation is solved::

        A * X * E^T + E * X * A^T - (L + E * X * C^T) * R^{-1} * (L + E * X * C^T)^T + Q = 0,

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
        Method to use {'scipy', 'slycot', 'pymess_care'}. If meth is None, a solver is chosen automatically.
    tol
        Tolerance parameter.
    """
    assert isinstance(A, OperatorInterface) and A.linear
    assert A.source == A.range
    assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source
    if not trans:
        if G is not None:
            assert B is None and R is None and D is None and L is None
            assert isinstance(G, OperatorInterface) and G.linear
            assert G.source == G.range == A.source
        else:
            assert B is not None
            assert isinstance(B, OperatorInterface) and B.linear
            assert B.range == A.source
        if C is not None:
            assert Q is None
            assert isinstance(C, OperatorInterface) and C.linear
            assert C.source == A.source
        else:
            assert isinstance(Q, OperatorInterface) and Q.linear
            assert Q.source == Q.range == A.source
        if D is not None:
            assert R is None
            assert isinstance(D, OperatorInterface) and D.linear
            assert D.source == B.source
        else:
            if R is not None:
                assert isinstance(R, OperatorInterface) and R.linear
                assert R.source == R.range == B.source
        if L is not None:
            assert isinstance(L, OperatorInterface) and L.linear
            assert L.source == B.source and L.range == B.range
    else:
        if G is not None:
            assert C is None and R is None and D is None and L is None
            assert isinstance(G, OperatorInterface) and G.linear
            assert G.source == G.range == A.source
        else:
            assert C is not None
            assert isinstance(C, OperatorInterface) and C.linear
            assert C.source == A.source
        if B is not None:
            assert Q is None
            assert isinstance(B, OperatorInterface) and B.linear
            assert B.range == A.source
        else:
            assert isinstance(Q, OperatorInterface) and Q.linear
            assert Q.source == Q.range == A.source
        if D is not None:
            assert R is None
            assert isinstance(D, OperatorInterface) and D.linear
            assert D.source == C.range
        else:
            if R is not None:
                assert isinstance(R, OperatorInterface) and R.linear
                assert R.source == R.range == C.range
        if L is not None:
            assert isinstance(L, OperatorInterface) and L.linear
            assert L.source == C.range and L.range == C.source
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
        if E is not None or G is not None or L is not None:
            raise NotImplementedError()
        import scipy.linalg as spla
        A_mat = A._matrix
        if A.sparse:
            A_mat = A_mat.toarray()
        if R is not None:
            R_mat = R._matrix
            if R.sparse:
                R_mat = R_mat.toarray()
        elif D is not None:
            D_mat = D._matrix
            if D.sparse:
                D_mat = D_mat.toarray()
            R_mat = D_mat.T.dot(D_mat)
        else:
            if not trans:
                R_mat = np.eye(B.source.dim)
            else:
                R_mat = np.eye(C.range.dim)
        if not trans:
            B_mat = B._matrix
            if B.sparse:
                B_mat = B_mat.toarray()
            if Q is not None:
                Q_mat = Q._matrix
                if Q.sparse:
                    Q_mat = Q_mat.toarray()
            else:
                C_mat = C._matrix
                if C.sparse:
                    C_mat = C_mat.toarray()
                Q_mat = C_mat.T.dot(C_mat)
            X = spla.solve_continuous_are(A_mat, B_mat, Q_mat, R_mat)
        else:
            C_mat = C._matrix
            if C.sparse:
                C_mat = C_mat.toarray()
            if Q is not None:
                Q_mat = Q._matrix
                if Q.sparse:
                    Q_mat = Q_mat.toarray()
            else:
                B_mat = B._matrix
                if B.sparse:
                    B_mat = B_mat.toarray()
                Q_mat = B_mat.dot(B_mat.T)
            X = spla.solve_continuous_are(A_mat.T, C_mat.T, Q_mat, R_mat)
        from pymor.algorithms.cholp import cholp
        Z = cholp(X, copy=False)
    elif meth == 'slycot':
        if G is not None:
            raise NotImplementedError()
        import slycot
        A_mat = A._matrix
        if trans:
            A_mat = A_mat.T
        if A.sparse:
            A_mat = A_mat.toarray()
        if E is not None:
            E_mat = E._matrix
            if trans:
                E_mat = E_mat.T
            if E.sparse:
                E_mat = E_mat.toarray()
        B_mat = B._matrix
        if trans:
            B_mat = B_mat.T
        if B.sparse:
            B_mat = B_mat.toarray()

        n = A_mat.shape[0]
        m = B_mat.shape[1] if not trans else B_mat.shape[0]
        dico = 'C'
        p = None
        fact = 'N'
        if C is not None:
            Q_mat = C._matrix
            if trans:
                Q_mat = Q_mat.T
            if C.sparse:
                Q_mat = Q_mat.toarray()
            p = Q_mat.shape[0]
            fact = 'C'
        else:
            Q_mat = Q._matrix
            if Q.sparse:
                Q_mat = Q_mat.toarray()
        if D is not None:
            R_mat = D._matrix
            if trans:
                R_mat = R_mat.T
            if D.sparse:
                R_mat = R_mat.toarray()
            p = R_mat.shape[0]
            if fact == 'N':
                fact = 'D'
            else:
                fact = 'B'
        else:
            if R is None:
                R_mat = np.eye(m)
            else:
                R_mat = R._matrix
                if R.sparse:
                    R_mat = R_mat.toarray()
        if L is not None:
            jobl = 'N'
            L_mat = L._matrix
            if trans:
                L_mat = L_mat.T
            if L.sparse:
                L_mat = L_mat.toarray()
        else:
            jobl = 'Z'
            L_mat = None

        if E is None:
            X, _, _, _, _ = slycot.sb02od(n, m, A_mat, B_mat, Q_mat, R_mat, dico, p=p, L=L_mat,
                                          fact=fact)
        else:
            jobb = 'B'
            uplo = 'L'
            scal = 'N'
            sort = 'S'
            acc = 'R'
            if L is None:
                L_mat = np.zeros((n, m))
            _, X, _, _, _, _, _, _, iwarn = slycot.sg02ad(dico, jobb, fact, uplo, jobl, scal, sort, acc, n, m, p,
                                                          A_mat, E_mat, B_mat, Q_mat, R_mat, L_mat)
            if iwarn == 1:
                print('slycot.sg02ad warning: solution may be inaccurate.')

        from pymor.algorithms.cholp import cholp
        Z = cholp(X, copy=False)
    elif meth == 'pymess_care':
        if Q is not None or R is not None or D is not None or G is not None or L is not None:
            raise NotImplementedError()
        import pymess
        A_mat = A._matrix
        if A.sparse:
            A_mat = A_mat.toarray()
        if E is not None:
            E_mat = E._matrix
            if E.sparse:
                E_mat = E_mat.toarray()
        B_mat = B._matrix
        if B.sparse:
            B_mat = B_mat.toarray()
        C_mat = C._matrix
        if C.sparse:
            C_mat = C_mat.toarray()
        if not trans:
            if E is None:
                Z = pymess.care(A_mat, None, B_mat, C_mat)
            else:
                Z = pymess.care(A_mat, E_mat, B_mat, C_mat)
        else:
            if E is None:
                Z = pymess.care(A_mat.T, None, C_mat.T, B_mat.T)
            else:
                Z = pymess.care(A_mat.T, E_mat.T, C_mat.T, B_mat.T)

    Z = NumpyVectorArray(np.array(Z).T)

    return Z
