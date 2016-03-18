# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse as sps

from pymor.operators.interfaces import OperatorInterface
from pymor.operators.constructions import IdentityOperator, LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorArray


def solve_ricc(A, B=None, E=None, Q=None, C=None, R=None, D=None, G=None, L=None, trans=False, meth='scipy', tol=None):
    """Find a factor of the solution of a Riccati equation

    Returns factor Z such that Z * Z^T is approximately the solution X of a Riccati equation::

        A^T * X * E + E^T * X * A - (L + E^T * X * B) * R^{-1} * (L + E^T * X * B)^T + Q = 0.

    If E in None, it is taken to be the identity matrix.
    Q can instead be given as C^T * C. In this case, Q needs to be None, and C not a None.
    R can instead be given as D^T * D. In this case, R needs to be None, and D not a None.
    B * R^-1 B^T can instead be given by G. In this case, B, R, D need to be None, and G not a None.
    If L in None, it is taken to be the zero matrix.
    If trans is True, then A and E are transposed.

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
    tol
        Tolerance parameter.
    """
    assert isinstance(A, OperatorInterface) and A.linear
    assert A.source == A.range
    assert isinstance(B, OperatorInterface) and B.linear
    assert not trans and B.range == A.source or trans and B.source == A.source
    assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source
    if Q is not None:
        assert isinstance(Q, OperatorInterface) and Q.linear
        assert Q.source == Q.range == A.source
        assert C is None
    else:
        assert isinstance(C, OperatorInterface) and C.linear
        assert not trans and C.source == A.source or trans and C.range == A.source
    if B is not None:
        if R is not None:
            assert isinstance(R, OperatorInterface) and R.linear
            assert not trans and R.source == R.range == B.source or trans and R.source == R.range == B.range
            assert D is None
        else:
            assert isinstance(D, OperatorInterface) and D.linear
            assert not trans and D.source == B.source or trans and D.range == B.range
    else:
        assert isinstance(G, OperatorInterface) and G.linear
        assert G.source == G.range == A.source
        assert L is None
    if L is not None:
        assert isinstance(L, OperatorInterface) and L.linear
        assert L.source == B.source and L.range == B.range
    assert meth in {'scipy', 'slycot', 'pymess_care'}

    if meth == 'scipy':
        if E is not None or G is not None or L is not None:
            raise NotImplementedError()
        import scipy.linalg as spla
        A_matrix = A._matrix
        if trans:
            A_matrix = A_matrix.T
        if A.sparse:
            A_matrix = A_matrix.toarray()
        B_matrix = B._matrix
        if trans:
            B_matrix = B_matrix.T
        if B.sparse:
            B_matrix = B_matrix.toarray()
        if Q is not None:
            Q_matrix = Q._matrix
            if Q.sparse:
                Q_matrix = Q_matrix.toarray()
        else:
            C_matrix = C._matrix
            if C.sparse:
                C_matrix = C_matrix.toarray()
            if not trans:
                Q_matrix = C_matrix.T.dot(C_matrix)
            else:
                Q_matrix = C_matrix.dot(C_matrix.T)
        R_matrix = R._matrix
        if R.sparse:
            R_matrix = R_matrix.toarray()
        X = spla.solve_continuous_are(A_matrix, B_matrix, Q_matrix, R_matrix)
        from pymor.algorithms.cholp import cholp
        Z = cholp(X, copy=False)
    elif meth == 'slycot':
        import slycot
        A_matrix = A._matrix
        if trans:
            A_matrix = A_matrix.T
        if A.sparse:
            A_matrix = A_matrix.toarray()
        if E is not None:
            E_matrix = E._matrix
            if trans:
                E_matrix = E_matrix.T
            if E.sparse:
                E_matrix = E_matrix.toarray()
        B_matrix = B._matrix
        if trans:
            B_matrix = B_matrix.T
        if B.sparse:
            B_matrix = B_matrix.toarray()

        n = A_matrix.shape[0]
        m = B_matrix.shape[1] if not trans else B_matrix.shape[0]
        dico = 'C'
        p = None
        fact = 'N'
        if C is not None:
            Q_matrix = Q._matrix
            if trans:
                Q_matrix = Q_matrix.T
            if C.sparse:
                Q_matrix = Q_matrix.toarray()
            p = Q_matrix.shape[1]
            fact = 'C'
        else:
            Q_matrix = Q._matrix
            if Q.sparse:
                Q_matrix = Q_matrix.toarray()
        if D is not None:
            R_matrix = R._matrix
            if trans:
                R_matrix = R_matrix.T
            if D.sparse:
                R_matrix = R_matrix.toarray()
            p = R_matrix.shape[1]
            if fact == 'N':
                fact = 'D'
            else:
                fact = 'B'
        else:
            R_matrix = R._matrix
            if R.sparse:
                R_matrix = R_matrix.toarray()
        if L is not None:
            jobl = 'N'
            L_matrix = L._matrix
            if trans:
                L_matrix = L_matrix.T
            if L.sparse:
                L_matrix = L_matrix.toarray()
        else:
            jobl = 'Z'
            L_matrix = None

        if E is None:
            X, _, _, _, _ = slycot.sb02od(n, m, A_matrix, B_matrix, Q_matrix, R_matrix, dico, p=p, L=L_matrix,
                                          fact=fact)
        else:
            jobb = 'B'
            uplo = 'L'
            scal = 'G'
            sort = 'S'
            acc = 'R'
            _, X, _, _, _, _, _, _, iwarn, _ = slycot.sg02ad(dico, jobb, fact, uplo, jobl, scal, sort, acc, n, m, p,
                                               A_matrix, E_matrix, B_matrix, Q_matrix, R_matrix, L_matrix)
            if iwarn == 1:
                print('slycot.sg02ad warning: solution may be inaccurate.')

        from pymor.algorithms.cholp import cholp
        Z = cholp(X, copy=False)
    elif meth == 'pymess_care':
        if Q is not None or R is not None or D is not None or G is not None or L is not None:
            raise NotImplementedError()
        import pymess
        A_matrix = A._matrix
        if trans:
            A_matrix = A_matrix.T
        if A.sparse:
            A_matrix = A_matrix.toarray()
        if E is not None:
            E_matrix = E._matrix
            if trans:
                E_matrix = E_matrix.T
            if E.sparse:
                E_matrix = E_matrix.toarray()
        else:
            E_matrix = None
        B_matrix = B._matrix
        if trans:
            B_matrix = B_matrix.T
        if B.sparse:
            B_matrix = B_matrix.toarray()
        C_matrix = C._matrix
        if trans:
            C_matrix = C_matrix.T
        if C.sparse:
            C_matrix = C_matrix.toarray()
        Z = pymess.care(A_matrix, E_matrix, B_matrix, C_matrix)

    Z = NumpyVectorArray(np.array(Z).T)

    return Z
