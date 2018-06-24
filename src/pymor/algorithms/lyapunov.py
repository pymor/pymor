# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import scipy.linalg as spla
import numpy as np

from pymor.operators.interfaces import OperatorInterface
from pymor.operators.constructions import IdentityOperator, LincombOperator
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.logger import getLogger
from pymor.algorithms.genericsolvers import _parse_options


def solve_lyap(A, E, B, trans=False, options=None):
    """Find a factor of the solution of a Lyapunov equation.

    Returns factor :math:`Z` such that :math:`Z Z^T` is approximately
    the solution :math:`X` of a Lyapunov equation (if E is `None`).

    .. math::
        A X + X A^T + B B^T = 0

    or generalized Lyapunov equation

    .. math::
        A X E^T + E X A^T + B B^T = 0.

    If trans is `True`, then solve (if E is `None`)

    .. math::
        A^T X + X A + B^T B = 0

    or

    .. math::
        A^T X E + E^T X A + B^T B = 0.

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E or `None`.
    B
        The |Operator| B.
    trans
        If the dual equation needs to be solved.
    options
        The |solver_options| to use (see :func:`lyap_solver_options`).

    Returns
    -------
    Z
        Low-rank factor of the Lyapunov equation solution, |VectorArray| from `A.source`.
    """
    _solve_lyap_check_args(A, E, B, trans)
    options = _parse_options(options, lyap_solver_options(), 'lradi', None, False)
    if options['type'] == 'lradi':
        return lradi(A, E, B, trans, options)
    else:
        raise ValueError('Unknown solver type')


def lradi(A, E, B, trans=False, options=None):
    logger = getLogger('pymor.algorithms.lyapunov.lradi')

    shift_options = options['shift_options'][options['shifts']]

    if shift_options['type'] == 'projection_shifts':
        init_shifts = projection_shifts_init
        iteration_shifts = projection_shifts
    else:
        raise ValueError('Unknown lradi shift strategy')

    if E is None:
        E = IdentityOperator(A.source)

    if trans:
        Z = A.range.empty(reserve=B.range.dim * options['maxiter'])
        W = B.as_source_array()
        shifts = init_shifts(A, E, W)
        size_shift = shifts.size
        j = 0
        j_shift = 0
        Btol = np.linalg.norm(W.gramian(), ord=2) * options['tol']
        # using for loop instead of while not working -> for loop won't allow j += 2
        while(np.linalg.norm(W.gramian(), ord=2) > Btol and j < options['maxiter']):
            logger.info("Residual at step {}: {}".format(j, np.linalg.norm(W.gramian(), ord=2)))
            if shifts[j_shift].imag == 0:
                AaE = LincombOperator([A, E], [1, shifts[j_shift].real])
                V = AaE.apply_inverse_transpose(W)
                W -= E.apply_transpose(V)*2*shifts[j_shift].real
                Z.append(V*np.sqrt(-2*shifts[j_shift].real))
                j += 1
                j_shift += 1
            else:
                AaE = LincombOperator([A, E], [1, shifts[j_shift]])
                V = AaE.apply_inverse_transpose(W)
                g = 2*np.sqrt(-shifts[j_shift].real)
                d = shifts[j_shift].real/shifts[j_shift].imag
                W += E.apply_transpose(V.real + V.imag*d)*g**2
                Z.append((V.real + V.imag*d)*g)
                Z.append(V.imag*g*np.sqrt(d**2+1))
                j += 2
                j_shift += 2
            if j_shift >= size_shift:
                j_shift = 0
                shifts = iteration_shifts(A, E, Z, W, shifts, j, shift_options)
                size_shift = shifts.size
        return Z
    else:
        Z = A.source.empty(reserve=B.source.dim * options['maxiter'])
        W = B.as_range_array()
        shifts = init_shifts(A, E, W)
        size_shift = shifts.size
        j = 0
        j_shift = 0
        Btol = np.linalg.norm(W.gramian(), ord=2) * options['tol']
        # using for loop instead of while not working -> for loop won't allow j += 2
        while(np.linalg.norm(W.gramian(), ord=2) > Btol and j < options['maxiter']):
            logger.info("Residual at step {}: {}".format(j, np.linalg.norm(W.gramian(), ord=2)))
            if shifts[j_shift].imag == 0:
                AaE = LincombOperator([A, E], [1, shifts[j_shift].real])
                V = AaE.apply_inverse(W)
                W -= E.apply(V)*2*shifts[j_shift].real
                Z.append(V*np.sqrt(-2*shifts[j_shift].real))
                j += 1
                j_shift += 1
            else:
                AaE = LincombOperator([A, E], [1, shifts[j_shift]])
                V = AaE.apply_inverse(W)
                g = 2*np.sqrt(-shifts[j_shift].real)
                d = shifts[j_shift].real/shifts[j_shift].imag
                W += E.apply(V.real + V.imag*d)*g**2
                Z.append((V.real + V.imag*d)*g)
                Z.append(V.imag*g*np.sqrt(d**2+1))
                j += 2
                j_shift += 2
            if j_shift >= size_shift:
                j_shift = 0
                shifts = iteration_shifts(A, E, Z, W, shifts, j, shift_options)
                size_shift = shifts.size
        return Z


def projection_shifts_init(A, E, B):
    Q = gram_schmidt(B, atol=0, rtol=0)
    shifts = spla.eigvals(A.apply2(Q, Q), b=E.apply2(Q, Q))
    shifts = shifts[np.real(shifts) < 0]  # make shifts stable rather than deleting them?
    if shifts.size == 0:
        # use random subspace instead of span{B} (with same dimensions)
        return projection_shifts_init(A, E, B.space.make_array(np.random.rand(len(B), B.space.dim)))
    else:
        return shifts


def projection_shifts(A, E, Z, W, prev_shifts, j, shift_options):
    L = prev_shifts.size
    r = len(W)
    u = shift_options['z_columns']
    if prev_shifts[L-u].imag < 0:
        u = u + 1
    d = L - u
    if d < 0:
        u = L
        d = 0
    Z = Z[(j - u) * r:j * r]
    B = np.zeros((u, u))
    G = np.zeros((u, 1))
    Ir = np.eye(r)
    iC = np.where(np.imag(prev_shifts) > 0)[0]  # complex shifts indices (first shift of complex pair)
    iR = np.where(np.isreal(prev_shifts))[0]  # real shifts indices
    iC = iC[iC >= d]  # remove unnecessary shifts
    iR = iR[iR >= d]
    i = 0
    # use pymor vectorarray instead of numpy arrays? what about spla.kron, and spla.svd?
    while i < u:
        rS = iR[iR < d + i]
        cS = iC[iC < d + i]
        rp = prev_shifts[d + i].real
        cp = prev_shifts[d + i].imag
        G[i, 0] = np.sqrt(-2 * rp)
        if cp == 0:
            B[i, i] = rp
            if rS.size > 0:
                B[i, rS - d] = -2 * np.sqrt(rp*np.real(prev_shifts[rS]))
            if cS.size > 0:
                B[i, cS - d] = -2 * np.sqrt(2*rp*np.real(prev_shifts[cS]))
            i = i + 1
        else:
            sri = np.sqrt(rp**2+cp**2)
            B[i: i + 2, i: i + 2] = [[2*rp, -sri], [sri, 0]]
            if rS.size > 0:
                B[i, rS - d] = -2 * np.sqrt(2*rp*np.real(prev_shifts[rS]))
            if cS.size > 0:
                B[i, cS - d] = -4 * np.sqrt(rp*np.real(prev_shifts[cS]))
            i = i + 2
    B = spla.kron(B, Ir)
    G = spla.kron(G, Ir)

    s, v = spla.svd(Z.gramian(), full_matrices=False)[1:3]
    P = v.T.dot(np.diag(1. / np.sqrt(s)))
    Q = Z.data.T.dot(P)

    E_V = E.apply(Z).data.T
    T = Q.T.dot(E_V)
    Ap = Q.T.dot(W.data.T).dot(G.T).dot(P) + T.dot(B.dot(P))
    Ep = T.dot(P)

    shifts = spla.eigvals(Ap, b=Ep)

    shifts = shifts[np.real(shifts) < 0]  # make shifts stable rather than deleting them?
    if shifts.size == 0:
        return prev_shifts
    else:
        return shifts


def lyap_solver_options(lradi_tol=1e-10,
                        lradi_maxiter=100,
                        lradi_shifts='projection_shifts',
                        projection_shifts_z_columns=1):
    """Returns available Lyapunov equation solvers with default |solver_options|.

    Returns
    -------
    A dict of available solvers with default |solver_options|.
    """
    return {'lradi': {'type': 'lradi',
                      'tol': lradi_tol,
                      'maxiter': lradi_maxiter,
                      'shifts': lradi_shifts,
                      'shift_options': {'projection_shifts': {'type': 'projection_shifts',
                                                              'z_columns': projection_shifts_z_columns}}}}


def _solve_lyap_check_args(A, E, B, trans=False):
    assert isinstance(A, OperatorInterface) and A.linear
    assert A.source == A.range
    assert isinstance(B, OperatorInterface) and B.linear
    assert not trans and B.range == A.source or trans and B.source == A.source
    assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source
