# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import scipy.linalg as spla
import numpy as np

from pymor.operators.interfaces import OperatorInterface
from pymor.operators.constructions import IdentityOperator, LincombOperator
from pymor.algorithms.gram_schmidt import gram_schmidt
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
    options = _parse_options(options, lyap_solver_options(), 'adi_iteration', None, False)
    assert options['type'] == 'adi_iteration'

    if E is None:
        E = IdentityOperator(A.source)

    Z = A.source.empty(reserve=B.source.dim * options['maxiter'])
    W = B.as_range_array()
    V = None
    j = 0
    shifts = galerkin_shifts(A, E, W)
    # compute largest eigenvalue via Poweriteration or Lanczosprocess?
    Btol = np.linalg.norm(W.gramian(), ord=2) * options['tol']
    while(np.linalg.norm(W.gramian(), ord=2) > Btol and j < options['maxiter']):
        print(j, np.linalg.norm(W.gramian(), ord=2))
        if j >= shifts.size:
            if V is None:
                raise NotImplementedError
            else:
                shifts = np.append(shifts, galerkin_shifts(A, E, V))
        AaE = LincombOperator([A, E], [1, shifts[j]])
        V = AaE.apply_inverse(W)
        if shifts[j].imag == 0:
            W -= E.apply(V)*2*shifts[j].real
            Z.append(V*np.sqrt(-2*shifts[j].real))
            j += 1
        else:
            g = 2*np.sqrt(-shifts[j].real)
            d = shifts[j].real/shifts[j].imag
            W += E.apply(V.real + V.imag*d)*g**2
            Z.append((V.real + V.imag*d)*g)
            Z.append(V.imag*g*np.sqrt(d**2+1))
            j += 2
    return Z


def galerkin_shifts(A, E, V):
    """
    todo:
    -consider special cases (e.g. no shifts, pairs of complex shifts,...)
    -consider different orthonormalization strategy
    -get shifts without the matrix products
    -...
    """
    Q = gram_schmidt(V, atol=0, rtol=0)
    shifts = spla.eigvals(A.apply2(Q, Q), b=E.apply2(Q, Q))
    shifts = shifts[np.where(np.real(shifts < 0))]
    if shifts.size == 0:
        raise NotImplementedError
    else:
        return shifts


def lyap_solver_options(adi_tol=1e-10,
                        adi_maxiter=200):

    """Returns available Lyapunov equation solvers with default |solver_options|.

    Returns
    -------
    A dict of available solvers with default |solver_options|.
    """
    return {'adi_iteration': {'type': 'adi_iteration',
                              'tol': adi_tol,
                              'maxiter': adi_maxiter}}


def _solve_lyap_check_args(A, E, B, trans=False):
    assert isinstance(A, OperatorInterface) and A.linear
    assert A.source == A.range
    assert isinstance(B, OperatorInterface) and B.linear
    assert not trans and B.range == A.source or trans and B.source == A.source
    assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source
