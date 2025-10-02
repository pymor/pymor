# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Module for computing (rational) Krylov subspaces' bases."""

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.logger import getLogger
from pymor.operators.constructions import IdentityOperator


def arnoldi(A, E, b, r):
    r"""Block Arnoldi algorithm.

    Computes an orthonormal basis for the Krylov subspace

    .. math::
        \mathrm{span}\left\{
            E^{-1} b,
            E^{-1} A E^{-1} b,
            {\left(E^{-1} A\right)}^2 E^{-1} b,
            \ldots,
            {\left(E^{-1} A\right)}^{r - 1} E^{-1} b,
        \right\}.

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E. If `None`, the identity operator is assumed.
    b
        The |VectorArray| b.
    r
        Order of the Krylov subspace (positive integer).

    Returns
    -------
    V
        Orthonormal basis for the Krylov subspace as a |VectorArray|
        with `len(V) <= r*len(b)` (strict inequality in case of
        deflation).
    """
    assert A.source == A.range
    if E is None:
        E = IdentityOperator(A.source)
    assert E.source == A.source
    assert E.range == A.source
    assert b in A.source
    assert len(b) > 0

    logger = getLogger('pymor.algorithms.krylov.arnoldi')

    V = A.source.empty(reserve=r*len(b))

    v = E.apply_inverse(b)
    V.append(v, remove_from_other=True)
    gram_schmidt(V, atol=0, rtol=0, copy=False)
    if len(V) < len(b):
        logger.warning('gram_schmidt removed vectors.')
    last_block_len = len(V)

    for i in range(1, r):
        if last_block_len == 0:
            logger.warning('Last block is empty. Returning.')
            return V

        v = A.apply(V[-last_block_len:])
        v = E.apply_inverse(v)
        len_v, len_V = len(v), len(V)
        V.append(v, remove_from_other=True)
        gram_schmidt(V, atol=0, rtol=0, offset=len_V, copy=False)
        last_block_len = len(V) - len_V

        if last_block_len < len_v:
            logger.warning('gram_schmidt removed vectors.')

    return V


def rational_arnoldi(A, E, b, sigma, trans=False, shifted_system_solver=None):
    r"""Rational Arnoldi algorithm.

    If `trans == False`, using Arnoldi process, computes a real
    orthonormal basis for the rational Krylov subspace

    .. math::
        \mathrm{span}\{
            (\sigma_1 E - A)^{-1} b,
            (\sigma_2 E - A)^{-1} b,
            \ldots,
            (\sigma_r E - A)^{-1} b
        \},

    otherwise, computes the same for

    .. math::
        \mathrm{span}\{
            (\sigma_1 E - A)^{-T} b^T,
            (\sigma_2 E - A)^{-T} b^T,
            \ldots,
            (\sigma_r E - A)^{-T} b^T
        \}.

    Interpolation points in `sigma` are allowed to repeat (in any
    order). Then, in the above expression,

    .. math::
        \underbrace{
            (\sigma_i E - A)^{-1} b,
            \ldots,
            (\sigma_i E - A)^{-1} b
        }_{m \text{ times}}

    is replaced by

    .. math::
        (\sigma_i E - A)^{-1} b,
        (\sigma_i E - A)^{-1} E (\sigma_i E - A)^{-1} b,
        \ldots,
        \left((\sigma_i E - A)^{-1} E\right)^{m - 1} (\sigma_i E - A)^{-1} b.

    Analogously for the `trans == True` case.

    Parameters
    ----------
    A
        Real |Operator| A.
    E
        Real |Operator| E.
    b
        Real vector-like operator (if trans is False) or functional (if
        trans is True).
    sigma
        Sequence of interpolation points (closed under conjugation).
    trans
        Boolean, see above.
    shifted_system_solver
        The |Solver| for the shifted systems.

    Returns
    -------
    V
        Orthonormal basis for the Krylov subspace as a |VectorArray|.
    """
    assert A.source == A.range
    assert E.source == A.source
    assert E.range == A.source
    assert (b.range if not trans else b.source) == A.source
    assert not trans and b.source.dim == 1 or trans and b.range.dim == 1

    r = len(sigma)
    V = A.source.empty(reserve=r)

    v = b.as_vector()
    v.scal(1 / v.norm()[0])

    for i in range(r):
        if sigma[i].imag < 0:
            continue
        if sigma[i].imag == 0:
            sEmA = sigma[i].real * E - A
        else:
            sEmA = sigma[i] * E - A
        if not trans:
            v = sEmA.apply_inverse(v if len(V) == 0 else E.apply(v), solver=shifted_system_solver)
        else:
            v = sEmA.apply_inverse_adjoint(v if len(V) == 0 else E.apply_adjoint(v), solver=shifted_system_solver)
        if sigma[i].imag == 0:
            V.append(v)
            gram_schmidt(V, atol=0, rtol=0, offset=len(V) - 1, copy=False)
        else:
            V.append(v.real)
            V.append(v.imag)
            gram_schmidt(V, atol=0, rtol=0, offset=len(V) - 2, copy=False)
        v = V[-1]

    return V


def tangential_rational_krylov(A, E, B, b, sigma, trans=False, orth=True, shifted_system_solver=None):
    r"""Tangential Rational Krylov subspace.

    If `trans == False`, computes a real basis for the rational Krylov
    subspace

    .. math::
        \mathrm{span}\{
            (\sigma_1 E - A)^{-1} B b_1,
            (\sigma_2 E - A)^{-1} B b_2,
            \ldots,
            (\sigma_r E - A)^{-1} B b_r
        \},

    otherwise, computes the same for

    .. math::
        \mathrm{span}\{
            (\sigma_1 E - A)^{-T} B^T b_1,
            (\sigma_2 E - A)^{-T} B^T b_2,
            \ldots,
            (\sigma_r E - A)^{-T} B^T b_r
        \}.

    Interpolation points in `sigma` are assumed to be pairwise distinct.

    Parameters
    ----------
    A
        Real |Operator| A.
    E
        Real |Operator| E.
    B
        Real |Operator| B.
    b
        |VectorArray| from `B.source`, if `trans == False`, or
         `B.range`, if `trans == True`.
    sigma
        Sequence of interpolation points (closed under conjugation), of
        the same length as `b`.
    trans
        Boolean, see above.
    orth
        If `True`, orthonormalizes the basis using
        :meth:`pymor.algorithms.gram_schmidt.gram_schmidt`.
    shifted_system_solver
        The |Solver| to use for the shifted systems.

    Returns
    -------
    V
        Optionally orthonormal basis for the Krylov subspace as a |VectorArray|.
    """
    assert A.source == A.range
    assert E.source == A.source
    assert E.range == A.source
    assert (B.range if not trans else B.source) == A.source
    assert b in (B.source if not trans else B.range)
    assert len(b) == len(sigma)

    r = len(sigma)
    V = A.source.empty(reserve=r)
    for i in range(r):
        if sigma[i].imag == 0:
            sEmA = sigma[i].real * E - A
            if not trans:
                Bb = B.apply(b.real[i])
                V.append(sEmA.apply_inverse(Bb, solver=shifted_system_solver))
            else:
                BTb = B.apply_adjoint(b.real[i])
                V.append(sEmA.apply_inverse_adjoint(BTb, solver=shifted_system_solver))
        elif sigma[i].imag > 0:
            sEmA = sigma[i] * E - A
            if not trans:
                Bb = B.apply(b[i])
                v = sEmA.apply_inverse(Bb, solver=shifted_system_solver)
            else:
                BTb = B.apply_adjoint(b[i].conj())
                v = sEmA.apply_inverse_adjoint(BTb, solver=shifted_system_solver)
            V.append(v.real)
            V.append(v.imag)
    if orth:
        gram_schmidt(V, atol=0, rtol=0, copy=False)
    return V
