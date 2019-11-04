# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.gram_schmidt import gram_schmidt


def rational_arnoldi(A, E, b, sigma, trans=False):
    r"""Rational Arnoldi algorithm.

    If `trans == False`, using Arnoldi process, computes a real
    orthonormal basis for the rational Krylov subspace

    .. math::
        \mathrm{span}\{(\sigma_1 E - A)^{-1} b, (\sigma_2 E - A)^{-1} b, \ldots,
        (\sigma_r E - A)^{-1} b\},

    otherwise, computes the same for

    .. math::
        \mathrm{span}\{(\sigma_1 E - A)^{-T} b^T, (\sigma_2 E - A)^{-T} b^T,
        \ldots, (\sigma_r E - A)^{-T} b^T\}.

    Interpolation points in `sigma` are allowed to repeat (in any
    order). Then, in the above expression,

    .. math::
        \underbrace{(\sigma_i E - A)^{-1} b, \ldots,
        (\sigma_i E - A)^{-1} b}_{m \text{ times}}

    is replaced by

    .. math::
        (\sigma_i E - A)^{-1} b, (\sigma_i E - A)^{-2} b, \ldots,
        (\sigma_i E - A)^{-m} b.

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
        Interpolation points (closed under conjugation).
    trans
        Boolean, see above.

    Returns
    -------
    V
        Projection matrix.
    """
    assert not trans and b.source.dim == 1 or trans and b.range.dim == 1

    r = len(sigma)
    V = A.source.empty(reserve=r)

    v = b.as_vector()
    v.scal(1 / v.l2_norm()[0])

    for i in range(r):
        if sigma[i].imag == 0:
            sEmA = sigma[i].real * E - A

            if not trans:
                v = sEmA.apply_inverse(v if len(V) == 0 else E.apply(v))
            else:
                v = sEmA.apply_inverse_adjoint(v if len(V) == 0 else E.apply_adjoint(v))

            V.append(v)
            V = gram_schmidt(V, atol=0, rtol=0, offset=len(V) - 1, copy=False)
            v = V[-1]
        elif sigma[i].imag > 0:
            sEmA = sigma[i] * E - A

            if not trans:
                v = sEmA.apply_inverse(v if len(V) == 0 else E.apply(v))
            else:
                v = sEmA.apply_inverse_adjoint(v if len(V) == 0 else E.apply_adjoint(v))

            V.append(v.real)
            V.append(v.imag)
            V = gram_schmidt(V, atol=0, rtol=0, offset=len(V) - 2, copy=False)
            v = V[-1]

    return V
