# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.constructions import LincombOperator, VectorArrayOperator


def arnoldi(A, E, b, sigma, trans=False):
    r"""Rational Arnoldi algorithm.

    If `trans == False`, using Arnoldi process, computes a real
    orthonormal basis for the rational Krylov subspace

    .. math::
        \span\{(\sigma_1 E - A)^{-1} b, (\sigma_2 E - A)^{-1} b, \ldots,
        (\sigma_r E - A)^{-1} b\},

    otherwise, computes the same for

    .. math::
        \span\{(\sigma_1 E - A)^{-*} b^*, (\sigma_2 E - A)^{-*} b^*,
        \ldots, (\sigma_r E - A)^{-*} b^*\}.

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
            sEmA = LincombOperator((E, A), (sigma[i].real, -1))

            if not trans:
                v = sEmA.apply_inverse(v)
            else:
                v = sEmA.apply_inverse_transpose(v)

            if i > 0:
                v_norm_orig = v.l2_norm()[0]
                Vop = VectorArrayOperator(V)
                v -= Vop.apply(Vop.apply_transpose(v))
                if v.l2_norm()[0] < v_norm_orig / 10:
                    v -= Vop.apply(Vop.apply_transpose(v))
            v.scal(1 / v.l2_norm()[0])
            V.append(v)
        elif sigma[i].imag > 0:
            sEmA = LincombOperator((E, A), (sigma[i], -1))

            if not trans:
                v = sEmA.apply_inverse(v)
            else:
                v = sEmA.apply_inverse_transpose(v)

            v1 = v.real
            if i > 0:
                v1_norm_orig = v1.l2_norm()[0]
                Vop = VectorArrayOperator(V)
                v1 -= Vop.apply(Vop.apply_transpose(v1))
                if v1.l2_norm()[0] < v1_norm_orig / 10:
                    v1 -= Vop.apply(Vop.apply_transpose(v1))
            v1.scal(1 / v1.l2_norm()[0])
            V.append(v1)

            v2 = v.imag
            v2_norm_orig = v2.l2_norm()[0]
            Vop = VectorArrayOperator(V)
            v2 -= Vop.apply(Vop.apply_transpose(v2))
            if v2.l2_norm()[0] < v2_norm_orig / 10:
                v2 -= Vop.apply(Vop.apply_transpose(v2))
            v2.scal(1 / v2.l2_norm()[0])
            V.append(v2)

            v = v2

    return V
