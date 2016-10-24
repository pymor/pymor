# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.constructions import LincombOperator, VectorArrayOperator
from pymor.operators.numpy import NumpyMatrixOperator


def arnoldi(A, E, b, sigma, trans=False):
    r"""Rational Arnoldi algorithm.

    If `trans == False`, using Arnoldi process, computes a real orthonormal
    basis for the rational Krylov subspace

    .. math::
        \span\{(\sigma_1 E - A)^{-1} b, (\sigma_2 E - A)^{-1} b, \ldots,
        (\sigma_r E - A)^{-1} b\},

    otherwise, computes the same for

    .. math::
        \span\{(\sigma_1 E - A)^{-*} b^*, (\sigma_2 E - A)^{-*} b^*, \ldots,
        (\sigma_r E - A)^{-*} b^*\}.

    Interpolation points in `sigma` are allowed to repeat (in any order).
    Then, in the above expression,

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
        Real vector-like operator (if trans is False) or functional
        (if trans is True).
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
    V = A.source.type.make_array(A.source.subtype, reserve=r)

    v = b.as_vector()
    v.scal(1 / v.l2_norm()[0])

    for i in range(r):
        if sigma[i].imag == 0:
            sEmA = LincombOperator((E, A), (sigma[i].real, -1))

            if not trans:
                v = sEmA.apply_inverse(v)
            else:
                v = sEmA.apply_inverse_adjoint(v)

            if i > 0:
                v_norm_orig = v.l2_norm()[0]
                Vop = VectorArrayOperator(V)
                v -= Vop.apply(Vop.apply_adjoint(v))
                if v.l2_norm()[0] < v_norm_orig / 10:
                    v -= Vop.apply(Vop.apply_adjoint(v))
            v.scal(1 / v.l2_norm()[0])
            V.append(v)
        elif sigma[i].imag > 0:
            sEmA = LincombOperator((E, A), (sigma[i], -1))

            if not trans:
                v = sEmA.apply_inverse(v)
            else:
                v = sEmA.apply_inverse_adjoint(v)

            v1 = v.real
            if i > 0:
                v1_norm_orig = v1.l2_norm()[0]
                Vop = VectorArrayOperator(V)
                v1 -= Vop.apply(Vop.apply_adjoint(v1))
                if v1.l2_norm()[0] < v1_norm_orig / 10:
                    v1 -= Vop.apply(Vop.apply_adjoint(v1))
            v1.scal(1 / v1.l2_norm()[0])
            V.append(v1)

            v2 = v.imag
            v2_norm_orig = v2.l2_norm()[0]
            Vop = VectorArrayOperator(V)
            v2 -= Vop.apply(Vop.apply_adjoint(v2))
            if v2.l2_norm()[0] < v2_norm_orig / 10:
                v2 -= Vop.apply(Vop.apply_adjoint(v2))
            v2.scal(1 / v2.l2_norm()[0])
            V.append(v2)

            v = v2

    return V


def arnoldi_tangential(A, E, B, sigma, directions, trans=False):
    r"""Tangential Rational Arnoldi algorithm.

    If `trans == False`, using tangential Arnoldi process [DSZ14]_,
    computes a real orthonormal basis for the rational Krylov subspace

    .. math::
        \span\{(\sigma_1 E - A)^{-1} B d_1, (\sigma_2 E - A)^{-1} B d_2,
        \ldots, (\sigma_r E - A)^{-1} B d_r\},

    otherwise, computes the same for

    .. math::
        \span\{(\sigma_1 E - A)^{-*} B^* d_1, (\sigma_2 E - A)^{-*} B^* d_2,
        \ldots, (\sigma_r E - A)^{-*} B^* d_r\}.

    Interpolation points in `sigma` are allowed to repeat (in any order).
    Then, in the above expression,

    .. math::
        \underbrace{(\sigma_i E - A)^{-1} B^* d_i, \ldots,
        (\sigma_i E - A)^{-1} B^* d_i}_{m \text{ times}}

    is replaced by

    .. math::
        (\sigma_i E - A)^{-1} B^* d_i, (\sigma_i E - A)^{-2} B^* d_i,
        \ldots, (\sigma_i E - A)^{-m} B^* d_i.

    Analogously for the `trans == True` case.

    .. warning::
        The implementation is still experimental.

    .. [DSZ14] V. Druskin, V. Simoncini, M. Zaslavsky, Adaptive Tangential
               Interpolation in Rational Krylov Subspaces for MIMO Dynamical
               Systems,
               SIAM Journal on Matrix Analysis and Applications, 35(2),
               476-498, 2014.

    Parameters
    ----------
    A
        Real |Operator| A.
    E
        Real |Operator| E.
    B
        Real |Operator| B.
    sigma
        Interpolation points (closed under conjugation).
    directions
        Tangential directions (closed under conjugation), |VectorArray| of
        length `len(sigma)` from `B.source` or `B.range` (depending on
        `trans`).
    trans
        Boolean, see above.

    Returns
    -------
    V
        Projection matrix.
    """
    r = len(sigma)
    assert len(directions) == r
    assert (not trans and B.source.dim > 1 and directions in B.source or
            trans and B.range.dim > 1 and directions in B.range)

    directions.scal(1 / directions.l2_norm())

    V = A.source.type.make_array(A.source.subtype, reserve=r)

    for i in range(r):
        if sigma[i].imag == 0:
            sEmA = LincombOperator((E, A), (sigma[i].real, -1))

            if not trans:
                if i == 0:
                    v = sEmA.apply_inverse(B.apply(directions.real[0]))
                else:
                    Bd = B.apply(directions.real[i])
                    VTBd = VectorArrayOperator(V, transposed=True).apply(Bd)
                    sEmA_proj_inv_VTBd = NumpyMatrixOperator(sEmA.apply2(V, V)).apply_inverse(VTBd)
                    V_sEmA_proj_inv_VTBd = VectorArrayOperator(V).apply(sEmA_proj_inv_VTBd)
                    rd = sEmA.apply(V_sEmA_proj_inv_VTBd) - Bd
                    v = sEmA.apply_inverse(rd)
            else:
                if i == 0:
                    v = sEmA.apply_inverse_adjoint(B.apply_adjoint(directions.real[0]))
                else:
                    CTd = B.apply_adjoint(directions.real[i])
                    VTCTd = VectorArrayOperator(V, transposed=True).apply(CTd)
                    sEmA_proj_inv_VTCTd = NumpyMatrixOperator(sEmA.apply2(V, V)).apply_inverse_adjoint(VTCTd)
                    V_sEmA_proj_inv_VTCTd = VectorArrayOperator(V).apply(sEmA_proj_inv_VTCTd)
                    rd = sEmA.apply_adjoint(V_sEmA_proj_inv_VTCTd) - CTd
                    v = sEmA.apply_inverse_adjoint(rd)

            if i > 0:
                v_norm_orig = v.l2_norm()[0]
                Vop = VectorArrayOperator(V)
                v -= Vop.apply(Vop.apply_adjoint(v))
                if v.l2_norm()[0] < v_norm_orig / 10:
                    v -= Vop.apply(Vop.apply_adjoint(v))
            v.scal(1 / v.l2_norm()[0])
            V.append(v)
        elif sigma[i].imag > 0:
            sEmA = LincombOperator((E, A), (sigma[i], -1))

            if not trans:
                if i == 0:
                    v = sEmA.apply_inverse(B.apply(directions[0]))
                else:
                    Bd = B.apply(directions[i])
                    VTBd = VectorArrayOperator(V, transposed=True).apply(Bd)
                    sEmA_proj_inv_VTBd = NumpyMatrixOperator(sEmA.apply2(V, V)).apply_inverse(VTBd)
                    V_sEmA_proj_inv_VTBd = VectorArrayOperator(V).apply(sEmA_proj_inv_VTBd)
                    rd = sEmA.apply(V_sEmA_proj_inv_VTBd) - Bd
                    v = sEmA.apply_inverse(rd)
            else:
                if i == 0:
                    v = sEmA.apply_inverse_adjoint(B.apply_adjoint(directions[0]))
                else:
                    CTd = B.apply_adjoint(directions[i])
                    VTCTd = VectorArrayOperator(V, transposed=True).apply(CTd)
                    sEmA_proj_inv_VTCTd = NumpyMatrixOperator(sEmA.apply2(V, V)).apply_inverse_adjoint(VTCTd)
                    V_sEmA_proj_inv_VTCTd = VectorArrayOperator(V).apply(sEmA_proj_inv_VTCTd)
                    rd = sEmA.apply_adjoint(V_sEmA_proj_inv_VTCTd) - CTd
                    v = sEmA.apply_inverse_adjoint(rd)

            v1 = v.real
            if i > 0:
                v1_norm_orig = v1.l2_norm()[0]
                Vop = VectorArrayOperator(V)
                v1 -= Vop.apply(Vop.apply_adjoint(v1))
                if v1.l2_norm()[0] < v1_norm_orig / 10:
                    v1 -= Vop.apply(Vop.apply_adjoint(v1))
            v1.scal(1 / v1.l2_norm()[0])
            V.append(v1)

            v2 = v.imag
            v2_norm_orig = v2.l2_norm()[0]
            Vop = VectorArrayOperator(V)
            v2 -= Vop.apply(Vop.apply_adjoint(v2))
            if v2.l2_norm()[0] < v2_norm_orig / 10:
                v2 -= Vop.apply(Vop.apply_adjoint(v2))
            v2.scal(1 / v2.l2_norm()[0])
            V.append(v2)

            v = v2

    return V
