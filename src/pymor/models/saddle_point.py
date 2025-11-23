# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.models.basic import StationaryModel
from pymor.operators.block import BlockColumnOperator, BlockDiagonalOperator, BlockOperator
from pymor.operators.constructions import AdjointOperator, IdentityOperator, VectorOperator
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray


class SaddlePointModel(StationaryModel):
    r"""Generic class for saddle-point models.

    Defines a stationary saddle-point model using a 2x2 |BlockOperator|:

    .. math::
        \begin{bmatrix}
            A & B^\ast \\
            B & C
        \end{bmatrix}
        \begin{bmatrix}
            u \\ p
        \end{bmatrix}
        =
        \begin{bmatrix}
            f \\ g
        \end{bmatrix},

    Here, ``A`` maps the ``u``-space into itself, ``B`` maps from the ``u``-space
    to the ``p``-space, and ``C`` is optional. If ``C`` is provided, it maps the
    ``p``-space into itself. Typical example: Stokes equation with velocity (``u``)
    and pressure (``p``).

    Parameters
    ----------
    A
        |Operator| on the ``u``-space with ``A.source == A.range``.
    B
        Coupling |Operator| between the ``u`` and ``p``-space with ``B.source == A.source``.
    C
        Optional |Operator| on the ``p``-space. One of:
        - |Operator| with ``C.source == C.range == B.range``.
        - ``None``, which yields a |ZeroOperator|.
    f
        ``f`` of the right-hand side. Either:
        - Linear |Operator| acting on the ``u``-space with a scalar source space.
        - |VectorArray| on the ``u``-space.
    g
        ``g`` of the right-hand side. Either
        - Linear |Operator| acting on the ``p``-space with a scalar source space.
        - |VectorArray| on the ``p``-space.
        - ``None``, then ``g`` is set to zero.
    u_product
        Inner product |Operator| acting on the ``u``-space.
    p_product
        Inner product |Operator| acting on the ``p``-space.
    error_estimator
        See :class:`~pymor.models.basic.StationaryModel`.
    visualizer
        See :class:`~pymor.models.basic.StationaryModel`.
    name
        See :class:`~pymor.models.basic.StationaryModel`.
    """

    def __init__(self, A, B, f, g=None, C=None, u_product=None, p_product=None,
                 error_estimator=None, visualizer=None, name=None):
        assert isinstance(A, Operator)
        assert isinstance(B, Operator)
        assert isinstance(f, Operator | VectorArray)
        assert isinstance(g, Operator | VectorArray | None)
        assert isinstance(C, Operator | None)

        assert A.range == A.source
        assert A.source == B.source

        if C is not None:
            assert C.source == C.range == B.range

        operator = BlockOperator([[A, AdjointOperator(B)], [B, C]])

        if isinstance(f, Operator):
            assert f.range == A.source
            assert f.linear
            assert f.source.is_scalar
        else:
            assert f in A.source
            f = VectorOperator(f)

        if isinstance(g, Operator):
            assert g.range == B.range
            assert g.linear
            assert g.source.is_scalar
        elif isinstance(g, VectorArray):
            assert g in B.range
            g = VectorOperator(g)
        else:
            g = VectorOperator(B.range.zeros())

        rhs = BlockColumnOperator([f, g], name='rhs')

        assert isinstance(u_product, Operator | None)
        assert isinstance(p_product, Operator | None)

        if u_product is not None:
            assert u_product.range == u_product.source == A.range
        if p_product is not None:
            assert p_product.range == p_product.source == B.range

        if u_product or p_product:
            blocks = [
                u_product if u_product else IdentityOperator(A.range),
                p_product if p_product else IdentityOperator(B.range)
            ]
            products = {'mixed': BlockDiagonalOperator(blocks=blocks)}
        else:
            products = None

        self.__auto_init(locals())
        super().__init__(operator=operator, rhs=rhs, products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)
