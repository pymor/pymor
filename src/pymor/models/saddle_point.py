# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.models.basic import StationaryModel
from pymor.operators.block import BlockColumnOperator, BlockOperator
from pymor.operators.constructions import AdjointOperator, VectorOperator
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
        - |VectorOperator| on the ``u``-space.
        - |VectorArray| on the ``u``-space.
    g
        ``g`` of the right-hand side. Either
        - |VectorOperator| on the ``p``-space.
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
        assert isinstance(f, VectorOperator | VectorArray)
        assert isinstance(g, VectorOperator | VectorArray | None)
        assert isinstance(C, Operator | None)

        assert A.range == A.source
        assert A.source == B.source

        if C is not None:
            assert C.source == C.range == B.range

        operator = BlockOperator([[A, AdjointOperator(B)], [B, C]])

        if isinstance(f, VectorOperator):
            assert f.range == A.source
        else:
            assert f in A.source
            f = VectorOperator(f)

        if isinstance(g, VectorOperator):
            assert g.range == B.range
        elif isinstance(g, VectorArray):
            assert g in B.range
            g = VectorOperator(g)
        else:
            g = VectorOperator(B.range.zeros())

        rhs = BlockColumnOperator([f, g], name='rhs')

        assert isinstance(u_product, Operator | None)
        assert isinstance(p_product, Operator | None)
        tmp_products = {}

        if u_product is not None:
            assert u_product.range == u_product.source == A.range
            tmp_products['u'] = u_product

        if p_product is not None:
            assert p_product.range == p_product.source == B.range
            tmp_products['p'] = p_product

        products = tmp_products or None

        self.__auto_init(locals())
        super().__init__(operator=operator, rhs=rhs, products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)
