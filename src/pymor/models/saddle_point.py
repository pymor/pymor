# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.models.basic import StationaryModel
from pymor.operators.block import BlockColumnOperator, BlockOperator
from pymor.operators.constructions import AdjointOperator, VectorOperator
from pymor.operators.interface import Operator
from pymor.vectorarrays.block import BlockVectorArray
from pymor.vectorarrays.interface import VectorArray


class StationarySaddelPointModel(StationaryModel):
    r"""Generic class for Saddel point models.

    Constructs a stationary model for a linear saddle-point problems
    using a 2x2 |BlockOperator|:

    .. math::
        \begin{bmatrix}
            A & B^\ast \\
            B & C
        \\end{bmatrix}
        \begin{bmatrix}
            u \\ p
        \\end{bmatrix}
        =
        \begin{bmatrix}
            f \\ g
        \\end{bmatrix},

    Here, ``A`` is square on the ``u``-space, ``B`` maps from the ``u``-space
    to the ``p``-space, and ``C`` is optional (stabilization/mass) on the``p``-space.
    Typical example: Stokes equation with velocity (``u``) and pressure (``p``).

    Parameters
    ----------
    A
        |Operator| on the ``u``-space with ``A.source == A.range``.
    B
        Coupling |Operator| between ``u`` and ``p``-space with ``B.source == A.source``.
    C
        Optional mass |Operator| on the ``p``-space. One of:
        - |Operator| with ``C.source == C.range == B.range``.
        - ``None``, which yields a |ZeroOperator|.
    rhs
        Right-hand side. Several forms are supported and coerced to a
        |BlockColumnOperator| of shape ``(2, 1)``:
        - |BlockColumnOperator| with blocks ``[f, g]``.
        - |BlockVectorArray| with blocks ``[f, g]``.
        - |VectorOperator| on the ``u``-space. Then ``g`` is set to zero.
        - |VectorArray| on the ``u``-space. Then ``g`` is set to zero.

    products
        A dict of inner product |Operators| for the ``u`` and ``p`` space.
        Entries (when given) are |Operators| and stored under the keys
        ``'u'`` (velocity) and ``'p'`` (pressure). Missing entries remain ``None``.
    visualizer
        See :class:`~pymor.models.basic.StationaryModel`.
    name
        See :class:`~pymor.models.basic.StationaryModel`.
    """

    def __init__(self, A, B, rhs, C=None, products=None, visualizer=None, name=None):
        assert isinstance(A, Operator)
        assert isinstance(B, Operator)
        assert isinstance(rhs, BlockColumnOperator | VectorOperator | VectorArray | BlockVectorArray)
        assert isinstance(C, Operator | type(None))

        assert A.range == A.source
        assert A.source == B.source

        if isinstance(C, Operator):
            assert C.source == C.range == B.range

        operator = BlockOperator([[A, AdjointOperator(B)], [B, C]])

        if isinstance(rhs, BlockColumnOperator):
            assert rhs.blocks.shape == (2,1)
            assert rhs.range == operator.source
        elif isinstance(rhs, BlockVectorArray):
            assert rhs in operator.source
            rhs = BlockColumnOperator([VectorOperator(rhs.blocks[0]), VectorOperator(rhs.blocks[1])], name='rhs')
        elif isinstance(rhs, VectorOperator):
            assert rhs.range == A.source
            rhs = BlockColumnOperator([rhs, VectorOperator(B.range.zeros())], name='rhs')
        else:
            assert rhs in A.source
            rhs = BlockColumnOperator([VectorOperator(rhs), VectorOperator(B.range.zeros())], name='rhs')

        assert isinstance(products, dict | type(None))

        if isinstance(products, dict):
            allowed = {'u', 'p'}
            unknown = set(products.keys()) - allowed
            assert not unknown, f'Unknown product keys: {unknown}. Allowed: {allowed}'

            assert len(products) <= 2
            it = iter(products)
            key1 = next(it, None)
            key2 = next(it, None)

            product_1 = products[key1] if key1 else None
            product_2 = products[key2] if key2 else None

            assert isinstance(product_1, Operator | type(None))
            assert isinstance(product_2, Operator | type(None))

            if isinstance(product_1, Operator):
                assert product_1.range == product_1.source == A.range

            if isinstance(product_2, Operator):
                assert product_2.range == product_2.source == B.range

            tmp = {}
            if product_1 is not None:
                tmp['u'] = product_1
            if product_2 is not None:
                tmp['p'] = product_2

            products = tmp if tmp else None

        self.__auto_init(locals())
        super().__init__(operator=operator, rhs=rhs, products=products, visualizer=visualizer, name=name)
