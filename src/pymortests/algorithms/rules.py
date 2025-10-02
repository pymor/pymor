# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typing import ClassVar

import numpy as np
import pytest

from pymor.algorithms.rules import RuleTable, match_class
from pymor.operators.block import BlockOperator
from pymor.operators.constructions import AdjointOperator, ZeroOperator
from pymor.operators.numpy import NumpyMatrixOperator

pytestmark = pytest.mark.builtin


def test_get_children_includes_blocks_for_blockoperator():
    A = NumpyMatrixOperator(np.eye(2))
    B = NumpyMatrixOperator(2 * np.eye(2))
    blocks = np.array([[A, B],
                       [B, None]], dtype=object)
    bop = BlockOperator(blocks)

    children = RuleTable.get_children(bop)

    # Sanity: RuleTable sees the blocks child
    assert 'blocks' in children

    # Custom table: transform each NumpyMatrixOperator child into its adjoint
    class AdjointNumpyOpTable(RuleTable):
        @match_class(ZeroOperator)
        def action_ZeroOperator(self, op):
            return op

        @match_class(NumpyMatrixOperator)
        def action_NumpyMatrixOperator(self, op):
            return AdjointOperator(op)

    tbl = AdjointNumpyOpTable()
    bop_applied = tbl.apply_children(bop, children=children)
    assert isinstance(bop_applied, dict)
    assert 'blocks' in bop_applied
    new_blocks = bop_applied['blocks']

    assert isinstance(new_blocks, np.ndarray)
    assert new_blocks.dtype == object
    assert new_blocks.shape == blocks.shape

    for idx, b in np.ndenumerate(bop.blocks):
        nb = new_blocks[idx]
        if isinstance(b, ZeroOperator):
            assert isinstance(nb, ZeroOperator)
        else:
            assert isinstance(nb, AdjointOperator), f'expected AdjointOperator at {idx}, got {type(nb)}'

    # Original must be unchanged (apply_children copies)
    for idx, b in np.ndenumerate(bop.blocks):
        assert bop.blocks[idx] is b


def test_get_children_excludes_all_none_object_array():
    """An np.ndarray(dtype=object) containing only None should NOT be treated as children."""
    class Dummy:
        _init_arguments: ClassVar[list[str]] = ['blocks']
        def __init__(self):
            self.blocks = np.array([[None, None], [None, None]], dtype=object)

    obj = Dummy()
    children = RuleTable.get_children(obj)
    assert 'blocks' not in children, f"'blocks' should be excluded (only None), got {children}"
