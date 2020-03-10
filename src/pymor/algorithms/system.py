# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np


from pymor.algorithms.projection import project
from pymor.algorithms.rules import RuleTable, match_class
from pymor.core.exceptions import RuleNotMatchingError
from pymor.operators.block import BlockOperator, BlockRowOperator, BlockColumnOperator
from pymor.operators.constructions import (Concatenation, VectorArrayOperator, ZeroOperator, LincombOperator)
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


def project_system(op, range_bases, source_bases, products=None):

    system_in_range = not isinstance(range_bases, (VectorArray, type(None)))
    system_in_source = not isinstance(source_bases, (VectorArray, type(None)))

    if system_in_range:
        assert all(rb is None or rb in ss for rb, ss in zip(range_bases, op.range.subspaces))
        products = products or [None] * len(range_bases)
        assert all(p is None or p.source == p.range == ss for p, ss in zip(products, range_bases))
    else:
        assert range_bases is None or range_bases in op.range
        products = products or None
        assert products is None or products.source == products.range == op.range

    if system_in_source:
        assert all(sb is None or sb in ss for sb, ss in zip(source_bases, op.source.subspaces))
    else:
        assert source_bases is None or source_bases in op.source

    return ProjectSystemRules(range_bases, source_bases,
                              system_in_range=system_in_range,
                              system_in_source=system_in_source,
                              products=products).apply(op)


class ProjectSystemRules(RuleTable):

    def __init__(self, range_bases, source_bases, system_in_range, system_in_source, products):
        super().__init__(True)
        self.range_bases, self.source_bases, self.system_in_range, self.system_in_source, self.products = \
            range_bases, source_bases, system_in_range, system_in_source, products

    @match_class(Operator)
    def action_no_system(self, op):
        if self.system_in_range or self.system_in_source:
            raise RuleNotMatchingError
        return project(op, self.range_bases, self.source_bases, product=self.products)

    @match_class(BlockOperator)
    def action_BlockOperator(self, op):
        range_bases, source_bases, system_in_range, system_in_source, products = \
            self.range_bases, self.source_bases, self.system_in_range, self.system_in_source, self.products
        if not system_in_source:
            raise NotImplementedError
        if not system_in_range:
            raise NotImplementedError

        def project_block(i, j, block):
            return project(block, range_bases[i], source_bases[j], products[i])

        projected_blocks = np.array([project_block(i, j, block) if not isinstance(block, ZeroOperator) else None
                                     for (i, j), block in np.ndenumerate(op.blocks)])
        projected_blocks.shape = op.blocks.shape
        return BlockOperator(blocks=projected_blocks)

    @match_class(BlockRowOperator)
    def action_BlockRowOperator(self, op):
        range_bases, source_bases, system_in_range, system_in_source, products = \
            self.range_bases, self.source_bases, self.system_in_range, self.system_in_source, self.products
        assert not system_in_range
        assert system_in_source  # Other case is handled by action_no_system

        def project_block(j, block):
            return project(block, range_bases, source_bases[j], products)

        projected_blocks = [project_block(j, o) if not isinstance(o, ZeroOperator) else None
                            for j, o in enumerate(op.blocks)]
        return BlockRowOperator(projected_blocks)

    @match_class(BlockColumnOperator)
    def action_BlockColumnOperator(self, op):
        range_bases, source_bases, system_in_range, system_in_source, products = \
            self.range_bases, self.source_bases, self.system_in_range, self.system_in_source, self.products
        assert not system_in_source
        assert system_in_range  # Other case is handled by action_no_system

        def project_block(i, block):
            return project(block, range_bases[i], source_bases, product=products[i])

        projected_blocks = [project_block(i, o) if not isinstance(o, ZeroOperator) else None
                            for i, o in enumerate(op.blocks)]
        return BlockColumnOperator(projected_blocks)

    @match_class(VectorArrayOperator)
    def action_VectorArrayOperator(self, op):
        range_bases, source_bases, system_in_range, system_in_source, products = \
            self.range_bases, self.source_bases, self.system_in_range, self.system_in_source, self.products
        if not op.transposed:
            assert not system_in_source
            if source_bases is not None:
                raise NotImplementedError
            assert system_in_range  # Other case is handled by action_no_system

            def inner(U, V, product=None):
                return U.dot(V) if product is None else product.apply2(U, V)

            projected_array = BlockVectorSpace.make_array(
                [NumpyVectorSpace.from_data(inner(block, range_basis, product), block.space.id)
                 for block, range_basis, product in zip(op._array._blocks, range_bases, products)]
            )
        else:
            assert not system_in_range
            if range_bases is not None:
                raise NotImplementedError
            assert system_in_source  # Other case is handled by action_no_system

            projected_array = BlockVectorSpace.make_array(
                [NumpyVectorSpace.from_data(block.dot(source_basis), block.space.id)
                 for block, source_basis in zip(op._array._blocks, source_bases)]
            )
        return VectorArrayOperator(projected_array, transposed=op.transposed, space_id=op.space_id,
                                   name=op.name)

    @match_class(LincombOperator)
    def action_recurse(self, op):
        return self.replace_children(op)

    @match_class(Concatenation)
    def action_Concatenation(self, op):
        """project linear non-parametric Concatenations"""
        range_bases, source_bases, system_in_range, system_in_source, products = \
            self.range_bases, self.source_bases, self.system_in_range, self.system_in_source, self.products

        if len(op.operators) == 1:
            return self.apply(op.operators[0])
        if op.parametric or not op.linear:
            raise RuleNotMatchingError('Only implemented for non-parametric linear Concatenations')

        second, first = Concatenation(op.operators[:-1]), op.operators[-1]
        projected_first = project_system(first, None, source_bases, products=None)

        if isinstance(projected_first, VectorArrayOperator):
            assert not system_in_source
            blocks = np.array([[projected_first]])
            second_system_in_source = False
        elif isinstance(projected_first, BlockColumnOperator):
            assert not system_in_source
            blocks = projected_first.blocks.reshape((-1, 1))
            second_system_in_source = True
        elif isinstance(projected_first, BlockRowOperator):
            blocks = projected_first.blocks.reshape((1, -1))
            second_system_in_source = False
        elif isinstance(projected_first, BlockOperator):
            blocks = projected_first.blocks
            second_system_in_source = True
        else:
            raise NotImplementedError

        if not all(isinstance(o, (VectorArrayOperator, ZeroOperator)) for o in blocks.ravel()):
            raise NotImplementedError

        if system_in_range:
            second_blocks = np.full((len(second.range.subspaces), blocks.shape[1]), None)
        else:
            second_blocks = np.full((1, blocks.shape[1]), None)

        for j in range(second_blocks.shape[1]):
            if not second_system_in_source:
                sb = blocks[0, j]
                if not isinstance(sb, ZeroOperator):
                    sb = sb.as_range_array()
                    projected_second = project_system(second, range_bases, sb, products=products)
                    if isinstance(projected_second, (VectorArrayOperator, NumpyMatrixOperator)):
                        assert not system_in_range
                        second_blocks[0, j] = projected_second
                    elif isinstance(projected_second, BlockColumnOperator):
                        second_blocks[:, j] = projected_second.blocks
                    else:
                        raise NotImplementedError

            else:
                sb = [b.as_range_array() if isinstance(b, VectorArrayOperator) else b.range.empty()
                      for b in blocks[:, j]]

                if isinstance(projected_second, BlockRowOperator):
                    assert not system_in_range
                    ops = [o for o, so in zip(projected_second.blocks, blocks[:, j])
                           if not isinstance(so, ZeroOperator)]
                    o = LincombOperator(ops, [1.] * len(ops)).assemble()
                    second_blocks[0, j] = o
                elif isinstance(projected_second, BlockOperator):
                    for i in range(len(projected_second.range.subspaces)):
                        ops = [o for o, so in zip(projected_second.blocks, blocks[i, j])
                               if not isinstance(so, ZeroOperator)]
                        o = LincombOperator(ops, [1.] * len(ops)).assemble()
                        second_blocks[i, j] = o
                else:
                    raise NotImplementedError

        for (i, j) in np.ndindex(second_blocks.shape):
            ss = projected_first.source.subspaces[j] if system_in_source else projected_first.source
            o = second_blocks[i, j]
            second_blocks[i, j] = \
                None if o is None else \
                o.with_(source=ss) if isinstance(o, ZeroOperator) else \
                o.with_(space_id=ss.id) if isinstance(o, VectorArrayOperator) else \
                o.with_(source_id=ss.id)

        if system_in_source:
            if system_in_range:
                return BlockOperator(second_blocks)
            else:
                return BlockRowOperator(second_blocks.ravel())
        else:
            if system_in_range:
                return BlockColumnOperator(second_blocks.ravel())
            else:
                assert False
