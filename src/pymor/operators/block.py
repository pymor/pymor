# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.operators.basic import OperatorBase
from pymor.operators.interfaces import OperatorInterface
from pymor.vectorarrays.block import BlockVectorArray
from pymor.vectorarrays.interfaces import VectorSpace


class BlockOperator(OperatorBase):
    """A sparse matrix of arbitrary operators

    Parameters
    ----------
    blocks
        Two-dimensional |NumPy| array where each entry is an operator or None.
    """

    def _operators(self):
        """Iterate over operators (not None)"""
        for row in self._blocks:
            for entry in row:
                if entry is not None:
                    yield entry

    def __init__(self, blocks):
        blocks = np.array(blocks)
        assert isinstance(blocks, np.ndarray) and blocks.ndim == 2
        self._blocks = blocks

        assert all(isinstance(op, OperatorInterface) for op in self._operators())
        assert all(any(self._blocks[i, j] is not None for j in range(self._blocks.shape[1]))
                   for i in range(self._blocks.shape[0]))
        assert all(any(self._blocks[i, j] is not None for i in range(self._blocks.shape[0]))
                   for j in range(self._blocks.shape[1]))

        source_types = [None for j in range(self._blocks.shape[1])]
        range_types = [None for i in range(self._blocks.shape[0])]
        for (i, j), op in np.ndenumerate(self._blocks):
            if op is not None:
                assert source_types[j] is None or op.source == source_types[j]
                source_types[j] = op.source
                assert range_types[i] is None or op.range == range_types[i]
                range_types[i] = op.range

        self.source = VectorSpace(BlockVectorArray, tuple(source_types))
        self.range = VectorSpace(BlockVectorArray, tuple(range_types))
        self._source_dims = tuple(space.dim for space in self.source.subtype)
        self._range_dims = tuple(space.dim for space in self.range.subtype)
        self.num_source_blocks = len(source_types)
        self.num_range_blocks = len(range_types)
        self.linear = all(op.linear for op in self._operators())
        self.build_parameter_type(inherits=list(self._operators()))

    @classmethod
    def hstack(cls, operators):
        """Horizontal stacking of operators

        Parameters
        ----------
        operators
            A tuple, list, array, or iterable of operators.
        """
        blocks = np.array([[op for op in operators]])
        return cls(blocks)

    @classmethod
    def vstack(cls, operators):
        """Vertical stacking of operators

        Parameters
        ----------
        operators
            A tuple, list, array, or iterable of operators.
        """
        blocks = np.array([[op] for op in operators])
        return cls(blocks)

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        assert U.check_ind(ind)

        blocks = [None for i in range(self.num_range_blocks)]
        for i in range(self.num_range_blocks):
            for j in range(self.num_source_blocks):
                op = self._blocks[i, j]
                if op is not None:
                    V = op.apply(U.block(j), ind=ind, mu=mu)
                    if blocks[i] is None:
                        blocks[i] = V
                    else:
                        blocks[i] += V

        return BlockVectorArray(blocks)

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        assert U in self.range
        assert U.check_ind(ind)
        assert source_product is None or source_product.source == source_product.range == self.source
        assert range_product is None or range_product.source == range_product.range == self.range

        if range_product is not None:
            U = range_product.apply(U)

        blocks = [None for j in range(self.num_source_blocks)]
        for j in range(self.num_source_blocks):
            for i in range(self.num_range_blocks):
                op = self._blocks[i, j]
                if op is not None:
                    V = op.apply_adjoint(U.block(i), ind=ind, mu=mu)
                    if blocks[j] is None:
                        blocks[j] = V
                    else:
                        blocks[j] += V

        V = BlockVectorArray(blocks)
        if source_product is not None:
            V = source_product.apply_inverse(V)

        return V


class BlockDiagonalOperator(BlockOperator):
    """Block diagonal operator with arbitrary operators"""

    def __init__(self, blocks):
        n = len(blocks)
        blocks2 = np.array([[None for j in range(n)] for i in range(n)])
        for i, op in enumerate(blocks):
            blocks2[i, i] = op
        super(BlockDiagonalOperator, self).__init__(blocks2)

    def apply_inverse(self, V, ind=None, mu=None, least_squares=False):
        assert V in self.range
        assert V.check_ind(ind)

        U = [None for i in range(self.num_source_blocks)]
        for i in range(self.num_source_blocks):
            U[i] = self._blocks[i, i].apply_inverse(V.block(i), ind=ind, mu=mu, least_squares=least_squares)

        return BlockVectorArray(U)

    def apply_inverse_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None,
                              least_squares=False):
        assert U in self.source
        assert U.check_ind(ind)
        assert source_product is None or source_product.source == source_product.range == self.source
        assert range_product is None or range_product.source == range_product.range == self.range

        if source_product or range_product:
            return super(BlockDiagonalOperator, self).apply_inverse_adjoint(
                U, ind=ind, mu=mu, source_product=source_product, range_product=range_product)

        V = [None for i in range(self.num_source_blocks)]
        for i in range(self.num_source_blocks):
            V[i] = self._blocks[i, i].apply_inverse_adjoint(U.block(i), ind=ind, mu=mu, least_squares=least_squares)

        return BlockVectorArray(V)
