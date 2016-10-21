# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import ZeroOperator
from pymor.operators.interfaces import OperatorInterface
from pymor.vectorarrays.block import BlockVectorArray
from pymor.vectorarrays.interfaces import VectorSpace


class BlockOperator(OperatorBase):
    """A matrix of arbitrary |Operators|.

    This operator can be :meth:`applied <pymor.operators.interfaces.OperatorInterface.apply>`
    to :class:`BlockVectorArrays <pymor.vectorarrays.block.BlockVectorArray>` of an
    appropriate :attr:`~pymor.vectorarrays.interfaces.VectorArrayInterface.subtype`.

    Parameters
    ----------
    blocks
        Two-dimensional array-like where each entry is an |Operator| or `None`.
    """

    kind = 'operator'

    def _operators(self):
        """Iterator over operators."""
        for (i, j) in np.ndindex(self._blocks.shape):
            yield self._blocks[i, j]

    def __init__(self, blocks):
        blocks = np.array(blocks)
        assert isinstance(blocks, np.ndarray) and blocks.ndim == 2
        self._blocks = blocks
        assert all(isinstance(op, OperatorInterface) or op is None for op in self._operators())

        # check if every row/column contains at least one operator
        assert all(any(blocks[i, j] is not None for j in range(blocks.shape[1]))
                   for i in range(blocks.shape[0]))
        assert all(any(blocks[i, j] is not None for i in range(blocks.shape[0]))
                   for j in range(blocks.shape[1]))

        # find source/range types for every column/row
        source_types = [None for j in range(blocks.shape[1])]
        range_types = [None for i in range(blocks.shape[0])]
        for (i, j), op in np.ndenumerate(blocks):
            if op is not None:
                assert source_types[j] is None or op.source == source_types[j]
                source_types[j] = op.source
                assert range_types[i] is None or op.range == range_types[i]
                range_types[i] = op.range

        # turn Nones to ZeroOperators
        for (i, j) in np.ndindex(blocks.shape):
            if blocks[i, j] is None:
                self._blocks[i, j] = ZeroOperator(source_types[j], range_types[i])

        self.source = VectorSpace(BlockVectorArray, tuple(source_types))
        self.range = VectorSpace(BlockVectorArray, tuple(range_types))
        self._source_dims = tuple(space.dim for space in self.source.subtype)
        self._range_dims = tuple(space.dim for space in self.range.subtype)
        self.num_source_blocks = len(source_types)
        self.num_range_blocks = len(range_types)
        self.linear = all(op.linear for op in self._operators())
        self.build_parameter_type(*self._operators())

    @classmethod
    def hstack(cls, operators):
        """Horizontal stacking of |Operators|.

        Parameters
        ----------
        operators
            An iterable of |Operators| or `None`s.
        """
        blocks = np.array([[op for op in operators]])
        return cls(blocks)

    @classmethod
    def vstack(cls, operators):
        """Vertical stacking of |Operators|.

        Parameters
        ----------
        operators
            An iterable of |Operators| or `None`s.
        """
        blocks = np.array([[op] for op in operators])
        return cls(blocks)

    def apply(self, U, mu=None):
        assert U in self.source

        V_blocks = [None for i in range(self.num_range_blocks)]
        for (i, j), op in np.ndenumerate(self._blocks):
            Vi = op.apply(U.block(j), mu=mu)
            if V_blocks[i] is None:
                V_blocks[i] = Vi
            else:
                V_blocks[i] += Vi

        return BlockVectorArray(V_blocks)

    def apply_adjoint(self, U, mu=None, source_product=None, range_product=None):
        assert U in self.range
        assert source_product is None or source_product.source == source_product.range == self.source
        assert range_product is None or range_product.source == range_product.range == self.range

        if range_product is not None:
            U = range_product.apply(U)

        V_blocks = [None for j in range(self.num_source_blocks)]
        for (i, j), op in np.ndenumerate(self._blocks):
            Vj = op.apply_adjoint(U.block(i), mu=mu)
            if V_blocks[j] is None:
                V_blocks[j] = Vj
            else:
                V_blocks[j] += Vj

        V = BlockVectorArray(V_blocks)
        if source_product is not None:
            V = source_product.apply_inverse(V)

        return V

    def assemble(self, mu=None):
        blocks = np.empty(self._blocks.shape, dtype=object)
        for (i, j) in np.ndindex(self._blocks.shape):
            blocks[i, j] = self._blocks[i, j].assemble(mu)
        if np.all(blocks == self._blocks):
            return self
        else:
            return self.__class__(blocks)

    def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
        assert operators[0] is self
        blocks = np.empty(self._blocks.shape, dtype=object)
        if len(operators) > 1:
            for (i, j) in np.ndindex(self._blocks.shape):
                operators_ij = [op._blocks[i, j] for op in operators]
                blocks[i, j] = operators_ij[0].assemble_lincomb(operators_ij, coefficients,
                                                                solver_options=solver_options, name=name)
                if blocks[i, j] is None:
                    return None
            return self.__class__(blocks)
        else:
            if coefficients[0] == 1:
                return self
            c = coefficients[0].real if coefficients[0].imag == 0 else coefficients[0]
            for (i, j) in np.ndindex(self._blocks.shape):
                blocks[i, j] = self._blocks[i, j] * c
            return self.__class__(blocks)


class BlockDiagonalOperator(BlockOperator):
    """Block diagonal |Operator| of arbitrary |Operators|.

    This is a specialization of :class:`BlockOperator` for the
    block diagonal case.
    """

    def __init__(self, blocks):
        n = len(blocks)
        blocks2 = np.empty((n, n), dtype=object)
        for i, op in enumerate(blocks):
            blocks2[i, i] = op
        super().__init__(blocks2)

    def apply(self, U, mu=None):
        assert U in self.source

        V_blocks = [self._blocks[i, i].apply(U.block(i), mu=mu) for i in range(self.num_range_blocks)]

        return BlockVectorArray(V_blocks)

    def apply_adjoint(self, U, mu=None, source_product=None, range_product=None):
        assert U in self.range
        assert source_product is None or source_product.source == source_product.range == self.source
        assert range_product is None or range_product.source == range_product.range == self.range

        if range_product is not None:
            U = range_product.apply(U)

        V_blocks = [self._blocks[i, i].apply_adjoint(U.block(i), mu=mu) for i in range(self.num_source_blocks)]

        V = BlockVectorArray(V_blocks)
        if source_product is not None:
            V = source_product.apply_inverse(V)

        return V

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range

        U_blocks = [self._blocks[i, i].apply_inverse(V.block(i), mu=mu, least_squares=least_squares)
                    for i in range(self.num_source_blocks)]

        return BlockVectorArray(U_blocks)

    def apply_inverse_adjoint(self, U, mu=None, source_product=None, range_product=None, least_squares=False):
        assert U in self.source
        assert source_product is None or source_product.source == source_product.range == self.source
        assert range_product is None or range_product.source == range_product.range == self.range

        if source_product or range_product:
            return super().apply_inverse_adjoint(
                U, mu=mu, source_product=source_product, range_product=range_product)

        V_blocks = [self._blocks[i, i].apply_inverse_adjoint(U.block(i), mu=mu, least_squares=least_squares)
                    for i in range(self.num_source_blocks)]

        return BlockVectorArray(V_blocks)

    def assemble(self, mu=None):
        blocks = np.empty((self.num_source_blocks,), dtype=object)
        assembled = True
        for i in range(self.num_source_blocks):
            block_i = self._blocks[i, i].assemble(mu)
            assembled = assembled and block_i == self._blocks[i, i]
            blocks[i] = block_i
        if assembled:
            return self
        else:
            return self.__class__(blocks)

    def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
        if not all(isinstance(op, self.__class__) for op in operators):
            return super().assemble_lincomb(operators, coefficients, solver_options=solver_options, name=name)

        assert operators[0] is self
        blocks = np.empty((self.num_source_blocks,), dtype=object)
        if len(operators) > 1:
            for i in range(self.num_source_blocks):
                operators_i = [op._blocks[i, i] for op in operators]
                blocks[i] = operators_i[0].assemble_lincomb(operators_i, coefficients,
                                                            solver_options=solver_options, name=name)
                if blocks[i] is None:
                    return None
            return self.__class__(blocks)
        else:
            if coefficients[0] == 1:
                return self
            c = coefficients[0].real if coefficients[0].imag == 0 else coefficients[0]
            for i in range(self.num_source_blocks):
                blocks[i] = self._blocks[i, i] * c
            return self.__class__(blocks)
