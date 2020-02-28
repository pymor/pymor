# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from numbers import Integral, Number
from scipy.sparse import bmat

from pymor.operators.constructions import IdentityOperator, ZeroOperator
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.block import BlockVectorSpace


class BlockOperatorBase(Operator):

    def _operators(self):
        """Iterator over operators."""
        for (i, j) in np.ndindex(self.blocks.shape):
            yield self.blocks[i, j]

    def __init__(self, blocks):
        self.blocks = blocks = np.array(blocks)
        assert 1 <= blocks.ndim <= 2
        if self.blocked_source and self.blocked_range:
            assert blocks.ndim == 2
        elif self.blocked_source:
            if blocks.ndim == 1:
                blocks.shape = (1, len(blocks))
        else:
            if blocks.ndim == 1:
                blocks.shape = (len(blocks), 1)
        assert all(isinstance(op, Operator) or op is None for op in self._operators())

        # check if every row/column contains at least one operator
        assert all(any(blocks[i, j] is not None for j in range(blocks.shape[1]))
                   for i in range(blocks.shape[0]))
        assert all(any(blocks[i, j] is not None for i in range(blocks.shape[0]))
                   for j in range(blocks.shape[1]))

        # find source/range spaces for every column/row
        source_spaces = [None for j in range(blocks.shape[1])]
        range_spaces = [None for i in range(blocks.shape[0])]
        for (i, j), op in np.ndenumerate(blocks):
            if op is not None:
                assert source_spaces[j] is None or op.source == source_spaces[j]
                source_spaces[j] = op.source
                assert range_spaces[i] is None or op.range == range_spaces[i]
                range_spaces[i] = op.range

        # turn Nones to ZeroOperators
        for (i, j) in np.ndindex(blocks.shape):
            if blocks[i, j] is None:
                self.blocks[i, j] = ZeroOperator(range_spaces[i], source_spaces[j])

        self.source = BlockVectorSpace(source_spaces) if self.blocked_source else source_spaces[0]
        self.range = BlockVectorSpace(range_spaces) if self.blocked_range else range_spaces[0]
        self.num_source_blocks = len(source_spaces)
        self.num_range_blocks = len(range_spaces)
        self.linear = all(op.linear for op in self._operators())
        self.build_parameter_type(*self._operators())

    @property
    def H(self):
        return self.adjoint_type(np.vectorize(lambda op: op.H)(self.blocks.T))

    def apply(self, U, mu=None):
        assert U in self.source

        V_blocks = [None for i in range(self.num_range_blocks)]
        for (i, j), op in np.ndenumerate(self.blocks):
            Vi = op.apply(U.block(j) if self.blocked_source else U, mu=mu)
            if V_blocks[i] is None:
                V_blocks[i] = Vi
            else:
                V_blocks[i] += Vi

        return self.range.make_array(V_blocks) if self.blocked_range else V_blocks[0]

    def apply_adjoint(self, V, mu=None):
        assert V in self.range

        U_blocks = [None for j in range(self.num_source_blocks)]
        for (i, j), op in np.ndenumerate(self.blocks):
            Uj = op.apply_adjoint(V.block(i) if self.blocked_range else V, mu=mu)
            if U_blocks[j] is None:
                U_blocks[j] = Uj
            else:
                U_blocks[j] += Uj

        return self.source.make_array(U_blocks) if self.blocked_source else U_blocks[0]

    def assemble(self, mu=None):
        blocks = np.empty(self.blocks.shape, dtype=object)
        for (i, j) in np.ndindex(self.blocks.shape):
            blocks[i, j] = self.blocks[i, j].assemble(mu)
        if np.all(blocks == self.blocks):
            return self
        else:
            return self.__class__(blocks)

    def as_range_array(self, mu=None):

        def process_row(row, space):
            R = space.empty()
            for op in row:
                R.append(op.as_range_array(mu))
            return R

        subspaces = self.range.subspaces if self.blocked_range else [self.range]
        blocks = [process_row(row, space) for row, space in zip(self.blocks, subspaces)]
        return self.range.make_array(blocks) if self.blocked_range else blocks[0]

    def as_source_array(self, mu=None):

        def process_col(col, space):
            R = space.empty()
            for op in col:
                R.append(op.as_source_array(mu))
            return R

        subspaces = self.source.subspaces if self.blocked_source else [self.source]
        blocks = [process_col(col, space) for col, space in zip(self.blocks.T, subspaces)]
        return self.source.make_array(blocks) if self.blocked_source else blocks[0]


class BlockOperator(BlockOperatorBase):
    """A matrix of arbitrary |Operators|.

    This operator can be :meth:`applied <pymor.operators.interface.Operator.apply>`
    to a compatible :class:`BlockVectorArrays <pymor.vectorarrays.block.BlockVectorArray>`.

    Parameters
    ----------
    blocks
        Two-dimensional array-like where each entry is an |Operator| or `None`.
    """

    blocked_source = True
    blocked_range = True


class BlockRowOperator(BlockOperatorBase):
    """A row vector of arbitrary |Operators|."""
    blocked_source = True
    blocked_range = False


class BlockColumnOperator(BlockOperatorBase):
    """A column vector of arbitrary |Operators|."""
    blocked_source = False
    blocked_range = True


BlockOperator.adjoint_type = BlockOperator
BlockRowOperator.adjoint_type = BlockColumnOperator
BlockColumnOperator.adjoint_type = BlockRowOperator


class BlockProjectionOperator(BlockRowOperator):

    def __init__(self, block_space, component):
        assert isinstance(block_space, BlockVectorSpace)
        assert 0 <= component < len(block_space.subspaces)
        blocks = [ZeroOperator(space, space) if i != component else IdentityOperator(space)
                  for i, space in enumerate(block_space.subspaces)]
        super().__init__(blocks)


class BlockEmbeddingOperator(BlockColumnOperator):

    def __init__(self, block_space, component):
        assert isinstance(block_space, BlockVectorSpace)
        assert 0 <= component < len(block_space.subspaces)
        blocks = [ZeroOperator(space, space) if i != component else IdentityOperator(space)
                  for i, space in enumerate(block_space.subspaces)]
        super().__init__(blocks)


class BlockDiagonalOperator(BlockOperator):
    """Block diagonal |Operator| of arbitrary |Operators|.

    This is a specialization of :class:`BlockOperator` for the
    block diagonal case.
    """

    def __init__(self, blocks):
        blocks = np.array(blocks)
        assert 1 <= blocks.ndim <= 2
        if blocks.ndim == 2:
            blocks = np.diag(blocks)
        n = len(blocks)
        blocks2 = np.empty((n, n), dtype=object)
        for i, op in enumerate(blocks):
            blocks2[i, i] = op
        super().__init__(blocks2)

    def apply(self, U, mu=None):
        assert U in self.source
        V_blocks = [self.blocks[i, i].apply(U.block(i), mu=mu) for i in range(self.num_range_blocks)]
        return self.range.make_array(V_blocks)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        U_blocks = [self.blocks[i, i].apply_adjoint(V.block(i), mu=mu) for i in range(self.num_source_blocks)]
        return self.source.make_array(U_blocks)

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range
        U_blocks = [self.blocks[i, i].apply_inverse(V.block(i), mu=mu, least_squares=least_squares)
                    for i in range(self.num_source_blocks)]
        return self.source.make_array(U_blocks)

    def apply_inverse_adjoint(self, U, mu=None, least_squares=False):
        assert U in self.source
        V_blocks = [self.blocks[i, i].apply_inverse_adjoint(U.block(i), mu=mu, least_squares=least_squares)
                    for i in range(self.num_source_blocks)]
        return self.range.make_array(V_blocks)

    def assemble(self, mu=None):
        blocks = np.empty((self.num_source_blocks,), dtype=object)
        assembled = True
        for i in range(self.num_source_blocks):
            block_i = self.blocks[i, i].assemble(mu)
            assembled = assembled and block_i == self.blocks[i, i]
            blocks[i] = block_i
        if assembled:
            return self
        else:
            return self.__class__(blocks)


class SparseBlockOperator(Operator):
    """A row-major sparse matrix of arbitrary |Operators|.

    This operator can be :meth:`applied <pymor.operators.interface.Operator.apply>`
    to a compatible :class:`BlockVectorArrays <pymor.vectorarrays.block.BlockVectorArray>`.

    Parameters
    ----------
    blocks
        List or tuple of dicts as sparse representations of rows, i.e. `blocks[i] = {j: op_ij, ...}`.
    """

    def _operators(self):
        """Iterator over operators."""
        for row in self.blocks:
            for _, op in row.items():
                yield op

    def __init__(self, blocks):
        assert isinstance(blocks, (list, tuple)), 'blocks has to be a list of dicts (each a sparse row)!'
        self.num_source_blocks = rows = len(blocks)
        assert all(isinstance(row, dict) for row in blocks), 'blocks has to be a list of dicts (each a sparse row)!'
        assert all(all(isinstance(col_ind, Number) for col_ind in row.keys()) for row in blocks)
        # check if every row/column contains at least one operator
        col_indcs = set()
        for row in blocks:
            col_indcs = col_indcs.union(set(row.keys()))
        self.num_range_blocks = cols = len(col_indcs)
        assert all(isinstance(col_ind, Integral) for col_ind in col_indcs)
        assert all(col_ind >= 0 for col_ind in col_indcs), f'Column indices have to be positive (are {col_indcs})!'
        assert all(col_ind < cols for col_ind in col_indcs)
        assert all(all(isinstance(op, Operator) for _, op in row.items()) for row in blocks)
        # find source/range spaces for every column/row
        source_spaces = [None for _ in range(cols)]
        range_spaces = [None for _ in range(rows)]
        for row_ind, row in enumerate(blocks):
            for col_ind, op in row.items():
                if not source_spaces[col_ind]:
                    source_spaces[col_ind] = op.source
                assert op.source == source_spaces[col_ind], 'All operators in a column need to have the same source!'
                if not range_spaces[row_ind]:
                    range_spaces[row_ind] = op.range
                assert op.range == range_spaces[row_ind], 'All operators in a row need to have the same range!'
        self.blocks = blocks
        self.source = BlockVectorSpace(source_spaces)
        self.range = BlockVectorSpace(range_spaces)
        self.linear = all(op.linear for op in self._operators())
        self.build_parameter_type(*self._operators())

    @property
    def H(self):
        rows = self.num_source_blocks
        cols = self.num_range_blocks
        adj_blocks = [{} for _ in range(cols)]
        for row_ind, row in enumerate(self.blocks):
            for col_ind, op in row.items():
                adj_blocks[col_ind][row_ind] = op.H
        return SparseBlockOperator(adj_blocks)

    def apply(self, U, mu=None):
        assert U in self.source

        V_blocks = [None for i in range(self.num_range_blocks)]
        for i, row in enumerate(self.blocks):
            for j, op in row.items():
                Vi = op.apply(U.block(j), mu=mu)
                if V_blocks[i] is None:
                    V_blocks[i] = Vi
                else:
                    V_blocks[i] += Vi

        return self.range.make_array(V_blocks)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range

        V_blocks = [None for j in range(self.num_source_blocks)]
        for i, row in enumerate(self.blocks):
            for j, op in row.items():
                Uj = op.apply_adjoint(V.block(i), mu=mu)
                if V_blocks[j] is None:
                    V_blocks[j] = Uj
                else:
                    V_blocks[j] += Uj

        return self.source.make_array(V_blocks)

    def assemble(self, mu=None):
        blocks = [{} for _ in range(self.num_range_blocks)]
        differs = 0
        for i, row in enumerate(self.blocks):
            for j, op in row.items():
                blocks[i][j] = op.assemble(mu=mu)
                if blocks[i][j] != op:
                    differs += 1
        if differs > 0:
            return self.__class__(blocks)
        else:
            return self

    def as_range_array(self, mu=None):
        blocks = [None for _ in range(self.num_range_blocks)]
        for i, row in enumerate(self.blocks):
            blocks[i] = self.range.subspaces[i].empty()
            for j in range(self.num_source_blocks):
                if j in row:
                    op = row[j]
                else:
                    op = ZeroOperator(self.range.subspaces[i], self.source.subspaces[j])
                blocks[i].append(op.as_range_array(mu))
        return self.range.make_array(blocks)

    def as_source_array(self, mu=None):
        blocks = [subspace.empty() for subspace in self.source.subspaces]
        for i, row in enumerate(self.blocks):
            for j in range(self.num_source_blocks):
                if j in row:
                    op = row[j]
                else:
                    op = ZeroOperator(self.range.subspaces[i], self.source.subspaces[j])
                blocks[j].append(op.as_source_array(mu))
        return self.source.make_array(blocks)


class SecondOrderModelOperator(BlockOperator):
    r"""BlockOperator appearing in SecondOrderModel.to_lti().

    This represents a block operator

    .. math::
        \mathcal{A} =
        \begin{bmatrix}
            0 & I \\
            -K & -E
        \end{bmatrix},

    which satisfies

    .. math::
        \mathcal{A}^H
        &=
        \begin{bmatrix}
            0 & -K^H \\
            I & -E^H
        \end{bmatrix}, \\
        \mathcal{A}^{-1}
        &=
        \begin{bmatrix}
            -K^{-1} E & -K^{-1} \\
            I & 0
        \end{bmatrix}, \\
        \mathcal{A}^{-H}
        &=
        \begin{bmatrix}
            -E^H K^{-H} & I \\
            -K^{-H} & 0
        \end{bmatrix}.

    Parameters
    ----------
    E
        |Operator|.
    K
        |Operator|.
    """

    def __init__(self, E, K):
        super().__init__([[None, IdentityOperator(E.source)],
                          [K * (-1), E * (-1)]])
        self.__auto_init(locals())

    def apply(self, U, mu=None):
        assert U in self.source
        V_blocks = [U.block(1),
                    -self.K.apply(U.block(0), mu=mu) - self.E.apply(U.block(1), mu=mu)]
        return self.range.make_array(V_blocks)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        U_blocks = [-self.K.apply_adjoint(V.block(1), mu=mu),
                    V.block(0) - self.E.apply_adjoint(V.block(1), mu=mu)]
        return self.source.make_array(U_blocks)

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range
        U_blocks = [-self.K.apply_inverse(self.E.apply(V.block(0), mu=mu) + V.block(1), mu=mu,
                                          least_squares=least_squares),
                    V.block(0)]
        return self.source.make_array(U_blocks)

    def apply_inverse_adjoint(self, U, mu=None, least_squares=False):
        assert U in self.source
        KitU0 = self.K.apply_inverse_adjoint(U.block(0), mu=mu, least_squares=least_squares)
        V_blocks = [-self.E.apply_adjoint(KitU0, mu=mu) + U.block(1),
                    -KitU0]
        return self.range.make_array(V_blocks)

    def assemble(self, mu=None):
        E = self.E.assemble(mu)
        K = self.K.assemble(mu)
        if E == self.E and K == self.K:
            return self
        else:
            return self.__class__(E, K)


class ShiftedSecondOrderModelOperator(BlockOperator):
    r"""BlockOperator appearing in second-order systems.

    This represents a block operator

    .. math::
        a \mathcal{E} + b \mathcal{A} =
        \begin{bmatrix}
            a I & b I \\
            -b K & a M - b E
        \end{bmatrix},

    which satisfies

    .. math::
        (a \mathcal{E} + b \mathcal{A})^H
        &=
        \begin{bmatrix}
            \overline{a} I & -\overline{b} K^H \\
            \overline{b} I & \overline{a} M^H - \overline{b} E^H
        \end{bmatrix}, \\
        (a \mathcal{E} + b \mathcal{A})^{-1}
        &=
        \begin{bmatrix}
            (a^2 M - a b E + b^2 K)^{-1} (a M - b E)
            & -b (a^2 M - a b E + b^2 K)^{-1} \\
            b (a^2 M - a b E + b^2 K)^{-1} K
            & a (a^2 M - a b E + b^2 K)^{-1}
        \end{bmatrix}, \\
        (a \mathcal{E} + b \mathcal{A})^{-H}
        &=
        \begin{bmatrix}
            (a M - b E)^H (a^2 M - a b E + b^2 K)^{-H}
            & \overline{b} K^H (a^2 M - a b E + b^2 K)^{-H} \\
            -\overline{b} (a^2 M - a b E + b^2 K)^{-H}
            & \overline{a} (a^2 M - a b E + b^2 K)^{-H}
        \end{bmatrix}.

    Parameters
    ----------
    M
        |Operator|.
    E
        |Operator|.
    K
        |Operator|.
    a, b
        Complex numbers.
    """

    def __init__(self, M, E, K, a, b):
        super().__init__([[IdentityOperator(M.source) * a, IdentityOperator(M.source) * b],
                          [((-b) * K).assemble(), (a * M - b * E).assemble()]])
        self.__auto_init(locals())

    def apply(self, U, mu=None):
        assert U in self.source
        V_blocks = [U.block(0) * self.a
                    + U.block(1) * self.b,
                    self.K.apply(U.block(0), mu=mu) * (-self.b)
                    + self.M.apply(U.block(1), mu=mu) * self.a
                    - self.E.apply(U.block(1), mu=mu) * self.b]
        return self.range.make_array(V_blocks)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        U_blocks = [V.block(0) * self.a.conjugate()
                    - self.K.apply_adjoint(V.block(1), mu=mu) * self.b.conjugate(),
                    V.block(0) * self.b.conjugate()
                    + self.M.apply_adjoint(V.block(1), mu=mu) * self.a.conjugate()
                    - self.E.apply_adjoint(V.block(1), mu=mu) * self.b.conjugate()]
        return self.source.make_array(U_blocks)

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range
        aMmbEV0 = self.M.apply(V.block(0), mu=mu) * self.a - self.E.apply(V.block(0), mu=mu) * self.b
        KV0 = self.K.apply(V.block(0), mu=mu)
        a2MmabEpb2K = (self.a**2 * self.M - self.a * self.b * self.E + self.b**2 * self.K).assemble(mu=mu)
        a2MmabEpb2KiV1 = a2MmabEpb2K.apply_inverse(V.block(1), mu=mu, least_squares=least_squares)
        U_blocks = [a2MmabEpb2K.apply_inverse(aMmbEV0, mu=mu, least_squares=least_squares)
                    - a2MmabEpb2KiV1 * self.b,
                    a2MmabEpb2K.apply_inverse(KV0, mu=mu, least_squares=least_squares) * self.b
                    + a2MmabEpb2KiV1 * self.a]
        return self.source.make_array(U_blocks)

    def apply_inverse_adjoint(self, U, mu=None, least_squares=False):
        assert U in self.source
        a2MmabEpb2K = (self.a**2 * self.M - self.a * self.b * self.E + self.b**2 * self.K).assemble(mu=mu)
        a2MmabEpb2KitU0 = a2MmabEpb2K.apply_inverse_adjoint(U.block(0), mu=mu, least_squares=least_squares)
        a2MmabEpb2KitU1 = a2MmabEpb2K.apply_inverse_adjoint(U.block(1), mu=mu, least_squares=least_squares)
        V_blocks = [self.M.apply_adjoint(a2MmabEpb2KitU0, mu=mu) * self.a.conjugate()
                    - self.E.apply_adjoint(a2MmabEpb2KitU0, mu=mu) * self.b.conjugate()
                    + self.K.apply_adjoint(a2MmabEpb2KitU1, mu=mu) * self.b.conjugate(),
                    -a2MmabEpb2KitU0 * self.b.conjugate()
                    + a2MmabEpb2KitU1 * self.a.conjugate()]
        return self.range.make_array(V_blocks)

    def assemble(self, mu=None):
        M = self.M.assemble(mu)
        E = self.E.assemble(mu)
        K = self.K.assemble(mu)
        if M == self.M and E == self.E and K == self.K:
            return self
        else:
            return self.__class__(M, E, K, self.a, self.b)
