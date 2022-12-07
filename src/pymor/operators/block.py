# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import sparse

from pymor.operators.constructions import IdentityOperator, ZeroOperator
from pymor.operators.interface import Operator
from pymor.vectorarrays.block import BlockVectorSpace


class BlockOperatorBase(Operator):

    def __init__(self, blocks):
        # all None entries need to be zeros because Nones will not be cleared.
        if isinstance(blocks, np.ndarray) or isinstance(blocks, list):
            blocks = np.array(blocks)
            blocks[blocks == None] = 0  # noqa: E711
        self.blocks = blocks = sparse.COO(blocks)
        assert 1 <= blocks.ndim <= 2
        if self.blocked_source and self.blocked_range:
            assert blocks.ndim == 2
        elif self.blocked_source:
            if blocks.ndim == 1:
                blocks.shape = (1, len(blocks))
        else:
            if blocks.ndim == 1:
                blocks.shape = (len(blocks), 1)
        assert all(isinstance(op, Operator) for op in self.blocks.data)

        # check if every row/column contains at least one operator
        assert all(any(blocks[i, j] for j in range(blocks.shape[1]))
                   for i in range(blocks.shape[0]))
        assert all(any(blocks[i, j] for i in range(blocks.shape[0]))
                   for j in range(blocks.shape[1]))

        # find source/range spaces for every column/row
        source_spaces = [None for j in range(blocks.shape[1])]
        range_spaces = [None for i in range(blocks.shape[0])]
        for (i, j) in zip(self.blocks.coords[0], self.blocks.coords[1]):
            op = self.blocks[i, j]
            assert source_spaces[j] is None or op.source == source_spaces[j]
            source_spaces[j] = op.source
            assert range_spaces[i] is None or op.range == range_spaces[i]
            range_spaces[i] = op.range

        self.source = BlockVectorSpace(source_spaces) if self.blocked_source else source_spaces[0]
        self.range = BlockVectorSpace(range_spaces) if self.blocked_range else range_spaces[0]
        self.num_source_blocks = len(source_spaces)
        self.num_range_blocks = len(range_spaces)
        self.linear = all(op.linear for op in self.blocks.data)

    @property
    def H(self):
        blocks_trans = self.blocks.T
        adjoint_data = np.vectorize(lambda op: op.H)(blocks_trans.data)
        return self.adjoint_type((adjoint_data, blocks_trans.coords))

    def apply(self, U, mu=None):
        assert U in self.source

        V_blocks = [None for i in range(self.num_range_blocks)]
        for (i, j) in zip(self.blocks.coords[0], self.blocks.coords[1]):
            op = self.blocks[i, j]
            if isinstance(op, ZeroOperator):
                Vi = op.range.zeros(len(U))
            else:
                Vi = op.apply(U.blocks[j] if self.blocked_source else U, mu=mu)
            if V_blocks[i] is None:
                V_blocks[i] = Vi
            else:
                V_blocks[i] += Vi

        return self.range.make_array(V_blocks) if self.blocked_range else V_blocks[0]

    def apply_adjoint(self, V, mu=None):
        assert V in self.range

        U_blocks = [None for j in range(self.num_source_blocks)]
        for (i, j) in zip(self.blocks.coords[0], self.blocks.coords[1]):
            op = self.blocks[i, j]
            if isinstance(op, ZeroOperator):
                Uj = op.source.zeros(len(V))
            else:
                Uj = op.apply_adjoint(V.blocks[i] if self.blocked_range else V, mu=mu)
            if U_blocks[j] is None:
                U_blocks[j] = Uj
            else:
                U_blocks[j] += Uj

        return self.source.make_array(U_blocks) if self.blocked_source else U_blocks[0]

    def assemble(self, mu=None):
        blocks = np.zeros(self.blocks.data.shape, dtype=object)
        for i, block in enumerate(self.blocks.data):
            blocks[i] = block.assemble(mu)
        if all(blocks[i] == self.blocks.data[i] for i in range(len(blocks))):
            return self
        else:
            return self.__class__((blocks, self.blocks.coords))

    def as_range_array(self, mu=None):

        def process_row(row, space, source_spaces):
            R = space.empty()
            for op, source_space in zip(row, source_spaces):
                if op:
                    R.append(op.as_range_array(mu))
                else:
                    # mimic what would have been done by a ZeroOperator
                    R.append(space.zeros(source_space.dim))
            return R

        range_subspaces = self.range.subspaces if self.blocked_range else [self.range]
        source_subspaces = self.source.subspaces if self.blocked_source else [self.source]
        blocks = [process_row(row, space, source_subspaces) for row, space in zip(self.blocks, range_subspaces)]
        return self.range.make_array(blocks) if self.blocked_range else blocks[0]

    def as_source_array(self, mu=None):

        def process_col(col, space, range_spaces):
            R = space.empty()
            for op, range_space in zip(col, range_spaces):
                if op:
                    R.append(op.as_source_array(mu))
                else:
                    # mimic what would have been done by a ZeroOperator
                    R.append(space.zeros(range_space.dim))
            return R

        range_subspaces = self.range.subspaces if self.blocked_range else [self.range]
        source_subspaces = self.source.subspaces if self.blocked_source else [self.source]
        blocks = [process_col(col, space, range_subspaces) for col, space in zip(self.blocks.T, source_subspaces)]
        return self.source.make_array(blocks) if self.blocked_source else blocks[0]

    def d_mu(self, parameter, index=0):
        blocks = np.zeros(self.blocks.shape, dtype=object)
        for (i, j) in zip(self.blocks.coords[0], self.blocks.coords[1]):
            blocks[i, j] = self.blocks[i, j].d_mu(parameter, index)
        return self.with_(blocks=blocks)

    def to_dense(self):
        # turn Nones to ZeroOperators
        blocks = np.zeros(self.blocks.shape, dtype=object)
        for (i, j) in np.ndindex(blocks.shape):
            if not self.blocks[i, j]:
                blocks[i, j] = ZeroOperator(self.range.subspaces[i], self.source.subspaces[j])
            else:
                blocks[i, j] = self.blocks[i, j]
        return self.with_(blocks=blocks)

    def to_sparse(self):
        # remove all ZeroOperators
        blocks = np.zeros(self.blocks.shape, dtype=object)
        for (i, j) in np.ndindex(blocks.shape):
            if isinstance(self.blocks[i, j], ZeroOperator):
                blocks[i, j] = 0
            else:
                blocks[i, j] = self.blocks[i, j]
        return self.with_(blocks=blocks)


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
        blocks = [0 if i != component else IdentityOperator(space)
                  for i, space in enumerate(block_space.subspaces)]
        super().__init__(blocks)


class BlockEmbeddingOperator(BlockColumnOperator):

    def __init__(self, block_space, component):
        assert isinstance(block_space, BlockVectorSpace)
        assert 0 <= component < len(block_space.subspaces)
        blocks = [0 if i != component else IdentityOperator(space)
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
        coords = np.arange(len(blocks))
        super().__init__((blocks, (coords, coords)))

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        assert V in self.range
        assert initial_guess is None or initial_guess in self.source and len(initial_guess) == len(V)
        U_blocks = [self.blocks[i, i].apply_inverse(V.blocks[i], mu=mu,
                                                    initial_guess=(initial_guess.blocks[i]
                                                                   if initial_guess is not None else None),
                                                    least_squares=least_squares)
                    for i in range(self.num_source_blocks)]
        return self.source.make_array(U_blocks)

    def apply_inverse_adjoint(self, U, mu=None, initial_guess=None, least_squares=False):
        assert U in self.source
        assert initial_guess is None or initial_guess in self.range and len(initial_guess) == len(U)
        V_blocks = [self.blocks[i, i].apply_inverse_adjoint(U.blocks[i], mu=mu,
                                                            initial_guess=(initial_guess.blocks[i]
                                                                           if initial_guess is not None else None),
                                                            least_squares=least_squares)
                    for i in range(self.num_source_blocks)]
        return self.range.make_array(V_blocks)

    def assemble(self, mu=None):
        blocks = np.zeros(self.blocks.data.shape, dtype=object)
        for i, block in enumerate(self.blocks.data):
            blocks[i] = block.assemble(mu)
        if all(blocks[i] == self.blocks.data[i] for i in range(len(blocks))):
            return self
        else:
            return self.__class__(blocks)


class SecondOrderModelOperator(BlockOperator):
    r"""BlockOperator appearing in SecondOrderModel.to_lti().

    This represents a block operator

    .. math::
        \mathcal{A} =
        \begin{bmatrix}
            \alpha I & \beta I \\
            B & A
        \end{bmatrix},

    which satisfies

    .. math::
        \mathcal{A}^H
        &=
        \begin{bmatrix}
            \overline{\alpha} I & B^H \\
            \overline{\beta} I & A^H
        \end{bmatrix}, \\
        \mathcal{A}^{-1}
        &=
        \begin{bmatrix}
            (\alpha A - \beta B)^{-1} A & -\beta (\alpha A - \beta B)^{-1} \\
            -(\alpha A - \beta B)^{-1} B & \alpha (\alpha A - \beta B)^{-1}
        \end{bmatrix}, \\
        \mathcal{A}^{-H}
        &=
        \begin{bmatrix}
            A^H (\alpha A - \beta B)^{-H}
            & -B^H (\alpha A - \beta B)^{-H} \\
            -\overline{\beta} (\alpha A - \beta B)^{-H}
            & \overline{\alpha} (\alpha A - \beta B)^{-H}
        \end{bmatrix}.

    Parameters
    ----------
    alpha
        Scalar.
    beta
        Scalar.
    A
        |Operator|.
    B
        |Operator|.
    """

    def __init__(self, alpha, beta, A, B):
        eye = IdentityOperator(A.source)
        super().__init__([[alpha * eye, beta * eye],
                          [B, A]])
        self.__auto_init(locals())

    def apply(self, U, mu=None):
        assert U in self.source
        V_blocks = [self.alpha * U.blocks[0] + self.beta * U.blocks[1],
                    self.B.apply(U.blocks[0], mu=mu) + self.A.apply(U.blocks[1], mu=mu)]
        return self.range.make_array(V_blocks)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        U_blocks = [self.alpha.conjugate() * V.blocks[0] + self.B.apply_adjoint(V.blocks[1], mu=mu),
                    self.beta.conjugate() * V.blocks[0] + self.A.apply_adjoint(V.blocks[1], mu=mu)]
        return self.source.make_array(U_blocks)

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        assert V in self.range
        assert initial_guess is None or initial_guess in self.source and len(initial_guess) == len(V)
        aAmbB = (self.alpha * self.A - self.beta * self.B).assemble(mu=mu)
        aAmbB_V1 = aAmbB.apply_inverse(V.blocks[1], least_squares=least_squares)
        aAmbB_A_V0 = aAmbB.apply_inverse(self.A.apply(V.blocks[0], mu=mu), least_squares=least_squares)
        aAmbB_B_V0 = aAmbB.apply_inverse(self.B.apply(V.blocks[0], mu=mu), least_squares=least_squares)
        U_blocks = [aAmbB_A_V0 - self.beta * aAmbB_V1,
                    self.alpha * aAmbB_V1 - aAmbB_B_V0]
        return self.source.make_array(U_blocks)

    def apply_inverse_adjoint(self, U, mu=None, initial_guess=None, least_squares=False):
        assert U in self.source
        assert initial_guess is None or initial_guess in self.range and len(initial_guess) == len(U)
        aAmbB = (self.alpha * self.A - self.beta * self.B).assemble(mu=mu)
        aAmbB_U0 = aAmbB.apply_inverse_adjoint(U.blocks[0], least_squares=least_squares)
        aAmbB_U1 = aAmbB.apply_inverse_adjoint(U.blocks[1], least_squares=least_squares)
        V_blocks = [self.A.apply_adjoint(aAmbB_U0, mu=mu) - self.B.apply_adjoint(aAmbB_U1, mu=mu),
                    self.alpha.conjugate() * aAmbB_U1 - self.beta.conjugate() * aAmbB_U0]
        return self.range.make_array(V_blocks)

    def assemble(self, mu=None):
        A = self.A.assemble(mu)
        B = self.B.assemble(mu)
        if A == self.A and B == self.B:
            return self
        else:
            return self.__class__(self.alpha, self.beta, A, B)
