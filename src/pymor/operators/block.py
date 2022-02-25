# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.operators.constructions import IdentityOperator, ZeroOperator
from pymor.operators.interface import Operator
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

    @property
    def H(self):
        return self.adjoint_type(np.vectorize(lambda op: op.H)(self.blocks.T))

    def apply(self, U, mu=None):
        assert U in self.source

        V_blocks = [None for i in range(self.num_range_blocks)]
        for (i, j), op in np.ndenumerate(self.blocks):
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
        for (i, j), op in np.ndenumerate(self.blocks):
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

    def d_mu(self, parameter, index=0):
        blocks = np.empty(self.blocks.shape, dtype=object)
        for (i, j) in np.ndindex(self.blocks.shape):
            blocks[i, j] = self.blocks[i, j].d_mu(parameter, index)
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
        V_blocks = [self.blocks[i, i].apply(U.blocks[i], mu=mu) for i in range(self.num_range_blocks)]
        return self.range.make_array(V_blocks)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        U_blocks = [self.blocks[i, i].apply_adjoint(V.blocks[i], mu=mu) for i in range(self.num_source_blocks)]
        return self.source.make_array(U_blocks)

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
