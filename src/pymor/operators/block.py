# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import IdentityOperator, ZeroOperator
from pymor.operators.interfaces import OperatorInterface
from pymor.vectorarrays.block import BlockVectorSpace


class BlockOperatorBase(OperatorBase):

    def _operators(self):
        """Iterator over operators."""
        for (i, j) in np.ndindex(self._blocks.shape):
            yield self._blocks[i, j]

    def __init__(self, blocks, source_id='STATE', range_id='STATE'):
        blocks = np.array(blocks)
        assert 1 <= blocks.ndim <= 2
        if self.blocked_source and self.blocked_range:
            assert blocks.ndim == 2
        elif self.blocked_source:
            if blocks.ndim == 1:
                blocks.shape = (1, len(blocks))
        else:
            if blocks.ndim == 1:
                blocks.shape = (len(blocks), 1)
        self._blocks = blocks
        assert all(isinstance(op, OperatorInterface) or op is None for op in self._operators())

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
                self._blocks[i, j] = ZeroOperator(range_spaces[i], source_spaces[j])

        self.source = BlockVectorSpace(source_spaces, id_=source_id) if self.blocked_source else source_spaces[0]
        self.range = BlockVectorSpace(range_spaces, id_=range_id) if self.blocked_range else range_spaces[0]
        self.num_source_blocks = len(source_spaces)
        self.num_range_blocks = len(range_spaces)
        self.linear = all(op.linear for op in self._operators())
        self.build_parameter_type(*self._operators())

    @property
    def H(self):
        return self.adjoint_type(np.vectorize(lambda op: op.H if op else None)(self._blocks.T))

    def apply(self, U, mu=None):
        assert U in self.source

        V_blocks = [None for i in range(self.num_range_blocks)]
        for (i, j), op in np.ndenumerate(self._blocks):
            Vi = op.apply(U.block(j) if self.blocked_source else U, mu=mu)
            if V_blocks[i] is None:
                V_blocks[i] = Vi
            else:
                V_blocks[i] += Vi

        return self.range.make_array(V_blocks) if self.blocked_range else V_blocks[0]

    def apply_adjoint(self, V, mu=None):
        assert V in self.range

        U_blocks = [None for j in range(self.num_source_blocks)]
        for (i, j), op in np.ndenumerate(self._blocks):
            Uj = op.apply_adjoint(V.block(i) if self.blocked_range else V, mu=mu)
            if U_blocks[j] is None:
                U_blocks[j] = Uj
            else:
                U_blocks[j] += Uj

        return self.source.make_array(U_blocks) if self.blocked_source else U_blocks[0]

    def assemble(self, mu=None):
        blocks = np.empty(self._blocks.shape, dtype=object)
        for (i, j) in np.ndindex(self._blocks.shape):
            blocks[i, j] = self._blocks[i, j].assemble(mu)
        if np.all(blocks == self._blocks):
            return self
        else:
            return self.__class__(blocks)

    def _assemble_lincomb_preprocess_operators(self, operators):
        return [
            BlockDiagonalOperator([IdentityOperator(s) for s in op.source.subspaces],
                                  source_id=op.source.id, range_id=op.range.id) if isinstance(op, IdentityOperator) else
            op
            for op in operators if not isinstance(op, ZeroOperator)
        ]

    def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):

        operators = self._assemble_lincomb_preprocess_operators(operators)

        if not all(isinstance(op, BlockOperatorBase) for op in operators):
            return None

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
            c = coefficients[0]
            if c == 1:
                return self
            for (i, j) in np.ndindex(self._blocks.shape):
                blocks[i, j] = self._blocks[i, j] * c
            return self.__class__(blocks)

    def as_range_array(self, mu=None):

        def process_row(row, space):
            R = space.empty()
            for op in row:
                if op is not None:
                    R.append(op.as_range_array(mu))
            return R

        blocks = [process_row(row, space) for row, space in zip(self._blocks, self.range.subspaces)]
        return self.range.make_array(blocks) if self.blocked_range else blocks[0]

    def as_source_array(self, mu=None):

        def process_col(col, space):
            R = space.empty()
            for op in col:
                if op is not None:
                    R.append(op.as_source_array(mu))
            return R

        blocks = [process_col(col, space) for col, space in zip(self._blocks.T, self.source.subspaces)]
        return self.source.make_array(blocks) if self.blocked_source else blocks[0]


class BlockOperator(BlockOperatorBase):
    """A matrix of arbitrary |Operators|.

    This operator can be :meth:`applied <pymor.operators.interfaces.OperatorInterface.apply>`
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

    def __init__(self, block_space, component, source_id='STATE'):
        assert isinstance(block_space, BlockVectorSpace)
        assert 0 <= component < len(block_space.subspaces)
        blocks = [ZeroOperator(space, space) if i != component else IdentityOperator(space)
                  for i, space in enumerate(block_space.subspaces)]
        super().__init__(blocks, source_id=source_id)


class BlockEmbeddingOperator(BlockColumnOperator):

    def __init__(self, block_space, component, range_id='STATE'):
        assert isinstance(block_space, BlockVectorSpace)
        assert 0 <= component < len(block_space.subspaces)
        blocks = [ZeroOperator(space, space) if i != component else IdentityOperator(space)
                  for i, space in enumerate(block_space.subspaces)]
        super().__init__(blocks, range_id=range_id)


class BlockDiagonalOperator(BlockOperator):
    """Block diagonal |Operator| of arbitrary |Operators|.

    This is a specialization of :class:`BlockOperator` for the
    block diagonal case.
    """

    def __init__(self, blocks, source_id='STATE', range_id='STATE'):
        blocks = np.array(blocks)
        assert 1 <= blocks.ndim <= 2
        if blocks.ndim == 2:
            blocks = np.diag(blocks)
        n = len(blocks)
        blocks2 = np.empty((n, n), dtype=object)
        for i, op in enumerate(blocks):
            blocks2[i, i] = op
        super().__init__(blocks2, source_id=source_id, range_id=range_id)

    def apply(self, U, mu=None):
        assert U in self.source
        V_blocks = [self._blocks[i, i].apply(U.block(i), mu=mu) for i in range(self.num_range_blocks)]
        return self.range.make_array(V_blocks)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        U_blocks = [self._blocks[i, i].apply_adjoint(V.block(i), mu=mu) for i in range(self.num_source_blocks)]
        return self.source.make_array(U_blocks)

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range
        U_blocks = [self._blocks[i, i].apply_inverse(V.block(i), mu=mu, least_squares=least_squares)
                    for i in range(self.num_source_blocks)]
        return self.source.make_array(U_blocks)

    def apply_inverse_adjoint(self, U, mu=None, least_squares=False):
        assert U in self.source
        V_blocks = [self._blocks[i, i].apply_inverse_adjoint(U.block(i), mu=mu, least_squares=least_squares)
                    for i in range(self.num_source_blocks)]
        return self.range.make_array(V_blocks)

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

        operators = self._assemble_lincomb_preprocess_operators(operators)

        if not all(isinstance(op, BlockOperator) for op in operators):
            return None

        assert operators[0] is self

        # return ShiftedSecondOrderSystemOperator if possible
        if len(operators) == 2 and isinstance(operators[1], SecondOrderSystemOperator):
            return operators[1].assemble_lincomb(operators[::-1], coefficients[::-1],
                                                 solver_options=solver_options, name=name)

        # return BlockOperator if not all operators are BlockDiagonalOperators
        if not all(isinstance(op, self.__class__) for op in operators):
            return super().assemble_lincomb(operators, coefficients, solver_options=solver_options, name=name)

        # return BlockDiagonalOperator
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
            c = coefficients[0]
            if c == 1:
                return self
            for i in range(self.num_source_blocks):
                blocks[i] = self._blocks[i, i] * c
            return self.__class__(blocks)


class SecondOrderSystemOperator(BlockOperator):
    r"""BlockOperator appearing in SecondOrderSystem.to_lti().

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
        self.E = E
        self.K = K

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

    def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):

        operators = self._assemble_lincomb_preprocess_operators(operators)

        if not all(isinstance(op, BlockOperator) for op in operators):
            return None

        assert operators[0] is self

        # return ShiftedSecondOrderSystemOperator if possible
        if (len(operators) == 2 and isinstance(operators[1], BlockDiagonalOperator) and
                operators[1].num_source_blocks == 2 and operators[1].num_range_blocks == 2 and
                isinstance(operators[1]._blocks[0, 0], IdentityOperator)):
            return ShiftedSecondOrderSystemOperator(operators[1]._blocks[1, 1],
                                                    self.E,
                                                    self.K,
                                                    coefficients[1],
                                                    coefficients[0])

        # return BlockOperator
        blocks = np.empty(self._blocks.shape, dtype=object)
        if len(operators) > 1:
            for (i, j) in np.ndindex(self._blocks.shape):
                operators_ij = [op._blocks[i, j] for op in operators]
                blocks[i, j] = operators_ij[0].assemble_lincomb(operators_ij, coefficients,
                                                                solver_options=solver_options, name=name)
                if blocks[i, j] is None:
                    return None
            return BlockOperator(blocks)
        else:
            c = coefficients[0]
            if c == 1:
                return self
            for (i, j) in np.ndindex(self._blocks.shape):
                blocks[i, j] = self._blocks[i, j] * c
            return BlockOperator(blocks)


class ShiftedSecondOrderSystemOperator(BlockOperator):
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
        self.M = M
        self.E = E
        self.K = K
        self.a = a
        self.b = b

    def apply(self, U, mu=None):
        assert U in self.source
        V_blocks = [U.block(0) * self.a + U.block(1) * self.b,
                    self.K.apply(U.block(0), mu=mu) * (-self.b) +
                    self.M.apply(U.block(1), mu=mu) * self.a -
                    self.E.apply(U.block(1), mu=mu) * self.b]
        return self.range.make_array(V_blocks)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        U_blocks = [V.block(0) * self.a.conjugate() -
                    self.K.apply_adjoint(V.block(1), mu=mu) * self.b.conjugate(),
                    V.block(0) * self.b.conjugate() +
                    self.M.apply_adjoint(V.block(1), mu=mu) * self.a.conjugate() -
                    self.E.apply_adjoint(V.block(1), mu=mu) * self.b.conjugate()]
        return self.source.make_array(U_blocks)

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range
        aMmbEV0 = self.M.apply(V.block(0), mu=mu) * self.a - self.E.apply(V.block(0), mu=mu) * self.b
        KV0 = self.K.apply(V.block(0), mu=mu)
        a2MmabEpb2K = (self.a**2 * self.M - self.a * self.b * self.E + self.b**2 * self.K).assemble(mu=mu)
        a2MmabEpb2KiV1 = a2MmabEpb2K.apply_inverse(V.block(1), mu=mu, least_squares=least_squares)
        U_blocks = [a2MmabEpb2K.apply_inverse(aMmbEV0, mu=mu, least_squares=least_squares) -
                    a2MmabEpb2KiV1 * self.b,
                    a2MmabEpb2K.apply_inverse(KV0, mu=mu, least_squares=least_squares) * self.b +
                    a2MmabEpb2KiV1 * self.a]
        return self.source.make_array(U_blocks)

    def apply_inverse_adjoint(self, U, mu=None, least_squares=False):
        assert U in self.source
        a2MmabEpb2K = (self.a**2 * self.M - self.a * self.b * self.E + self.b**2 * self.K).assemble(mu=mu)
        a2MmabEpb2KitU0 = a2MmabEpb2K.apply_inverse_adjoint(U.block(0), mu=mu, least_squares=least_squares)
        a2MmabEpb2KitU1 = a2MmabEpb2K.apply_inverse_adjoint(U.block(1), mu=mu, least_squares=least_squares)
        V_blocks = [self.M.apply_adjoint(a2MmabEpb2KitU0, mu=mu) * self.a.conjugate() -
                    self.E.apply_adjoint(a2MmabEpb2KitU0, mu=mu) * self.b.conjugate() +
                    self.K.apply_adjoint(a2MmabEpb2KitU1, mu=mu) * self.b.conjugate(),
                    -a2MmabEpb2KitU0 * self.b.conjugate() + a2MmabEpb2KitU1 * self.a.conjugate()]
        return self.range.make_array(V_blocks)

    def assemble(self, mu=None):
        M = self.M.assemble(mu)
        E = self.E.assemble(mu)
        K = self.K.assemble(mu)
        if M == self.M and E == self.E and K == self.K:
            return self
        else:
            return self.__class__(M, E, K, self.a, self.b)

    def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):

        operators = self._assemble_lincomb_preprocess_operators(operators)

        if not all(isinstance(op, BlockOperator) for op in operators):
            return None

        assert operators[0] is self
        blocks = np.empty(self._blocks.shape, dtype=object)
        if len(operators) > 1:
            for (i, j) in np.ndindex(self._blocks.shape):
                operators_ij = [op._blocks[i, j] for op in operators]
                blocks[i, j] = operators_ij[0].assemble_lincomb(operators_ij, coefficients,
                                                                solver_options=solver_options, name=name)
                if blocks[i, j] is None:
                    return None
            return BlockOperator(blocks)
        else:
            c = coefficients[0]
            if c == 1:
                return self
            for (i, j) in np.ndindex(self._blocks.shape):
                blocks[i, j] = self._blocks[i, j] * c
            return BlockOperator(blocks)
