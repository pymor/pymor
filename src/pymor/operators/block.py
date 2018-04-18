# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import LincombOperator, IdentityOperator, ZeroOperator
from pymor.operators.interfaces import OperatorInterface
from pymor.vectorarrays.block import BlockVectorSpace


class BlockOperator(OperatorBase):
    """A matrix of arbitrary |Operators|.

    This operator can be :meth:`applied <pymor.operators.interfaces.OperatorInterface.apply>`
    to a compatible :class:`BlockVectorArrays <pymor.vectorarrays.block.BlockVectorArray>`.

    Parameters
    ----------
    blocks
        Two-dimensional array-like where each entry is an |Operator| or `None`.
    """

    def _operators(self):
        """Iterator over operators."""
        for (i, j) in np.ndindex(self._blocks.shape):
            yield self._blocks[i, j]

    def __init__(self, blocks, source_id='STATE', range_id='STATE'):
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
                self._blocks[i, j] = ZeroOperator(range_types[i], source_types[j])

        self.source = BlockVectorSpace(source_types, id_=source_id)
        self.range = BlockVectorSpace(range_types, id_=range_id)
        self.num_source_blocks = len(source_types)
        self.num_range_blocks = len(range_types)
        self.linear = all(op.linear for op in self._operators())
        self.build_parameter_type(*self._operators())

    @property
    def T(self):
        return type(self)(np.vectorize(lambda op: op.T if op else None)(self._blocks.T),
                          source_id=self.range.id, range_id=self.source.id)

    @classmethod
    def hstack(cls, operators, source_id='STATE', range_id='STATE'):
        """Horizontal stacking of |Operators|.

        Parameters
        ----------
        operators
            An iterable where each item is an |Operator| or `None`.
        """
        blocks = np.array([[op for op in operators]])
        return cls(blocks, source_id=source_id, range_id=range_id)

    @classmethod
    def vstack(cls, operators, source_id='STATE', range_id='STATE'):
        """Vertical stacking of |Operators|.

        Parameters
        ----------
        operators
            An iterable where each item is an |Operator| or `None`.
        """
        blocks = np.array([[op] for op in operators])
        return cls(blocks, source_id=source_id, range_id=range_id)

    def apply(self, U, mu=None):
        assert U in self.source

        V_blocks = [None for i in range(self.num_range_blocks)]
        for (i, j), op in np.ndenumerate(self._blocks):
            Vi = op.apply(U.block(j), mu=mu)
            if V_blocks[i] is None:
                V_blocks[i] = Vi
            else:
                V_blocks[i] += Vi

        return self.range.make_array(V_blocks)

    def apply_transpose(self, V, mu=None):
        assert V in self.range

        U_blocks = [None for j in range(self.num_source_blocks)]
        for (i, j), op in np.ndenumerate(self._blocks):
            Uj = op.apply_transpose(V.block(i), mu=mu)
            if U_blocks[j] is None:
                U_blocks[j] = Uj
            else:
                U_blocks[j] += Uj

        U = self.source.make_array(U_blocks)

        return U

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
            c = coefficients[0]
            if c == 1:
                return self
            for (i, j) in np.ndindex(self._blocks.shape):
                blocks[i, j] = self._blocks[i, j] * c
            return self.__class__(blocks)

    def as_range_array(self):

        def process_row(row, space):
            R = space.empty()
            for op in row:
                if op is not None:
                    R.append(op.as_range_array())
            return R

        blocks = [process_row(row, space) for row, space in zip(self._blocks, self.range.subspaces)]
        return self.range.make_array(blocks)

    def as_source_array(self):

        def process_col(col, space):
            R = space.empty()
            for op in col:
                if op is not None:
                    R.append(op.as_source_array())
            return R

        blocks = [process_col(col, space) for col, space in zip(self._blocks.T, self.source.subspaces)]
        return self.source.make_array(blocks)


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

    def apply_transpose(self, V, mu=None):
        assert V in self.range
        U_blocks = [self._blocks[i, i].apply_transpose(V.block(i), mu=mu) for i in range(self.num_source_blocks)]
        return self.source.make_array(U_blocks)

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range
        U_blocks = [self._blocks[i, i].apply_inverse(V.block(i), mu=mu, least_squares=least_squares)
                    for i in range(self.num_source_blocks)]
        return self.source.make_array(U_blocks)

    def apply_inverse_transpose(self, U, mu=None, least_squares=False):
        assert U in self.source
        V_blocks = [self._blocks[i, i].apply_inverse_transpose(U.block(i), mu=mu, least_squares=least_squares)
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
        assert operators[0] is self

        # return ShiftedSecondOrderOperator if possible
        if len(operators) == 2 and isinstance(operators[1], SecondOrderOperator):
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


class SecondOrderOperator(BlockOperator):
    """BlockOperator appearing in SecondOrderSystem.to_lti().

    This represents a block operator

    .. math::
        \mathcal{A} =
        \begin{bmatrix}
            0 & I \\
            -K & -D
        \end{bmatrix},

    which satisfies

    .. math::
        \mathcal{A}^T
        &=
        \begin{bmatrix}
            0 & -K^T \\
            I & -D^T
        \end{bmatrix}, \\
        \mathcal{A}^{-1}
        &=
        \begin{bmatrix}
            -K^{-1} D & -K^{-1} \\
            I & 0
        \end{bmatrix}, \\
        \mathcal{A}^{-T}
        &=
        \begin{bmatrix}
            -D^T K^{-T} & I \\
            -K^{-T} & 0
        \end{bmatrix}.

    Parameters
    ----------
    D
        |Operator|.
    K
        |Operator|.
    """

    def __init__(self, D, K):
        super().__init__([[None, IdentityOperator(D.source)],
                          [K * (-1), D * (-1)]])
        self.D = D
        self.K = K

    def apply(self, U, mu=None):
        assert U in self.source
        V_blocks = [U.block(1),
                    -self.K.apply(U.block(0), mu=mu) - self.D.apply(U.block(1), mu=mu)]
        return self.range.make_array(V_blocks)

    def apply_transpose(self, V, mu=None):
        assert V in self.range
        U_blocks = [-self.K.apply_transpose(V.block(1), mu=mu),
                    V.block(0) - self.D.apply_transpose(V.block(1), mu=mu)]
        return self.source.make_array(U_blocks)

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range
        U_blocks = [-self.K.apply_inverse(self.D.apply(V.block(0), mu=mu) + V.block(1), mu=mu,
                                          least_squares=least_squares),
                    V.block(0)]
        return self.source.make_array(U_blocks)

    def apply_inverse_transpose(self, U, mu=None, least_squares=False):
        assert U in self.source
        KitU0 = self.K.apply_inverse_transpose(U.block(0), mu=mu, least_squares=least_squares)
        V_blocks = [-self.D.apply_transpose(KitU0, mu=mu) + U.block(1),
                    -KitU0]
        return self.range.make_array(V_blocks)

    def assemble(self, mu=None):
        D = self.D.assemble(mu)
        K = self.K.assemble(mu)
        if D == self.D and K == self.K:
            return self
        else:
            return self.__class__(D, K)

    def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
        assert operators[0] is self

        # return ShiftedSecondOrderOperator if possible
        if (len(operators) == 2 and isinstance(operators[1], BlockDiagonalOperator) and
                operators[1].num_source_blocks == 2 and operators[1].num_range_blocks == 2 and
                isinstance(operators[1]._blocks[0, 0], IdentityOperator)):
            return ShiftedSecondOrderOperator(operators[1]._blocks[1, 1],
                                              self.D,
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


class ShiftedSecondOrderOperator(BlockOperator):
    """BlockOperator appearing in second-order systems.

    This represents a block operator

    .. math::
        a \mathcal{E} + b \mathcal{A} =
        \begin{bmatrix}
            a I & b I \\
            -b K & a M - b D
        \end{bmatrix},

    which satisfies

    .. math::
        (a \mathcal{E} + b \mathcal{A})^T
        &=
        \begin{bmatrix}
            a I & -b K^T \\
            b I & a M^T - b D^T
        \end{bmatrix}, \\
        (a \mathcal{E} + b \mathcal{A})^{-1}
        &=
        \begin{bmatrix}
            (a^2 M - a b D + b^2 K)^{-1} (a M - b D)
            & -b (a^2 M - a b D + b^2 K)^{-1} \\
            b (a^2 M - a b D + b^2 K)^{-1} K
            & a (a^2 M - a b D + b^2 K)^{-1}
        \end{bmatrix}, \\
        (a \mathcal{E} + b \mathcal{A})^{-T}
        &=
        \begin{bmatrix}
            (a M - b D)^T (a^2 M - a b D + b^2 K)^{-T}
            & b K^T (a^2 M - a b D + b^2 K)^{-T} \\
            -b (a^2 M - a b D + b^2 K)^{-T} & a (a^2 M - a b D + b^2 K)^{-T}
        \end{bmatrix}.

    Parameters
    ----------
    M
        |Operator|.
    D
        |Operator|.
    K
        |Operator|.
    p
        Complex number.
    """

    def __init__(self, M, D, K, a, b):
        super().__init__([[IdentityOperator(M.source) * a, IdentityOperator(M.source) * b],
                          [K * (-b), LincombOperator([M, D], [a, -b])]])
        self.M = M
        self.D = D
        self.K = K
        self.a = a
        self.b = b

    def apply(self, U, mu=None):
        assert U in self.source
        V_blocks = [U.block(0) * self.a + U.block(1) * self.b,
                    self.K.apply(U.block(0), mu=mu) * (-self.b) +
                    self.M.apply(U.block(1), mu=mu) * self.a -
                    self.D.apply(U.block(1), mu=mu) * self.b]
        return self.range.make_array(V_blocks)

    def apply_transpose(self, V, mu=None):
        assert V in self.range
        U_blocks = [V.block(0) * self.a - self.K.apply_transpose(V.block(1), mu=mu) * self.b,
                    V.block(0) * self.b +
                    self.M.apply_transpose(V.block(1), mu=mu) * self.a -
                    self.D.apply_transpose(V.block(1), mu=mu) * self.b]
        return self.source.make_array(U_blocks)

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range
        aMmbDV0 = self.M.apply(V.block(0), mu=mu) * self.a - self.D.apply(V.block(0), mu=mu) * self.b
        KV0 = self.K.apply(V.block(0), mu=mu)
        a2MmabDpb2K = LincombOperator([self.M, self.D, self.K], [self.a ** 2, -self.a * self.b, self.b ** 2])
        a2MmabDpb2KiV1 = a2MmabDpb2K.apply_inverse(V.block(1), mu=mu, least_squares=least_squares)
        U_blocks = [a2MmabDpb2K.apply_inverse(aMmbDV0, mu=mu, least_squares=least_squares) -
                    a2MmabDpb2KiV1 * self.b,
                    a2MmabDpb2K.apply_inverse(KV0, mu=mu, least_squares=least_squares) * self.b +
                    a2MmabDpb2KiV1 * self.a]
        return self.source.make_array(U_blocks)

    def apply_inverse_transpose(self, U, mu=None, least_squares=False):
        assert U in self.source
        a2MmabDpb2K = LincombOperator([self.M, self.D, self.K], [self.a ** 2, -self.a * self.b, self.b ** 2])
        a2MmabDpb2KitU0 = a2MmabDpb2K.apply_inverse_transpose(U.block(0), mu=mu, least_squares=least_squares)
        a2MmabDpb2KitU1 = a2MmabDpb2K.apply_inverse_transpose(U.block(1), mu=mu, least_squares=least_squares)
        V_blocks = [self.M.apply_transpose(a2MmabDpb2KitU0, mu=mu) * self.a -
                    self.D.apply_transpose(a2MmabDpb2KitU0, mu=mu) * self.b +
                    self.K.apply_transpose(a2MmabDpb2KitU1, mu=mu) * self.b,
                    -a2MmabDpb2KitU0 * self.b + a2MmabDpb2KitU1 * self.a]
        return self.range.make_array(V_blocks)

    def assemble(self, mu=None):
        M = self.M.assemble(mu)
        D = self.D.assemble(mu)
        K = self.K.assemble(mu)
        if M == self.M and D == self.D and K == self.K:
            return self
        else:
            return self.__class__(M, D, K, self.a, self.b)

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
            return BlockOperator(blocks)
        else:
            c = coefficients[0]
            if c == 1:
                return self
            for (i, j) in np.ndindex(self._blocks.shape):
                blocks[i, j] = self._blocks[i, j] * c
            return BlockOperator(blocks)
