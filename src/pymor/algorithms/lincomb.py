# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import chain

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.rules import RuleTable, match_generic, match_class_all, match_class_any, match_always
from pymor.core.exceptions import RuleNotMatchingError
from pymor.operators.block import (BlockOperator, BlockRowOperator, BlockColumnOperator, BlockOperatorBase,
                                   BlockDiagonalOperator, SecondOrderModelOperator)
from pymor.operators.constructions import (ZeroOperator, IdentityOperator, VectorArrayOperator, LincombOperator,
                                           LowRankOperator, LowRankUpdatedOperator)
from pymor.vectorarrays.constructions import cat_arrays


def assemble_lincomb(operators, coefficients, solver_options=None, name=None):
    """Try to assemble a linear combination of the given operators.

    Returns a new |Operator| which represents the sum ::

        c_1*O_1 + ... + c_N*O_N

    where `O_i` are |Operators| and `c_i` scalar coefficients.

    This function is called in the :meth:`assemble` method of |LincombOperator| and
    is not intended to be used directly.

    To form the linear combination of backend |Operators| (containing actual matrix data),
    :meth:`~pymor.operators.interface.Operator._assemble_lincomb` will be called
    on the first |Operator| in the linear combination.

    Parameters
    ----------
    operators
        List of |Operators| `O_i` whose linear combination is formed.
    coefficients
        List of the corresponding linear coefficients `c_i`.
    solver_options
        |solver_options| for the assembled operator.
    name
        Name of the assembled operator.

    Returns
    -------
    The assembled |Operator|.
    """
    return AssembleLincombRules(tuple(coefficients), solver_options, name).apply(tuple(operators))


class AssembleLincombRules(RuleTable):
    def __init__(self, coefficients, solver_options, name):
        super().__init__(use_caching=False)
        self.__auto_init(locals())

    @match_always
    def action_zero_coeff(self, ops):
        if all(coeff != 0 for coeff in self.coefficients):
            raise RuleNotMatchingError
        without_zero = [(op, coeff)
                        for op, coeff in zip(ops, self.coefficients)
                        if coeff != 0]
        if len(without_zero) == 0:
            return ZeroOperator(ops[0].range, ops[0].source, name=self.name)
        else:
            new_ops, new_coeffs = zip(*without_zero)
            return assemble_lincomb(new_ops, new_coeffs,
                                    solver_options=self.solver_options, name=self.name)

    @match_class_any(ZeroOperator)
    def action_ZeroOperator(self, ops):
        without_zero = [(op, coeff)
                        for op, coeff in zip(ops, self.coefficients)
                        if not isinstance(op, ZeroOperator)]
        if len(without_zero) == 0:
            return ZeroOperator(ops[0].range, ops[0].source, name=self.name)
        else:
            new_ops, new_coeffs = zip(*without_zero)
            return assemble_lincomb(new_ops, new_coeffs,
                                    solver_options=self.solver_options, name=self.name)

    @match_class_all(IdentityOperator)
    def action_IdentityOperator(self, ops):
        coeff = sum(self.coefficients)
        if coeff == 0:
            return ZeroOperator(ops[0].source, ops[0].source, name=self.name)
        else:
            return LincombOperator([IdentityOperator(ops[0].source, name=self.name)],
                                   [coeff],
                                   name=self.name)

    @match_class_any(BlockOperatorBase)
    @match_class_any(IdentityOperator)
    def action_BlockSpaceIdentityOperator(self, ops):
        new_ops = tuple(
            BlockDiagonalOperator([IdentityOperator(s) for s in op.source.subspaces])
            if isinstance(op, IdentityOperator) else op
            for op in ops if not isinstance(op, ZeroOperator)
        )
        return self.apply(new_ops)

    @match_class_all(VectorArrayOperator)
    def action_VectorArrayOperator(self, ops):
        if not all(op.adjoint == ops[0].adjoint for op in ops):
            raise RuleNotMatchingError

        adjoint = ops[0].adjoint
        assert not self.solver_options

        coeffs = np.conj(self.coefficients) if adjoint else self.coefficients

        if coeffs[0] == 1:
            array = ops[0].array.copy()
        else:
            array = ops[0].array * coeffs[0]
        for op, c in zip(ops[1:], coeffs[1:]):
            array.axpy(c, op.array)

        return VectorArrayOperator(array, adjoint=adjoint, space_id=ops[0].space_id, name=self.name)

    @match_class_any(SecondOrderModelOperator)
    def action_SecondOrderModelOperator(self, ops):
        def is_scaled_iden_like(op):
            if isinstance(op, (ZeroOperator, IdentityOperator)):
                return True
            if (isinstance(op, LincombOperator)
                    and len(op.operators) == 1
                    and isinstance(op.operators[0], IdentityOperator)):
                return True
            return False

        def is_so_op_like(op):
            if isinstance(op, SecondOrderModelOperator):
                return True
            if is_scaled_iden_like(op.blocks[0, 0]) and is_scaled_iden_like(op.blocks[0, 1]):
                return True
            return False

        if not all(is_so_op_like(op) for op in ops):
            raise RuleNotMatchingError

        def so_op_parts(op):
            if isinstance(op, SecondOrderModelOperator):
                return op.alpha, op.beta, op.A, op.B

            def scaled_iden_coeff(op):
                if isinstance(op, ZeroOperator):
                    return 0
                if isinstance(op, IdentityOperator):
                    return 1
                return op.coefficients[0]

            return (scaled_iden_coeff(op.blocks[0, 0]),
                    scaled_iden_coeff(op.blocks[0, 1]),
                    op.blocks[1, 1],
                    op.blocks[1, 0])

        alphas, betas, As, Bs = zip(*(so_op_parts(op) for op in ops))
        alpha = sum(a * b for a, b in zip(self.coefficients, alphas))
        beta = sum(a * b for a, b in zip(self.coefficients, betas))
        A = assemble_lincomb(As, self.coefficients)
        B = assemble_lincomb(Bs, self.coefficients)
        so_op = SecondOrderModelOperator(alpha, beta, A, B)
        return so_op

    @match_class_all(BlockDiagonalOperator)
    def action_BlockDiagonalOperator(self, ops):
        coefficients = self.coefficients
        num_source_blocks = ops[0].num_source_blocks
        blocks = np.empty((num_source_blocks,), dtype=object)
        if len(ops) > 1:
            for i in range(num_source_blocks):
                operators_i = [op.blocks[i, i] for op in ops]
                blocks[i] = assemble_lincomb(operators_i, coefficients,
                                             solver_options=self.solver_options, name=self.name)
                if blocks[i] is None:
                    return None
            return BlockDiagonalOperator(blocks)
        else:
            c = coefficients[0]
            if c == 1:
                return ops[0]
            for i in range(num_source_blocks):
                blocks[i] = ops[0].blocks[i, i] * c
            return BlockDiagonalOperator(blocks)

    @match_class_all(BlockOperatorBase)
    def action_BlockOperatorBase(self, ops):
        coefficients = self.coefficients
        shape = ops[0].blocks.shape
        blocks = np.empty(shape, dtype=object)
        operator_type = ((BlockOperator if ops[0].blocked_source else BlockColumnOperator) if ops[0].blocked_range
                         else BlockRowOperator)
        if len(ops) > 1:
            for (i, j) in np.ndindex(shape):
                operators_ij = [op.blocks[i, j] for op in ops]
                blocks[i, j] = assemble_lincomb(operators_ij, coefficients,
                                                solver_options=self.solver_options, name=self.name)
                if blocks[i, j] is None:
                    return None
            return operator_type(blocks)
        else:
            c = coefficients[0]
            if c == 1:
                return ops[0]
            for (i, j) in np.ndindex(shape):
                blocks[i, j] = ops[0].blocks[i, j] * c
            return operator_type(blocks)

    @match_generic(lambda ops: sum(1 for op in ops if isinstance(op, LowRankOperator)) >= 2)
    def action_merge_low_rank_operators(self, ops):
        low_rank = []
        not_low_rank = []
        for op, coeff in zip(ops, self.coefficients):
            if isinstance(op, LowRankOperator):
                low_rank.append((op, coeff))
            else:
                not_low_rank.append((op, coeff))
        inverted = [op.inverted for op, _ in low_rank]
        if len(inverted) >= 2 and any(inverted) and any(not _ for _ in inverted):
            return None
        inverted = inverted[0]
        left = cat_arrays([op.left for op, _ in low_rank])
        right = cat_arrays([op.right for op, _ in low_rank])
        core = []
        for op, coeff in low_rank:
            core.append(op.core)
            if inverted:
                core[-1] /= coeff
            else:
                core[-1] *= coeff
        core = spla.block_diag(*core)
        new_low_rank_op = LowRankOperator(left, core, right, inverted=inverted)
        if len(not_low_rank) == 0:
            return new_low_rank_op
        else:
            new_ops, new_coeffs = zip(*not_low_rank)
            return assemble_lincomb(chain(new_ops, [new_low_rank_op]), chain(new_coeffs, [1]),
                                    solver_options=self.solver_options, name=self.name)

    @match_generic(lambda ops: len(ops) >= 2)
    @match_class_any(LowRankOperator, LowRankUpdatedOperator)
    def action_merge_into_low_rank_updated_operator(self, ops):
        new_ops = []
        new_lr_ops = []
        new_coeffs = []
        new_lr_coeffs = []
        for op, coeff in zip(ops, self.coefficients):
            if isinstance(op, LowRankOperator):
                new_lr_ops.append(op)
                new_lr_coeffs.append(coeff)
            elif isinstance(op, LowRankUpdatedOperator):
                new_ops.append(op.operators[0])
                new_coeffs.append(coeff * op.coefficients[0])
                new_lr_ops.append(op.operators[1])
                new_lr_coeffs.append(coeff * op.coefficients[1])
            else:
                new_ops.append(op)
                new_coeffs.append(coeff)
        lru_op = assemble_lincomb(new_ops, new_coeffs)
        lru_lr_op = assemble_lincomb(new_lr_ops, new_lr_coeffs)
        lru_lr_coeff = 1
        if isinstance(lru_lr_op, LincombOperator):
            lru_lr_op, lru_lr_coeff = lru_lr_op.operators[0], lru_lr_op.coefficients[0]
        return LowRankUpdatedOperator(lru_op, lru_lr_op, 1, lru_lr_coeff,
                                      solver_options=self.solver_options, name=self.name)

    @match_always
    def action_call_assemble_lincomb_method(self, ops):
        id_coeffs, ops_without_id, coeffs_without_id = [], [], []
        for op, coeff in zip(ops, self.coefficients):
            if isinstance(op, IdentityOperator):
                id_coeffs.append(coeff)
            else:
                ops_without_id.append(op)
                coeffs_without_id.append(coeff)
        id_coeff = sum(id_coeffs)

        op = ops_without_id[0]._assemble_lincomb(ops_without_id, coeffs_without_id, identity_shift=id_coeff,
                                                 solver_options=self.solver_options, name=self.name)

        if not op:
            raise RuleNotMatchingError
        return op

    @match_always
    def action_return_lincomb(self, ops):
        return LincombOperator(ops, self.coefficients, name=self.name)
