# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.rules import RuleTable, match_generic
from pymor.operators.block import (BlockOperator, BlockOperatorBase, BlockDiagonalOperator, SecondOrderModelOperator,
                                   ShiftedSecondOrderModelOperator)
from pymor.operators.constructions import LincombOperator, ZeroOperator, IdentityOperator, VectorArrayOperator


def assemble_lincomb(operators, coefficients, solver_options=None, name=None):
    return AssembleLincombRules(tuple(coefficients), solver_options, name).apply(tuple(operators))


class AssembleLincombRules(RuleTable):
    def __init__(self, coefficients, solver_options, name):
        super().__init__(use_caching=False)
        self.coefficients, self.solver_options, self.name \
            = coefficients, solver_options, name

    @match_generic(lambda ops: any(isinstance(op, ZeroOperator) for op in ops))
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

    @match_generic(lambda ops: (isinstance(ops[0], BlockOperatorBase)
                                and any(isinstance(op, IdentityOperator) for op in ops)))
    def action_BlockSpaceIdentityOperator(self, ops):
        new_ops = [BlockDiagonalOperator([IdentityOperator(s) for s in op.source.subspaces])
                   if isinstance(op, IdentityOperator) else op
                   for op in ops if not isinstance(op, ZeroOperator)]
        return self.apply(new_ops)

    @match_generic(lambda ops: any(isinstance(op, IdentityOperator) for op in ops))
    def action_IdentityOperator(self, ops):
        id_coeffs, new_ops, new_coeffs = [], [], []
        for op, coeff in zip(ops, self.coefficients):
            if isinstance(op, IdentityOperator):
                id_coeffs.append(coeff)
            else:
                new_ops.append(op)
                new_coeffs.append(coeff)

        id_coeff = sum(id_coeffs)
        id_op = IdentityOperator(ops[0].source)
        op_without_id = assemble_lincomb(new_ops, new_coeffs,
                                         solver_options=self.solver_options, name=self.name)

        if op_without_id:
            if id_coeff == 0:
                return op_without_id
            else:
                op = op_without_id._assemble_lincomb((op_without_id, id_op),
                                                     (1., id_coeff),
                                                     solver_options=self.solver_options, name=self.name)
                if op:
                    return op
                else:
                    return LincombOperator((op_without_id, id_op), (1., id_coeff),
                                           solver_options=self.solver_options, name=self.name)
        else:
            if id_coeff == 0:
                return LincombOperator(new_ops, new_coeffs,
                                       solver_options=self.solver_options, name=self.name)
            elif len(id_coeffs) > 1:
                return LincombOperator(new_ops + [id_op], new_coeffs + [id_coeff],
                                       solver_options=self.solver_options, name=self.name)
            else:
                return None

    @match_generic(lambda ops: all(isinstance(op, VectorArrayOperator)
                                   and op.adjoint == ops[0].adjoint
                                   and op.source == ops[0].source
                                   and op.range == ops[0].range
                                   for op in ops))
    def action_VectorArrrayOperator(self, ops):
        adjoint = ops[0].adjoint
        assert not self.solver_options

        coeffs = np.conj(self.coefficients) if adjoint else self.coefficients

        if coeffs[0] == 1:
            array = ops[0]._array.copy()
        else:
            array = ops[0]._array * coeffs[0]
        for op, c in zip(ops[1:], coeffs[1:]):
            array.axpy(c, op._array)

        return VectorArrayOperator(array, adjoint=adjoint, space_id=ops[0].space_id, name=self.name)

    @match_generic(lambda ops: (len(ops) == 2
                                and isinstance(ops[0], BlockDiagonalOperator)
                                and isinstance(ops[1], SecondOrderModelOperator)))
    def action_IdentityAndSecondOrderModelOperator(self, ops):
        return assemble_lincomb(ops[::-1], self.coefficients[::-1],
                                solver_options=self.solver_options, name=self.name)

    @match_generic(lambda ops: (len(ops) == 2
                                and isinstance(ops[0], SecondOrderModelOperator)
                                and isinstance(ops[1], BlockDiagonalOperator)
                                and isinstance(ops[1]._blocks[0, 0], IdentityOperator)))
    def action_SecondOrderModelOperatorAndShift(self, ops):
        return ShiftedSecondOrderModelOperator(ops[1]._blocks[1, 1],
                                               ops[0].E,
                                               ops[0].K,
                                               self.coefficients[1],
                                               self.coefficients[0])

    @match_generic(lambda ops: all(isinstance(op, BlockDiagonalOperator) for op in ops))
    def action_BlockDiagonalOpertors(self, ops):
        coefficients = self.coefficients
        num_source_blocks = ops[0].num_source_blocks
        blocks = np.empty((num_source_blocks,), dtype=object)
        if len(ops) > 1:
            for i in range(num_source_blocks):
                operators_i = [op._blocks[i, i] for op in ops]
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
                blocks[i] = ops[0]._blocks[i, i] * c
            return BlockDiagonalOperator(blocks)

    @match_generic(lambda ops: (isinstance(ops[0], SecondOrderModelOperator)
                                and all(isinstance(op, BlockOperatorBase) for op in ops[1:])))
    def action_SecondOrderModelOperator(self, ops):
        coefficients = self.coefficients
        shape = ops[0]._blocks.shape
        blocks = np.empty(shape, dtype=object)
        if len(ops) > 1:
            for (i, j) in np.ndindex(shape):
                operators_ij = [op._blocks[i, j] for op in ops]
                blocks[i, j] = assemble_lincomb(operators_ij, coefficients,
                                                solver_options=self.solver_options, name=self.name)
                if blocks[i, j] is None:
                    return None
            return BlockOperator(blocks)
        else:
            c = coefficients[0]
            if c == 1:
                return ops[0]
            for (i, j) in np.ndindex(shape):
                blocks[i, j] = ops[0]._blocks[i, j] * c
            return BlockOperator(blocks)

    @match_generic(lambda ops: (isinstance(ops[0], ShiftedSecondOrderModelOperator)
                                and all(isinstance(op, BlockOperatorBase) for op in ops[1:])))
    def action_ShiftedSecondOrderModelOperator(self, ops):
        coefficients = self.coefficients
        shape = ops[0]._blocks.shape
        blocks = np.empty(shape, dtype=object)
        if len(ops) > 1:
            for (i, j) in np.ndindex(shape):
                operators_ij = [op._blocks[i, j] for op in ops]
                blocks[i, j] = assemble_lincomb(operators_ij, coefficients,
                                                solver_options=self.solver_options, name=self.name)
                if blocks[i, j] is None:
                    return None
            return BlockOperator(blocks)
        else:
            c = coefficients[0]
            if c == 1:
                return ops[0]
            for (i, j) in np.ndindex(shape):
                blocks[i, j] = ops[0]._blocks[i, j] * c
            return BlockOperator(blocks)

    @match_generic(lambda ops: all(isinstance(op, BlockOperatorBase) for op in ops))
    def action_BlockOperatorBase(self, ops):
        coefficients = self.coefficients
        shape = ops[0]._blocks.shape
        blocks = np.empty(shape, dtype=object)
        if len(ops) > 1:
            for (i, j) in np.ndindex(shape):
                operators_ij = [op._blocks[i, j] for op in ops]
                blocks[i, j] = assemble_lincomb(operators_ij, coefficients,
                                                solver_options=self.solver_options, name=self.name)
                if blocks[i, j] is None:
                    return None
            return ops[0].__class__(blocks)
        else:
            c = coefficients[0]
            if c == 1:
                return ops[0]
            for (i, j) in np.ndindex(shape):
                blocks[i, j] = ops[0]._blocks[i, j] * c
            return ops[0].__class__(blocks)

    @match_generic(lambda ops: True)
    def action_call_assemble_lincomb_method(self, ops):
        op = ops[0]._assemble_lincomb(ops, self.coefficients,
                                      solver_options=self.solver_options, name=self.name)
        return op
