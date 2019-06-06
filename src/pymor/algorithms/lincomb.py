# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.rules import (
    RuleTable,
    match_generic,
    match_class_all,
    match_class_any,
    match_always,
)
from pymor.core.exceptions import RuleNotMatchingError
from pymor.operators.block import (
    BlockOperator,
    BlockRowOperator,
    BlockColumnOperator,
    BlockOperatorBase,
    BlockDiagonalOperator,
    SecondOrderModelOperator,
    ShiftedSecondOrderModelOperator,
)
from pymor.operators.constructions import (
    ZeroOperator,
    IdentityOperator,
    VectorArrayOperator,
    LincombOperator,
)


def assemble_lincomb(operators, coefficients, solver_options=None, name=None):
    """Try to assemble a linear combination of the given operators.

    Returns a new |Operator| which represents the sum ::

        c_1*O_1 + ... + c_N*O_N

    where `O_i` are |Operators| and `c_i` scalar coefficients.

    This function is called in the :meth:`assemble` method of |LincombOperator| and
    is not intended to be used directly. The assembled |Operator| is expected to
    no longer be a |LincombOperator| nor should it contain any |LincombOperators|.
    If an assembly of the given linear combination is not possible, `None` is returned.
    The special case of a |LincombOperator| with a single operator (i.e. a scaled |Operator|)
    is allowed as assemble_lincomb implements `apply_inverse` for this special case.

    To form the linear combination of backend |Operators| (containing actual matrix data),
    :meth:`~pymor.operators.interfaces.OperatorInterface._assemble_lincomb` will be called
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
    The assembled |Operator| if assembly is possible, otherwise `None`.
    """

    return AssembleLincombRules(tuple(coefficients), solver_options, name).apply(
        tuple(operators)
    )


class AssembleLincombRules(RuleTable):
    def __init__(self, coefficients, solver_options, name):
        super().__init__(use_caching=False)
        self.coefficients, self.solver_options, self.name = (
            coefficients,
            solver_options,
            name,
        )

    @match_class_any(ZeroOperator)
    def action_ZeroOperator(self, ops):
        without_zero = [
            (op, coeff)
            for op, coeff in zip(ops, self.coefficients)
            if not isinstance(op, ZeroOperator)
        ]
        if len(without_zero) == 0:
            return ZeroOperator(ops[0].range, ops[0].source, name=self.name)
        else:
            new_ops, new_coeffs = zip(*without_zero)
            return assemble_lincomb(
                new_ops, new_coeffs, solver_options=self.solver_options, name=self.name
            )

    @match_class_all(IdentityOperator)
    def action_IdentityOperator(self, ops):
        coeff = sum(self.coefficients)
        if coeff == 0:
            return ZeroOperator(ops[0].source, ops[0].source, name=self.name)
        else:
            return LincombOperator(
                [IdentityOperator(ops[0].source, name=self.name)],
                [coeff],
                name=self.name,
            )

    @match_class_any(BlockOperatorBase)
    @match_class_any(IdentityOperator)
    def action_BlockSpaceIdentityOperator(self, ops):
        new_ops = [
            BlockDiagonalOperator([IdentityOperator(s) for s in op.source.subspaces])
            if isinstance(op, IdentityOperator)
            else op
            for op in ops
            if not isinstance(op, ZeroOperator)
        ]
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

        return VectorArrayOperator(
            array, adjoint=adjoint, space_id=ops[0].space_id, name=self.name
        )

    @match_generic(lambda ops: len(ops) == 2)
    @match_class_any(SecondOrderModelOperator)
    @match_class_any(BlockDiagonalOperator)
    def action_IdentityAndSecondOrderModelOperator(self, ops):
        if isinstance(ops[1], SecondOrderModelOperator):
            ops, coeffs = ops[::-1], self.coefficients[::-1]
        else:
            ops, coeffs = ops, self.coefficients
        if not isinstance(ops[1].blocks[0, 0], IdentityOperator):
            raise RuleNotMatchingError

        return ShiftedSecondOrderModelOperator(
            ops[1].blocks[1, 1], ops[0].E, ops[0].K, coeffs[1], coeffs[0]
        )

    @match_class_all(BlockDiagonalOperator)
    def action_BlockDiagonalOperator(self, ops):
        coefficients = self.coefficients
        num_source_blocks = ops[0].num_source_blocks
        blocks = np.empty((num_source_blocks,), dtype=object)
        if len(ops) > 1:
            for i in range(num_source_blocks):
                operators_i = [op.blocks[i, i] for op in ops]
                blocks[i] = assemble_lincomb(
                    operators_i,
                    coefficients,
                    solver_options=self.solver_options,
                    name=self.name,
                )
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
        operator_type = (
            (BlockOperator if ops[0].blocked_source else BlockColumnOperator)
            if ops[0].blocked_range
            else BlockRowOperator
        )
        if len(ops) > 1:
            for (i, j) in np.ndindex(shape):
                operators_ij = [op.blocks[i, j] for op in ops]
                blocks[i, j] = assemble_lincomb(
                    operators_ij,
                    coefficients,
                    solver_options=self.solver_options,
                    name=self.name,
                )
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

        op = ops_without_id[0]._assemble_lincomb(
            ops_without_id,
            coeffs_without_id,
            identity_shift=id_coeff,
            solver_options=self.solver_options,
            name=self.name,
        )

        if not op:
            raise RuleNotMatchingError
        return op

    @match_generic(lambda ops: len(ops) == 1)
    def action_only_one_operator(self, ops):
        return LincombOperator(ops, self.coefficients, name=self.name)

    @match_always
    def action_failed(self, ops):
        return None
