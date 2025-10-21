# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.rules import RuleTable, match_class
from pymor.models.interface import Model
from pymor.operators.block import BlockOperator
from pymor.operators.constructions import (
    AdjointOperator,
    ConcatenationOperator,
    IdentityOperator,
    LincombOperator,
    VectorArrayOperator,
)
from pymor.operators.interface import Operator, as_array_max_length
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ParameterFunctional
from pymor.vectorarrays.numpy import NumpyVectorSpace


def expand(obj):
    """Expand concatenations of |LincombOperators|.

    To any given |Operator| or |Model|, the following
    transformations are applied recursively:

    - :class:`Concatenations <pymor.operators.constructions.ConcatenationOperator>`
      of |LincombOperators| are expanded. E.g. ::

          (O1 + O2) @ (O3 + O4)

      becomes::

          O1 @ O3 + O1 @ O4 + O2 @ O3 + O2 @ O4

    - |LincombOperators| inside |LincombOperators| are merged into a single
      |LincombOperator|.

    - |ConcatenationOperators| inside |ConcatenationOperators| are merged into a
      single |ConcatenationOperator|.

    - |LincombOperators| inside |AdjointOperators| are distributed over the linear combination ::

          (∑ c_i A_i)^* = ∑ conj(c_i) A_i^*.

    - |ConcatenationOperators| inside |AdjointOperators| are reversed ::

          (A @ B @ C)^* = C^* @ B^* @ A^*.

    - |BlockOperators| inside |LincombOperator| are distributed element-wise and collected
      into a single |LincombOperator| of |BlockOperators|. Numeric coefficients are merged
      into the constant block using :func:`contract <pymor.algorithms.simplify.contract>`,
      while parametric coefficients remain as coefficients of the resulting |LincombOperator|.

    Parameters
    ----------
    obj
        Either a |Model| or an |Operator| to which the expansion rules are
        applied recursively for all
        :meth:`children <pymor.algorithms.rules.RuleTable.get_children>`.

    Returns
    -------
    The transformed object.
    """
    return ExpandRules().apply(obj)


def contract(obj):
    """Contract linear combinations and concatenations of |Operators|.

    To any given |Operator| or |Model|, the following
    transformations are applied recursively:

    - :class:`Concatenations <pymor.operators.constructions.ConcatenationOperator>`
      of non-parametric linear |Operators| are contracted into a single |Operator|
      when possible. Products of sparse matrices are not computed, however.

    - Non-parametric :class:`linear combinations <pymor.operators.constructions.LincombOperator>`
      of non-parametric operators are merged.

    Parameters
    ----------
    obj
        Either a |Model| or an |Operator| to which the contraction rules are
        applied recursively for all
        :meth:`children <pymor.algorithms.rules.RuleTable.get_children>`.

    Returns
    -------
    The transformed object.
    """
    return ContractRules().apply(obj)


class ExpandRules(RuleTable):
    """|RuleTable| for the :func:`expand` algorithm."""

    def __init__(self):
        super().__init__(use_caching=True)

    @match_class(LincombOperator)
    def action_LincombOperator(self, op):
        # recursively expand all children
        op = self.replace_children(op)

        # merge child LincombOperators
        if any(isinstance(o, LincombOperator) for o in op.operators):
            ops, coeffs = [], []
            for c, o in zip(op.coefficients, op.operators, strict=True):
                if isinstance(o, LincombOperator):
                    coeffs.extend(c * cc for cc in o.coefficients)
                    ops.extend(o.operators)
                else:
                    coeffs.append(c)
                    ops.append(o)
            op = op.with_(operators=ops, coefficients=coeffs)
        return op

    @match_class(AdjointOperator)
    def action_AdjointOperator(self, op):
        inner = self.apply(op.operator)

        # If the inside is a LincombOperator, distribute adjoint over the sum:
        # (∑ c_i A_i)^* = ∑ conj(c_i) A_i^*
        if isinstance(inner, LincombOperator):
            ops = [AdjointOperator(o, source_product=op.source_product, range_product=op.range_product)
                   for o in inner.operators]
            coeffs = [c.conjugate() for c in inner.coefficients]
            return LincombOperator(ops, coeffs)

        # If the inside is a ConcatenationOperator, distribute adjoint over the concatenation:
        # (A @ B @ C)^* = C^* @ B^* @ A^*
        if isinstance(inner, ConcatenationOperator):
            if len(inner.operators) >= 2:
                last, *middle, first = inner.operators[::-1]
                adj_factors = ([AdjointOperator(last, source_product=op.source_product)] +
                            [op.H for op in middle] +
                            [AdjointOperator(first, range_product=op.range_product)])

                return ConcatenationOperator(adj_factors)

        return AdjointOperator(inner, source_product=op.source_product, range_product=op.range_product)

    @match_class(BlockOperator)
    def action_BlockOperator(self, op):
        # recursively expand all children
        op = self.replace_children(op)

        nrows, ncols = op.blocks.shape

        # Build the constant part: copy non-Lincomb blocks; Lincomb blocks start as zero
        const_blocks = np.full((nrows, ncols), None, dtype=object)
        for (i, j), b in np.ndenumerate(op.blocks):
            if not isinstance(b, LincombOperator):
                const_blocks[i, j] = b

        expanded_terms = []
        for (i, j), b in np.ndenumerate(op.blocks):
            if not isinstance(b, LincombOperator):
                continue
            for c_k, a_k in zip(b.coefficients, b.operators, strict=True):
                if isinstance(c_k, ParameterFunctional):
                    blocks_k = np.full((nrows, ncols), None, dtype=object)
                    blocks_k[i, j] = a_k
                    expanded_terms.append((c_k, BlockOperator(blocks_k, range_spaces=op.range.subspaces,
                                                            source_spaces=op.source.subspaces)))
                else:
                    cur = const_blocks[i, j]
                    if cur is None:
                        const_blocks[i, j] = contract(c_k * a_k)
                    else:
                        const_blocks[i, j] += contract(c_k * a_k)

        const_part = BlockOperator(const_blocks, range_spaces=op.range.subspaces, source_spaces=op.source.subspaces)

        coeffs, ops = zip(*expanded_terms, strict=True)
        if const_blocks.any():
            ops = [const_part] + list(ops)
            coeffs = [1.] + list(coeffs)

        if len(ops) == 1 and list(coeffs) == [1.]:
            return ops[0]

        return LincombOperator(ops, coeffs)

    @match_class(ConcatenationOperator)
    def action_ConcatenationOperator(self, op):
        op = self.replace_children(op)

        # merge child ConcatenationOperators
        if any(isinstance(o, ConcatenationOperator) for o in op.operators):
            ops = []
            for o in op.operators:
                if isinstance(o, ConcatenationOperator):
                    ops.extend(o.operators)
                else:
                    ops.append(o)
            op = op.with_(operators=ops)

        # expand concatenations with LincombOperators
        if any(isinstance(o, LincombOperator) for o in op.operators):
            i = next(iter(i for i, o in enumerate(op.operators) if isinstance(o, LincombOperator)))
            left, right = op.operators[:i], op.operators[i+1:]
            ops = [ConcatenationOperator(left + (o,) + right) for o in op.operators[i].operators]
            op = op.operators[i].with_(operators=ops)

            # there can still be LincombOperators within the summands so we recurse ..
            op = self.apply(op)

        return op

    @match_class(Model, Operator)
    def action_recurse(self, op):
        return self.replace_children(op)


class ContractRules(RuleTable):
    """|RuleTable| for the :func:`contract` algorithm."""

    def __init__(self):
        super().__init__(use_caching=True)

    @match_class(LincombOperator)
    def action_LincombOperator(self, op):
        # recursively contract all children
        op = self.replace_children(op)

        # merge non-parametric part
        param_ops, param_coeffs = [], []
        non_param_ops, non_param_coeffs = [], []
        for o, c in zip(op.operators, op.coefficients, strict=True):
            if o.parametric or isinstance(c, ParameterFunctional):
                param_ops.append(o)
                param_coeffs.append(c)
            else:
                non_param_ops.append(o)
                non_param_coeffs.append(c)

        if not non_param_ops:  # there is not non-parametric part at all
            return op

        non_parametric_part = LincombOperator(non_param_ops, non_param_coeffs).assemble()

        if not param_ops:  # there is no parametric part at all
            return non_parametric_part

        if isinstance(non_parametric_part, LincombOperator):
            if len(non_parametric_part.operators) == len(non_param_ops):  # nothing could be contracted
                return op
            else:
                param_ops.extend(non_parametric_part.operators)
                param_coeffs.extend(non_parametric_part.param_coeffs)
        else:
            param_ops.append(non_parametric_part)
            param_coeffs.append(1.)

        return op.with_(operators=param_ops, coefficients=param_coeffs)

    @match_class(ConcatenationOperator)
    def action_ConcatenationOperator(self, op):
        op = self.replace_children(op)

        ops_rev = list(op.operators[::-1])
        i = 0
        while i + 1 < len(ops_rev):
            if isinstance(ops_rev[i], IdentityOperator) and len(ops_rev) > 1:
                del ops_rev[i]
            elif (ops_rev[i + 1].linear and not ops_rev[i + 1].parametric):
                if isinstance(ops_rev[i], NumpyMatrixOperator):
                    if not ops_rev[i].sparse:  # do not touch sparse matrices
                        U = ops_rev[i + 1].source.from_numpy(ops_rev[i].matrix)
                        ops_rev[i + 1] = VectorArrayOperator(ops_rev[i + 1].apply(U))
                        del ops_rev[i]
                    else:
                        i += 1
                elif (ops_rev[i].linear and not ops_rev[i].parametric
                      and isinstance(ops_rev[i].source, NumpyVectorSpace)
                      and ops_rev[i].source.dim <= as_array_max_length()):
                    # the following might in fact convert small sparse matrices in external solvers
                    # to dense ...
                    U = ops_rev[i].as_range_array()
                    ops_rev[i + 1] = VectorArrayOperator(ops_rev[i + 1].apply(U))
                    del ops_rev[i]
                else:
                    i += 1
            else:
                i += 1

        if len(ops_rev) == 1:
            op = ops_rev[0]
            if isinstance(op, VectorArrayOperator) and isinstance(op.array.space, NumpyVectorSpace):
                array = op.array.to_numpy()
                op = NumpyMatrixOperator(array.conj().T if op.adjoint else array)
            return op

        op = op.with_(operators=ops_rev[::-1])

        return op

    @match_class(Model, Operator)
    def action_recurse(self, op):
        return self.replace_children(op)
