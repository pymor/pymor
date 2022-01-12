# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.rules import RuleTable, match_class, match_generic
from pymor.models.interface import Model
from pymor.operators.constructions import (LincombOperator, ConcatenationOperator, ProjectedOperator,
                                           AffineOperator, AdjointOperator, SelectionOperator)
from pymor.operators.interface import Operator


def preassemble(obj):
    """Preassemble non-parametric operators.

    If `obj` is a non-parametric |Operator|, return
    `obj.assemble()` otherwise return `obj`. Recursively
    replaces children of `obj`.
    """
    return PreAssembleRules().apply(obj)


class PreAssembleRules(RuleTable):

    def __init__(self):
        super().__init__(use_caching=True)

    @match_class(Model, AffineOperator, ConcatenationOperator, SelectionOperator)
    def action_recurse(self, op):
        return self.replace_children(op)

    @match_class(LincombOperator)
    def action_recurse_and_assemble(self, op):
        op = self.replace_children(op)
        if not op.parametric:
            return op.assemble()
        else:
            return op

    @match_class(AdjointOperator, ProjectedOperator)
    def action_AdjointOperator(self, op):
        new_operator = self.apply(op.operator)
        if new_operator is op.operator:
            return op
        elif not (op.source_product or op.range_product):
            return new_operator.H
        else:
            return op.with_(operator=new_operator)

    @match_generic(lambda op: not op.parametric, 'non-parametric operator')
    def action_assemble(self, op):
        return op.assemble()

    @match_class(Operator)
    def action_identity(self, op):
        return op
