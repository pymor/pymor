# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.rules import RuleTable, match_class, match_generic
from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.operators.basic import ProjectedOperator
from pymor.operators.constructions import (LincombOperator, Concatenation,
                                           AffineOperator, AdjointOperator, SelectionOperator)
from pymor.operators.interfaces import OperatorInterface


def preassemble(obj):
    return PreAssembleRules.apply(obj)


class PreAssembleRules(RuleTable):

    @match_class(DiscretizationInterface)
    def action_Discretization(self, d, *args, **kwargs):
        new_operators = {k: self.apply(v, *args, **kwargs) if v else v for k, v in d.operators.items()}
        new_products = {k: self.apply(v, *args, **kwargs) if v else v for k, v in d.products.items()}
        return d.with_(operators=new_operators, products=new_products)

    @match_class(LincombOperator, SelectionOperator)
    def action_LincombOrSeclectionOperator(self, op, *args, **kwargs):
        new_operators = [self.apply(o, *args, **kwargs) for o in op.operators]
        if any(o_new is not o_old for o_new, o_old in zip(new_operators, op.operators)):
            op = op.with_(operators=new_operators)
        if not op.parametric:
            op = op.assemble()
        return op

    @match_class(Concatenation)
    def action_Concatenation(self, op, *args, **kwargs):
        new_first = self.apply(op.first, *args, **kwargs)
        new_second = self.apply(op.second, *args, **kwargs)
        return op.with_(first=new_first, second=new_second)

    @match_class(AffineOperator)
    def action_AffineOperator(self, op, *args, **kwargs):
        new_affine_shift = self.apply(op.affine_shift, *args, **kwargs)
        new_linear_part = self.apply(op.linear_part, *args, **kwargs)
        return op.with_(affine_shift=new_affine_shift, linear_part=new_linear_part)

    @match_class(AdjointOperator, ProjectedOperator)
    def action_AdjointOperator(self, op, *args, **kwargs):
        new_operator = self.apply(op.operator, *args, **kwargs)
        if new_operator is op.operator:
            return op
        elif not (op.source_product or op.range_product):
            return new_operator.T
        else:
            return op.with_(operator=new_operator)

    @match_generic(lambda op: not op.parametric, 'non-parametric operator')
    def action_assemble(self, op):
        return op.assemble()

    @match_class(OperatorInterface)
    def action_identity(self, op):
        return op
