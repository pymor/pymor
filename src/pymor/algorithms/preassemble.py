# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.rules import RuleTable, rule
from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.operators.basic import ProjectedOperator
from pymor.operators.constructions import (LincombOperator, Concatenation,
                                           AffineOperator, AdjointOperator, SelectionOperator)
from pymor.operators.interfaces import OperatorInterface


def preassemble(obj):
    return PreAssembleRules.apply(obj)


class PreAssembleRules(RuleTable):

    @rule((LincombOperator, SelectionOperator))
    def LincombOrSeclectionOperator(self, op, *args, **kwargs):
        """replace sub-operators"""
        new_operators = [self.apply(op, *args, **kwargs) for op in op.operators]
        if any(o_new is not o_old for o_new, o_old in zip(new_operators, op.operators)):
            op = op.with_(operators=new_operators)
        if not op.parametric:
            op = op.assemble()
        return op

    @rule(Concatenation)
    def Concatenation(self, op, *args, **kwargs):
        """replace sub-operators"""
        new_first = self.apply(op.first, *args, **kwargs)
        new_second = self.apply(op.second, *args, **kwargs)
        return op.with_(first=new_first, second=new_second)

    @rule(AffineOperator)
    def AffineOperator(self, op, *args, **kwargs):
        """replace sub-operators"""
        new_affine_shift = self.apply(op.affine_shift, *args, **kwargs)
        new_linear_part = self.apply(op.linear_part, *args, **kwargs)
        return op.with_(affine_shift=new_affine_shift, linear_part=new_linear_part)

    @rule((AdjointOperator, ProjectedOperator))
    def AdjointOperator(self, op, *args, **kwargs):
        """replace sub-operators"""
        new_operator = self.apply(op.operator, *args, **kwargs)
        if new_operator is op.operator:
            return op
        elif not (op.source_product or op.range_product):
            return new_operator.T
        else:
            return op.with_(operator=new_operator)

    @rule(lambda op: not op.parametric, 'non-parametric operator')
    def non_parametric(self, op):
        """replace with assembled operator"""
        return op.assemble()

    @rule(OperatorInterface)
    def identity(self, op, *args, **kwargs):
        """do nothing"""
        return op

    @rule(DiscretizationInterface)
    def Discretization(self, d, *args, **kwargs):
        """replace operators"""
        new_operators = {k: self.apply(v, *args, **kwargs) if v else v for k, v in d.operators.items()}
        new_products = {k: self.apply(v, *args, **kwargs) if v else v for k, v in d.products.items()}
        return d.with_(operators=new_operators, products=new_products)
