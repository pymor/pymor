# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from numbers import Number

from pymor.algorithms.lincomb import assemble_lincomb
from pymor.algorithms.rules import RuleTable, match_class, match_generic, match_class_all, match_class_any
from pymor.core.exceptions import RuleNotMatchingError, NoMatchingRuleError
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import IdentityOperator, Concatenation, ConstantOperator, LincombOperator
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorArray


def simplify(obj, assemble=True):
    """Simplification of various constructs, e.g. concatenation of |Operator|s.

    How the simplification is realized will depend on the given object,
    the exact algorithm is specified in :class:`SimplifyRules`.

    Parameters
    ----------
    obj
        The object to be simplified (only |Operator|s supported at the moment).

    Returns
    -------
    The simplified object, if applicable, else the original object.
    """

    try:
        return SimplifyRules(assemble).apply(obj)
    except NoMatchingRuleError:
        obj.logger.warning(f'could not be simplified')
        return obj


class SimplifyConcatenationOfTwoNonLincombOps(RuleTable):

    def __init__(self, assemble=True):
        super().__init__(use_caching=True)
        self.__auto_init(locals())

    @match_class_any(IdentityOperator)
    def action_IdentityOperator(self, tpl):
        if len(tpl) != 2:
            raise RuleNotMatchingError
        second, first = tpl
        if isinstance(second, IdentityOperator):
            return first
        else:
            return second

    @match_generic(lambda tpl: isinstance(tpl[0], ConstantOperator), 'ConstantOperator last')
    def action_ConstantOperator_last(self, tpl):
        if len(tpl) != 2:
            raise RuleNotMatchingError
        return tpl[0]

    @match_generic(lambda tpl: not tpl[0].parametric and isinstance(tpl[1], ConstantOperator) and not tpl[1].parametric,
                   'ConstantOperator first')
    def action_nonparametric_ConstantOperator_first(self, tpl):
        if len(tpl) != 2:
            raise RuleNotMatchingError
        second, first = tpl
        return ConstantOperator(self.apply((second, first.value)), source=first.source)

    @match_generic(lambda tpl: isinstance(tpl[0], NumpyMatrixOperator) and isinstance(tpl[1], NumpyVectorArray),
                   'NumpyMatrixOperator applied to NumpyVectorArray')
    def action_NumpyMatrixOperator_applied_to_NumpyVectorArray(self, tpl):
        if len(tpl) != 2:
            raise RuleNotMatchingError
        second, first = tpl
        return second.apply(first)

    @match_class_all(NumpyMatrixOperator)
    def action_NumpyMatrixOperator(self, tpl):
        if len(tpl) != 2:
            raise RuleNotMatchingError
        second, first = tpl
        if second.source != first.range:
            raise RuleNotMatchingError(f'second.source = {second.source} != {first.range} = first.range')
        if second.solver_options and not first.solver_options:
            solver_options = second.solver_options
        elif not second.solver_options and first.solver_options:
            solver_options = first.solver_options
        elif second.solver_options and first.solver_options and (second.solver_options == first.solver_options):
            solver_options = first.solver_options
        else:
            solver_options = None
        return NumpyMatrixOperator(second.matrix@first.matrix, source_id=first.source_id, range_id=second.range_id,
                solver_options=solver_options, name=f'product of {second.name} and {first.name}')


class SimplifyRules(RuleTable):
    """|RuleTable| for the :func:`simplify` algorithm."""

    def __init__(self, assemble=True):
        super().__init__(use_caching=True)
        self._simplify_concat = SimplifyConcatenationOfTwoNonLincombOps(assemble)
        self.__auto_init(locals())

    @match_class(StationaryModel)
    def action_StationaryModel(self, model):
        return model.with_(
                operator=self.apply(model.operator),
                rhs=self.apply(model.rhs),
                output_functional=self.apply(model.output_functional),
                products={name: self.apply(prod) for name, prod in model.products.items()})

    @match_class(NumpyMatrixOperator)
    def action_NumpyMatrixOperator(self, op):
        return op

    @match_class(Concatenation)
    def action_Concatenation(self, cop):
        if len(cop.operators) == 1:
            return self.apply(cop.operators[0])
        last, rest = cop.operators[0], cop.operators[1:]
        if len(rest) > 1:
            rest = self.apply(Concatenation(rest)).operators
            if len(rest) > 1:
                raise RuleNotMatchingError('Could not simplify remaining parts of a Concatenation.')
        first = rest[0]
        last = self.apply(last)
        first = self.apply(first)
        if isinstance(last, LincombOperator):
            ops = [self.apply(Concatenation((op, first))) for op in last.operators]
            coeffs = last.coefficients
            return self.apply(LincombOperator(ops, coeffs, name=cop.name))
        elif last.linear and isinstance(first, LincombOperator):
            ops = [self.apply(Concatenation((last, op))) for op in first.operators]
            coeffs = first.coefficients
            return self.apply(LincombOperator(ops, coeffs, name=cop.name))
        else:
            return self._simplify_concat.apply((last, first))

    @match_class(ConstantOperator)
    def action_ConstantOperator(self, op):
        return op

    @match_class(IdentityOperator)
    def action_IdentityOperator(self, op):
        return op

    @match_class(LincombOperator)
    def action_LincombOperator(self, lop):
        lop = self.replace_children(lop)

        def flatten(op, coeff):
            if not isinstance(op, LincombOperator):
                return [op,], [coeff,]
            ops, coeffs = [], []
            for oo, cc in zip(op.operators, op.coefficients):
                oo, cc = flatten(oo, cc)
                ops.extend(oo)
                coeffs.extend([c if coeff == 1 else coeff*c for c in cc])
            return ops, coeffs

        lops, lcoeffs = flatten(lop, 1.)

        # sort ops by coefficient
        nonparametric_part = ([], [])
        parametric_part = {}
        for op, coeff in zip(lops, lcoeffs):
            if isinstance(coeff, Number):
                nonparametric_part[0].append(coeff)
                nonparametric_part[1].append(op)
            elif not coeff.parametric:
                nonparametric_part[0].append(coeff.evaluate())
                nonparametric_part[1].append(op)
            else:
                if coeff.parameter_type in parametric_part:
                    parametric_part[coeff.parameter_type][0].append(coeff)
                    parametric_part[coeff.parameter_type][1].append(op)
                else:
                    parametric_part[coeff.parameter_type] = ([coeff,], [op,])

        # nonparametric_part
        if len(nonparametric_part[0]) > 0:
            if self.assemble:
                op = assemble_lincomb(nonparametric_part[1], nonparametric_part[0])
                if op and len(parametric_part) == 0:
                    return op
                elif op:
                    ops = [op,]
                    coeffs = [1.,]
                else:
                    ops = nonparametric_part[1]
                    coeffs = nonparametric_part[0]
            else: 
                ops = nonparametric_part[1]
                coeffs = nonparametric_part[0]
        else:
            ops, coeffs = [], []
        
        # parametric part
        for _, (ccs, oos) in parametric_part.items():
            op = assemble_lincomb(oos, np.ones(len(oos)))
            if not op:
                a = b
                raise RuleNotMatchingError('Could not assemble parametric contributions of LincombOperator.')
            coeff = ccs[0]
            for i in range(1, len(ccs)):
                coeff += ccs[i]
            ops.append(op)
            coeffs.append(coeff)

        return LincombOperator(ops, coeffs, solver_options=lop.solver_options, name=lop.name)

    @match_generic(lambda op: isinstance(op, OperatorInterface) and op.linear and not op.parametric,
                   'linear and nonparametric operator')
    def action_linear_nonparametric_operator(self, op):
        if self.assemble:
            return op.assemble()
        else:
            return op

