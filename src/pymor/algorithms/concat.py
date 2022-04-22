from pymor.algorithms.rules import RuleTable, match_always, match_class_any
from pymor.core.exceptions import RuleNotMatchingError
from pymor.operators.constructions import ConcatenationOperator, IdentityOperator, ZeroOperator


def assemble_concat(operators, solver_options=None, name=None):
    return AssembleConcatRules(solver_options, name).apply(tuple(operators))


class AssembleConcatRules(RuleTable):
    def __init__(self, solver_options, name):
        super().__init__(use_caching=False)
        self.__auto_init(locals())

    @match_class_any(ZeroOperator)
    def action_ZeroOperator(self, ops):
        return ZeroOperator(ops[0].range, ops[-1].source, name=self.name)

    @match_class_any(IdentityOperator)
    def action_IdentityOperator(self, ops):
        without_identity = [op for op in ops if not isinstance(op, IdentityOperator)]
        if len(without_identity) == 0:
            return IdentityOperator(ops[0].range, name=self.name)
        else:
            return assemble_concat(without_identity, solver_options=self.solver_options, name=self.name)

    @match_always
    def action_call_assemble_concat_method(self, ops):
        op = ops[0]._assemble_concat(ops, solver_options=self.solver_options, name=self.name)
        if not op:
            raise RuleNotMatchingError
        else:
            return op

    @match_always
    def action_return_concat(self, ops):
        return ConcatenationOperator(ops, name=self.name)
