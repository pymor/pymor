from __future__ import absolute_import, division, print_function, unicode_literals

from itertools import izip

from .interfaces import ILinearDiscreteOperator


class LinearAffinelyDecomposedDOP(ILinearDiscreteOperator):

    def __init__(self, operators, operator_affine_part=None, functionals=None, parameter_dim=None):
        assert functionals is None or len(operators) == len(functionals),\
                ValueError('Operators and functionals must have the same length.')
        assert functionals is None or parameter_dim is not None,\
                ValueError('If functionals is specified, parameter_dim must also be specified.')
        assert parameter_dim is None or functionals is not None,\
                ValueError('If parameter_dim is specified, functionals must also be specified.')
        self.parameter_dim = parameter_dim or len(operators)
        self.operators = operators
        self.operator_affine_part = operator_affine_part
        self.functionals = functionals

    def assemble(self, mu):
        assert mu.size == self.parameter_dim,\
                ValueError('Invalid parameter dimensions (was {}, expected {})'.format(mu.size, self.parameter_dim))

        if self.functionals is not None:
            A = (sum(op.matrix() * f(mu) for op, f in izip(self.operators, self.functionals))
                 + self.operator_affine_part.matrix())
        else:
            A = (sum(op.matrix() * m for op, m in izip(self.operators, mu))
                 + self.operator_affine_part.matrix())

        return A
