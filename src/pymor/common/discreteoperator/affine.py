from __future__ import absolute_import, division, print_function, unicode_literals

from itertools import izip

from .interfaces import LinearDiscreteOperatorInterface


class LinearAffinelyDecomposedDOP(LinearDiscreteOperatorInterface):

    def __init__(self, operators, operator_affine_part=None, functionals=None, parameter_dim=None, name=None):
        assert functionals is None or len(operators) == len(functionals),\
                ValueError('Operators and functionals must have the same length.')
        assert functionals is None or parameter_dim is not None,\
                ValueError('If functionals is specified, parameter_dim must also be specified.')
        assert parameter_dim is None or functionals is not None,\
                ValueError('If parameter_dim is specified, functionals must also be specified.')

        if operator_affine_part is not None:
            self.source_dim = operator_affine_part.source_dim
            self.range_dim = operator_affine_part.range_dim
        else:
            self.source_dim = operators[0].source_dim
            self.range_dim = operators[0].range_dimj

        assert all(op.source_dim == self.source_dim for op in operators),\
                ValueError('All operators must have the same source dimension.')
        assert all(op.range_dim == self.range_dim for op in operators),\
                ValueError('All operators must have the same range dimension.')

        self.parameter_dim = parameter_dim or len(operators)
        self.operators = operators
        self.operator_affine_part = operator_affine_part
        self.functionals = functionals
        self.name = name

    def assemble(self, mu):
        assert mu.size == self.parameter_dim,\
                ValueError('Invalid parameter dimensions (was {}, expected {})'.format(mu.size, self.parameter_dim))

        if self.functionals is not None:
            A = sum(op.matrix() * f(mu) for op, f in izip(self.operators, self.functionals))
        else:
            A = sum(op.matrix() * m for op, m in izip(self.operators, mu))

        if self.operator_affine_part is not None:
            A = A + self.operator_affine_part.matrix()

        return A
