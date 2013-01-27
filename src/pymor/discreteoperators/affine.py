from __future__ import absolute_import, division, print_function

from itertools import izip

from .interfaces import LinearDiscreteOperatorInterface


class LinearAffinelyDecomposedOperator(LinearDiscreteOperatorInterface):

    def __init__(self, operators, operator_affine_part=None, functionals=None, name=None):
        assert functionals is None or len(operators) == len(functionals),\
                ValueError('Operators and functionals must have the same length.')
        assert functionals is None or parameter_dim is not None,\
                ValueError('If functionals is specified, parameter_dim must also be specified.')

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

        self.operators = operators
        self.operator_affine_part = operator_affine_part
        self.functionals = functionals
        if functionals is not None:
            self.set_parameter_type(inherits={'operators':operators,
                                              'operator_affine_part':operator_affine_part,
                                             'functionals':functionals})
        else:
            self.set_parameter_type([('coefficients',len(operators))],
                                    inherits={'operators':operators,
                                              'operator_affine_part':operator_affine_part})
        self.name = name

    def assemble(self, mu):
        if self.functionals is not None:
            A = sum(op.matrix(self.map_parameter(mu, 'operators', n)) * f(self.map_parameter(mu, 'functionals', n))
                                    for n, op, f in izip(xrange(len(self.operators)), self.operators, self.functionals))
        else:
            my_mu = self.map_parameter(mu)
            A = sum(op.matrix(self.map_parameter(mu, 'operators', n)) * m
                                    for n, op, m in izip(xrange(len(self.operators)), self.operators, my_mu['coefficients']))

        if self.operator_affine_part is not None:
            A = A + self.operator_affine_part.matrix(self.map_parameter(mu, 'operator_affine_part'))

        return A
