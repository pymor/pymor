from __future__ import absolute_import, division, print_function

from itertools import izip

import numpy as np

from .interfaces import LinearDiscreteOperatorInterface


class LinearAffinelyDecomposedOperator(LinearDiscreteOperatorInterface):

    def __init__(self, operators, operator_affine_part=None, functionals=None, name=None):
        assert functionals is None or len(operators) == len(functionals),\
                ValueError('Operators and functionals must have the same length.')

        if operator_affine_part is not None:
            self.dim_source = operator_affine_part.dim_source
            self.dim_range = operator_affine_part.dim_range
        else:
            self.dim_source = operators[0].dim_source
            self.dim_range = operators[0].dim_range

        assert all(op.dim_source == self.dim_source for op in operators),\
                ValueError('All operators must have the same source dimension.')
        assert all(op.dim_range == self.dim_range for op in operators),\
                ValueError('All operators must have the same range dimension.')

        self.operators = operators
        self.operator_affine_part = operator_affine_part
        self.functionals = functionals
        if functionals is not None:
            self.build_parameter_type(inherits={'operators':operators,
                                              'operator_affine_part':operator_affine_part,
                                             'functionals':functionals})
        else:
            self.build_parameter_type([('coefficients',len(operators))],
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

    def evaluate_coefficients(self, mu):
        if self.functionals is not None:
            return np.array(tuple(f(self.map_parameter(mu, 'functionals', n)) for n, f in enumerate(self.functionals)))
        else:
            return self.map_parameter(mu)['coefficients']
