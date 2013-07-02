# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip
import numpy as np

from pymor.operators.interfaces import LinearOperatorInterface


class LinearAffinelyDecomposedOperator(LinearOperatorInterface):
    '''Affine combination of given linear operators.

    Given operators L_k and functionals θ_k, this operator represents ::

      |                K
      |  L =  L_0  +   ∑  θ_k ⋅ L_k
      |               k=1

    Parameters
    ----------
    operators
        List of the discrete linear operators L_1, ..., L_K.
    operator_affine_part
        The discrete linear operator L_0.
    functionals
        If not None, list of the functionals θ_1, ..., θ_K.
        If None, `.coefficients` is added to the `parameter_type` of the
        operator with shape (K,) and θ_k(μ) is defined to be
        `mu.coefficients[k-1]`.
    name
        Name of the operator.
    '''

    def __init__(self, operators, operator_affine_part=None, functionals=None, name_map=None, name=None):
        assert functionals is None or len(operators) == len(functionals), \
            ValueError('Operators and functionals must have the same length.')

        super(LinearAffinelyDecomposedOperator, self).__init__()

        if operator_affine_part is not None:
            self.dim_source = operator_affine_part.dim_source
            self.dim_range = operator_affine_part.dim_range
            self.type_source = operator_affine_part.type_source
            self.type_range = operator_affine_part.type_range
        else:
            self.dim_source = operators[0].dim_source
            self.dim_range = operators[0].dim_range
            self.type_source = operators[0].type_source
            self.type_range = operators[0].type_range

        assert all(op.dim_source == self.dim_source for op in operators), \
            ValueError('All operators must have the same source dimension.')
        assert all(op.dim_range == self.dim_range for op in operators), \
            ValueError('All operators must have the same range dimension.')
        assert all(op.type_source == self.type_source for op in operators), \
            ValueError('All operators must have the same source type.')
        assert all(op.type_range == self.type_range for op in operators), \
            ValueError('All operators must have the same range type.')

        self.operators = operators
        self.operator_affine_part = operator_affine_part
        self.functionals = functionals
        if functionals is not None:
            self.build_parameter_type(inherits={'operators': operators,
                                                'operator_affine_part': operator_affine_part,
                                                'functionals': functionals},
                                      name_map=name_map)
        else:
            self.build_parameter_type([('coefficients', len(operators))],
                                      inherits={'operators': operators,
                                                'operator_affine_part': operator_affine_part},
                                      name_map=name_map)
        self.name = name
        self.lock()

    def _assemble(self, mu):
        if self.functionals is not None:
            A = sum(op.assemble(self.map_parameter(mu, 'operators', n)) * f(self.map_parameter(mu, 'functionals', n))
                    for n, op, f in izip(xrange(len(self.operators)), self.operators, self.functionals))
        else:
            my_mu = self.map_parameter(mu)
            A = sum(op.assemble(self.map_parameter(mu, 'operators', n)) * m
                    for n, op, m in izip(xrange(len(self.operators)), self.operators, my_mu['coefficients']))

        if self.operator_affine_part is not None:
            A = A + self.operator_affine_part.assemble(self.map_parameter(mu, 'operator_affine_part'))

        return A

    def evaluate_coefficients(self, mu):
        '''Returns [θ_1(mu), ..., θ_K(mu)].'''
        if self.functionals is not None:
            return np.array(tuple(f(self.map_parameter(mu, 'functionals', n)) for n, f in enumerate(self.functionals)))
        else:
            return self.map_parameter(mu)['coefficients']
