from __future__ import absolute_import, division, print_function

import numpy as np

from .interfaces import DiscreteOperatorInterface, LinearDiscreteOperatorInterface


class GenericOperator(DiscreteOperatorInterface):

    def __init__(self, mapping, dim_domain=1, dim_range=1, parameter_type=None, name=None):
        self.dim_domain = dim_domain
        self.dim_range = dim_range
        self.name = name
        self._mapping = mapping
        if parameter_type is not None:
            self.build_parameter_type(parameter_type)
            self._with_mu = True
        else:
            self._with_mu = False

    def apply(self, U, mu={}):
        mu = self.map_parameter(mu)
        assert U.shape[-1] == self.dim_domain
        if self._with_mu:
            return self._mapping(U, mu)
        else:
            return self._mapping(U)


class GenericLinearOperator(LinearDiscreteOperatorInterface):

    def __init__(self, matrix, name=None):
        self.dim_domain = matrix.shape[1]
        self.dim_range = matrix.shape[0]
        self.name = name
        self._matrix = matrix

    def assemble(self, mu={}):
        mu = self.parse_parameter(mu)
        return self._matrix
