from __future__ import absolute_import, division, print_function

import numpy as np

from .interfaces import DiscreteOperatorInterface, LinearDiscreteOperatorInterface


class ProjectedOperator(DiscreteOperatorInterface):

    def __init__(self, operator, source_basis, range_basis=None, product=None, name=None):
        range_basis = range_basis or source_basis
        assert isinstance(operator, DiscreteOperatorInterface)
        assert operator.source_dim == source_basis.shape[1]
        assert operator.range_dim == range_basis.shape[1]
        self.set_parameter_type(inherits={'operator':operator})
        self.source_dim = source_basis.shape[0]
        self.range_dim = range_basis.shape[0]
        self.name = name
        self.operator = operator
        self.source_basis = source_basis
        self.range_basis = range_basis
        self.product = product

    def apply(self, U, mu={}):
        V = np.dot(U, self.source_basis)
        AV = self.operator.apply(V, self.map_parameter(mu, 'operator'))
        if self.product is None:
            return np.dot(AV, self.range_basis.T)
        elif isinstance(self.product, DiscreteOperatorInterface):
            return self.product.apply2(AV, self.range_basis, pairwise=False)
        else:
            return np.dot(np.dot(AV, self.product), self.range_basis.T)


class ProjectedLinearOperator(LinearDiscreteOperatorInterface):

    def __init__(self, operator, source_basis, range_basis=None, product=None, name=None):
        range_basis = range_basis or source_basis
        assert isinstance(operator, LinearDiscreteOperatorInterface)
        assert operator.source_dim == source_basis.shape[1]
        assert operator.range_dim == range_basis.shape[1]
        self.set_parameter_type(inherits={'operator':operator})
        self.source_dim = source_basis.shape[0]
        self.range_dim = range_basis.shape[0]
        self.name = name
        self.operator = operator
        self.source_basis = source_basis
        self.range_basis = range_basis
        self.product = product

    def assemble(self, mu={}):
        M = self.operator.matrix(self.map_parameter(mu, 'operator'))
        MB = M.dot(self.source_basis.T)
        if self.product is None:
            return np.dot(self.range_basis, MB)
        elif isinstance(self.product, DiscreteOperatorInterface):
            return self.product.apply2(self.range_basis, MB.T, pairwise=False)
        else:
            return np.dot(self.range_basis, np.dot(self.product, AV))

