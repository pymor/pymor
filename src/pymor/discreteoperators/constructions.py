from __future__ import absolute_import, division, print_function

import numpy as np

from .interfaces import DiscreteOperatorInterface, LinearDiscreteOperatorInterface
from .affine import LinearAffinelyDecomposedOperator
from .basic import GenericLinearOperator


class ProjectedOperator(DiscreteOperatorInterface):

    def __init__(self, operator, source_basis, range_basis=None, product=None, name=None):
        if range_basis is None:
            range_basis = np.ones((1,1)) if operator.range_dim == 1 else source_basis
        assert isinstance(operator, DiscreteOperatorInterface)
        assert operator.source_dim == source_basis.shape[1]
        assert operator.range_dim == range_basis.shape[1]
        self.build_parameter_type(inherits={'operator':operator})
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
        if range_basis is None:
            range_basis = np.ones((1,1)) if operator.range_dim == 1 else source_basis
        assert isinstance(operator, LinearDiscreteOperatorInterface)
        assert operator.source_dim == source_basis.shape[1]
        assert operator.range_dim == range_basis.shape[1]
        self.build_parameter_type(inherits={'operator':operator})
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


def project_operator(operator, source_basis, range_basis=None, product=None, name=None):

    name = name or '{}_projected'.format(operator.name)

    if isinstance(operator, LinearAffinelyDecomposedOperator):
        proj_operators = tuple(project_operator(op, source_basis, range_basis, product,
                                                name='{}_projected'.format(op.name))
                                                                        for op in operator.operators)
        if operator.operator_affine_part is not None:
            proj_operator_ap = project_operator(operator.operator_affine_part, source_basis, range_basis, product,
                                                name='{}_projected'.format(operator.operator_affine_part.name))
        else:
            proj_operator_ap = None
        return LinearAffinelyDecomposedOperator(proj_operators, proj_operator_ap, operator.functionals, name)

    elif isinstance(operator, LinearDiscreteOperatorInterface):
        proj_operator = ProjectedLinearOperator(operator, source_basis, range_basis, product, name)
        if proj_operator.parameter_type == {}:
            return GenericLinearOperator(proj_operator.matrix(), name)
        else:
            return proj_operator

    else:
        return ProjectedOperator(operator, source_basis, range_basis, product, name)
