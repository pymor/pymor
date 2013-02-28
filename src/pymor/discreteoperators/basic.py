from __future__ import absolute_import, division, print_function

import numpy as np

from .interfaces import DiscreteOperatorInterface, LinearDiscreteOperatorInterface


class GenericOperator(DiscreteOperatorInterface):
    '''Wraps an apply function as a proper discrete operator.

    Parameters
    ----------
    mapping
        The function to wrap. If parameter_type is None, mapping is called with
        the DOF-vector U as only argument. If parameter_type is not None, mapping
        is called with the arguments U and mu.
    dim_source
        Dimension of the operator's source.
    dim_range
        Dimension of the operator's range.
    parameter_type
        Type of the parameter that mapping expects or None.
    name
        Name of the operator.

    Inherits
    --------
    DiscreteOperatorInterface
    '''

    def __init__(self, mapping, dim_source=1, dim_range=1, parameter_type=None, name=None):
        self.dim_source = dim_source
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
        assert U.shape[-1] == self.dim_source
        if self._with_mu:
            return self._mapping(U, mu)
        else:
            return self._mapping(U)


class GenericLinearOperator(LinearDiscreteOperatorInterface):
    '''Wraps a matrix as a proper linear discrete operator.

    The resulting operator will be parameter independent.

    Parameters
    ----------
    matrix
        The matrix which is to be wrapped.
    name
        Name of the operator.

    Inherits
    --------
    LinearDiscreteOperatorInterface
    '''

    def __init__(self, matrix, name=None):
        self.dim_source = matrix.shape[1]
        self.dim_range = matrix.shape[0]
        self.name = name
        self._matrix = matrix

    def assemble(self, mu={}):
        mu = self.parse_parameter(mu)
        return self._matrix
