# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from numbers import Number

import numpy as np

from pymor.la import VectorArrayInterface, NumpyVectorArray
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.basic import OperatorBase, ProjectedOperator, ProjectedLinearOperator


def rb_project_operator(operator, rb, product=None, name=None):
    assert operator is None or isinstance(operator, OperatorInterface)

    if operator is None:
        return None
    if operator.dim_source > 0:
        assert operator.dim_source == rb.dim
        source_basis = rb
    else:
        source_basis = None
    if operator.dim_range > 1:
        assert operator.dim_range == rb.dim
        range_basis = rb
    else:
        range_basis = None

    return operator.projected(source_basis, range_basis, product=product, name=name)


class Concatenation(OperatorBase):

    def __init__(self, second, first, name=None):
        assert isinstance(second, OperatorInterface)
        assert isinstance(first, OperatorInterface)
        assert second.dim_source == first.dim_range
        assert second.type_source == first.type_range
        super(Concatenation, self).__init__()
        self.first = first
        self.second = second
        self.build_parameter_type(inherits=(second, first))
        self.dim_source = first.dim_source
        self.dim_range = second.dim_range
        self.type_source = first.type_source
        self.type_range = second.type_range
        self.linear = second.linear and first.linear
        self.name = name
        self.lock()

    def apply(self, U, ind=None, mu=None):
        mu = self.parse_parameter(mu)
        return self.second.apply(self.first.apply(U, ind=ind, mu=mu), mu=mu)


class ComponentProjection(OperatorBase):

    type_range = NumpyVectorArray

    def __init__(self, components, dim, type_source, name=None):
        assert all(0 <= c < dim for c in components)
        self.components = components
        self.dim_source = dim
        self.dim_range = len(components)
        self.type_source = type_source
        self.name = name
        self.lock()

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)
        assert isinstance(U, self.type_source)
        assert U.dim == self.dim_source
        return NumpyVectorArray(U.components(self.components, ind), copy=False)


class IdentityOperator(OperatorBase):

    def __init__(self, dim, type_source, name=None):
        assert issubclass(type_source, VectorArrayInterface)

        super(IdentityOperator, self).__init__()
        self.dim_range = self.dim_source = dim
        self.type_range = self.type_source = type_source
        self.name = name
        self.lock()

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)
        assert isinstance(U, self.type_source)
        assert U.dim == self.dim_source
        return U.copy(ind=ind)


class ConstantOperator(OperatorBase):

    type_source = NumpyVectorArray

    dim_source = 0

    def __init__(self, value, name=None):
        assert isinstance(value, VectorArrayInterface)
        assert len(value) == 1

        super(ConstantOperator, self).__init__()
        self.dim_range = value.dim
        self.type_range = type(value)
        self.name = name
        self._value = value.copy()
        self.lock()

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)
        assert isinstance(U, (NumpyVectorArray, Number))
        if isinstance(U, Number):
            assert U == 0.
            assert ind == None
            return self._value.copy()
        else:
            assert U.dim == 0
            if ind is not None:
                raise NotImplementedError
            return self._value.copy()

    def as_vector(self):
        '''Returns the image of the operator as a VectorArray of length 1.'''
        return self._value.copy()

    def __add__(self, other):
        if isinstance(other, ConstantOperator):
            return ConstantOperator(self._vector + other._vector)
        elif isinstance(other, Number):
            return ConstantOperator(self._vector + other)
        else:
            return NotImplemented

    __radd__ = __add__

    def __mul__(self, other):
        return ConstantOperator(self._vector * other)


class FixedParameterOperator(OperatorBase):

    def __init__(self, operator, mu=None):
        assert isinstance(operator, OperatorInterface)
        assert operator.check_parameter(mu)
        self.operator = operator
        self.mu = mu.copy()
        self.lock()

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)
        return self.operator.apply(U, self.mu)

    @property
    def invert_options(self):
        return self.operator.invert_options

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        self.operator.apply_inverse(U, ind=ind, mu=self.mu, options=options)
