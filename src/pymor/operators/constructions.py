# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from numbers import Number

import numpy as np

from pymor.la import VectorArrayInterface, NumpyVectorArray
from pymor.operators.basic import OperatorBase, ProjectedOperator, ProjectedLinearOperator
from pymor.operators.interfaces import OperatorInterface


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
        if hasattr(first, 'restricted') and hasattr(second, 'restricted'):
            self.restricted = self._restricted
        self.name = name

    def apply(self, U, ind=None, mu=None):
        mu = self.parse_parameter(mu)
        return self.second.apply(self.first.apply(U, ind=ind, mu=mu), mu=mu)

    def _restricted(self, components):
        restricted_second, second_source_components = self.second.restricted(components)
        restricted_first, first_source_components = self.first.restricted(second_source_components)
        if isinstance(restricted_second, IdentityOperator):
            return restricted_first, first_source_components
        elif isinstance(restricted_first, IdentityOperator):
            return restricted_second, first_source_components
        else:
            return Concatenation(restricted_second, restricted_first), first_source_components


class ComponentProjection(OperatorBase):

    type_range = NumpyVectorArray

    def __init__(self, components, dim, type_source, name=None):
        assert all(0 <= c < dim for c in components)
        self.components = np.array(components)
        self.dim_source = dim
        self.dim_range = len(components)
        self.type_source = type_source
        self.name = name

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)
        assert isinstance(U, self.type_source)
        assert U.dim == self.dim_source
        return NumpyVectorArray(U.components(self.components, ind), copy=False)

    def restricted(self, components):
        assert all(0 <= c < self.dim_range for c in components)
        source_components = self.components[components]
        return IdentityOperator(dim=len(source_components), type_source=self.type_source), source_components


class IdentityOperator(OperatorBase):

    def __init__(self, dim, type_source, name=None):
        assert issubclass(type_source, VectorArrayInterface)

        super(IdentityOperator, self).__init__()
        self.dim_range = self.dim_source = dim
        self.type_range = self.type_source = type_source
        self.name = name

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)
        assert isinstance(U, self.type_source)
        assert U.dim == self.dim_source
        return U.copy(ind=ind)


class ConstantOperator(OperatorBase):

    linear = False

    def __init__(self, value, dim_source, type_source=NumpyVectorArray, copy=True, name=None):
        assert isinstance(value, VectorArrayInterface)
        assert len(value) == 1
        super(ConstantOperator, self).__init__()
        self.dim_source = dim_source
        self.dim_range = value.dim
        self.type_source = type_source
        self.type_range = type(value)
        self.name = name
        self._value = value.copy() if copy else value

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)
        assert isinstance(U, self.type_source) and U.dim == self.dim_source
        count = len(U) if ind is None else 1 if isinstance(ind, Number) else len(ind)
        return self._value.copy(ind=([0] * count))


class VectorOperator(OperatorBase):

    linear = True
    type_source = NumpyVectorArray
    dim_source = 1

    def __init__(self, vector, copy=True, name=None):
        assert isinstance(vector, VectorArrayInterface)
        assert len(vector) == 1
        super(VectorOperator, self).__init__()
        self.dim_range = vector.dim
        self.type_range = type(vector)
        self.name = name
        self._vector = vector.copy() if copy else vector

    def as_vector(self, mu=None):
        return self._vector.copy()

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)
        assert isinstance(U, NumpyVectorArray) and U.dim == 1
        count = len(U) if ind is None else 1 if isinstance(ind, Number) else len(ind)
        R = self._vector.copy(ind=([0] * count))
        for i, c in enumerate(U.data):
            R.scal(c[0], ind=i)
        return R


class VectorFunctional(OperatorBase):

    linear = True
    type_range = NumpyVectorArray
    dim_range = 1

    def __init__(self, vector, product=None, copy=True, name=None):
        assert isinstance(vector, VectorArrayInterface)
        assert len(vector) == 1
        assert product is None or isinstance(product, OperatorInterface)
        super(VectorFunctional, self).__init__()
        self.dim_source = vector.dim
        self.type_source = type(vector)
        self.name = name
        if product is None:
            self._vector = vector.copy() if copy else vector
        else:
            self._vector = product.apply(vector)

    def as_vector(self, mu=None):
        return self._vector.copy()

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)
        assert isinstance(U, NumpyVectorArray) and U.dim == 1
        return NumpyVectorArray(U.dot(self._vector, ind=ind), copy=False)


class FixedParameterOperator(OperatorBase):

    def __init__(self, operator, mu=None):
        assert isinstance(operator, OperatorInterface)
        assert operator.check_parameter(mu)
        self.operator = operator
        self.mu = mu.copy()

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)
        return self.operator.apply(U, self.mu)

    @property
    def invert_options(self):
        return self.operator.invert_options

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        self.operator.apply_inverse(U, ind=ind, mu=self.mu, options=options)
