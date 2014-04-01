# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Module containing some constructions to obtain new operators from old ones.'''

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np

from pymor.la import VectorArrayInterface, NumpyVectorArray
from pymor.operators.basic import OperatorBase
from pymor.operators.interfaces import OperatorInterface


class Concatenation(OperatorBase):
    '''|Operator| representing the concatenation of two |Operators|.

    Parameters
    ----------
    second
        The |Operator| which is applied as second operator.
    first
        The |Operator| which is applied as first operator.
    name
        Name of the operator.
    '''

    def __init__(self, second, first, name=None):
        assert isinstance(second, OperatorInterface)
        assert isinstance(first, OperatorInterface)
        assert second.dim_source == first.dim_range
        assert second.type_source == first.type_range
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
    '''|Operator| representing the projection of a Vector on some of its components.

    Parameters
    ----------
    components
        List or 1D |NumPy array| of the indices of the vector components that are to
        be extracted by the operator.
    dim_source
        Source dimension of the operator.
    type_source
        The type of |VectorArray| the operator accepts.
    name
        Name of the operator.
    '''

    type_range = NumpyVectorArray
    linear = True

    def __init__(self, components, dim_source, type_source, name=None):
        assert all(0 <= c < dim_source for c in components)
        self.components = np.array(components)
        self.dim_source = dim_source
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
    '''The identity |Operator|.

    In other word ::

        op.apply(U) == U

    Parameters
    ----------
    dim
        Source dimension (= range dimension) of the operator.
    type_source
        The type of |VectorArray| the operator accepts.
    name
        Name of the operator.
    '''

    linear = True

    def __init__(self, dim, type_source, name=None):
        assert issubclass(type_source, VectorArrayInterface)

        self.dim_range = self.dim_source = dim
        self.type_range = self.type_source = type_source
        self.name = name

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)
        assert isinstance(U, self.type_source)
        assert U.dim == self.dim_source
        return U.copy(ind=ind)


class ConstantOperator(OperatorBase):
    '''A constant |Operator| always returning the same vector.

    Parameters
    ----------
    value
        A |VectorArray| of length 1 containing the vector which is
        returned by the operator.
    dim_source
        Source dimension of the operator.
    type_source
        The type of |VectorArray| the operator accepts.
    copy
        If `True`, store a copy of `vector` instead of `vector`
        itself.
    name
        Name of the operator.
    '''

    linear = False

    def __init__(self, value, dim_source, type_source=NumpyVectorArray, copy=True, name=None):
        assert isinstance(value, VectorArrayInterface)
        assert len(value) == 1
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
    '''Wrap a vector as a vector-like |Operator|.

    Given a vector `v` of dimension `d`, this class represents
    the operator ::

        op: R^1 ----> R^d
             x  |---> x⋅v

    In particular ::

        VectorOperator(vector).as_vector() == vector

    Parameters
    ----------
    vector
        |VectorArray| of length 1 containing the vector `v`.
    copy
        If `True`, store a copy of `vector` instead of `vector`
        itself.
    name
        Name of the operator.
    '''

    linear = True
    type_source = NumpyVectorArray
    dim_source = 1

    def __init__(self, vector, copy=True, name=None):
        assert isinstance(vector, VectorArrayInterface)
        assert len(vector) == 1
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
    '''Wrap a vector as a linear |Functional|.

    Given a vector `v` of dimension `d`, this class represents
    the functional ::

        f: R^d ----> R^1
            u  |---> (u, v)

    where `( , )` denotes the scalar product given by `product`.

    In particular, if `product` is `None` ::

        VectorFunctional(vector).as_vector() == vector.

    If `product` is not none, we obtain ::

        VectorFunctional(vector).as_vector() == product.apply(vector).

    Parameters
    ----------
    vector
        |VectorArray| of length 1 containing the vector `v`.
    product
        |Operator| representing the scalar product to use.
    copy
        If `True`, store a copy of `vector` instead of `vector`
        itself.
    name
        Name of the operator.
    '''

    linear = True
    type_range = NumpyVectorArray
    dim_range = 1

    def __init__(self, vector, product=None, copy=True, name=None):
        assert isinstance(vector, VectorArrayInterface)
        assert len(vector) == 1
        assert product is None or isinstance(product, OperatorInterface)
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
        assert isinstance(U, self.type_source) and U.dim == self.dim_source
        return NumpyVectorArray(U.dot(self._vector, ind=ind, pairwise=False), copy=False)


class FixedParameterOperator(OperatorBase):
    '''Makes an |Operator| |Parameter|-independent by providing it a fixed |Parameter|.

    Parameters
    ----------
    operator
        The |Operator| to wrap.
    mu
        The fixed |Parameter| that will be fed to the
        :meth:`~pymor.operators.interfaces.OperatorInterface.apply` method
        of `operator`.
    '''

    def __init__(self, operator, mu=None):
        assert isinstance(operator, OperatorInterface)
        assert operator.check_parameter(mu)
        self.operator = operator
        self.mu = mu.copy()
        self.linear = operator.linear

    def apply(self, U, ind=None, mu=None):
        assert self.check_parameter(mu)
        return self.operator.apply(U, self.mu)

    @property
    def invert_options(self):
        return self.operator.invert_options

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        self.operator.apply_inverse(U, ind=ind, mu=self.mu, options=options)
