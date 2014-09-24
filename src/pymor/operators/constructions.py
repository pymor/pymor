# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Module containing some constructions to obtain new operators from old ones."""

from __future__ import absolute_import, division, print_function

from numbers import Number
from itertools import izip

import numpy as np

from pymor.la import VectorArrayInterface, NumpyVectorArray, NumpyVectorSpace
from pymor.operators.basic import OperatorBase
from pymor.operators.interfaces import OperatorInterface


class Concatenation(OperatorBase):
    """|Operator| representing the concatenation of two |Operators|.

    Parameters
    ----------
    second
        The |Operator| which is applied as second operator.
    first
        The |Operator| which is applied as first operator.
    name
        Name of the operator.
    """

    def __init__(self, second, first, name=None):
        assert isinstance(second, OperatorInterface)
        assert isinstance(first, OperatorInterface)
        assert first.range == second.source
        self.first = first
        self.second = second
        self.build_parameter_type(inherits=(second, first))
        self.source = first.source
        self.range = second.range
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
    """|Operator| representing the projection of a Vector on some of its components.

    Parameters
    ----------
    components
        List or 1D |NumPy array| of the indices of the vector components that are to
        be extracted by the operator.
    source
        Source |VectorSpace| of the operator.
    name
        Name of the operator.
    """

    linear = True

    def __init__(self, components, source, name=None):
        assert all(0 <= c < source.dim for c in components)
        self.components = np.array(components)
        self.range = NumpyVectorSpace(len(components))
        self.source = source
        self.name = name

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        return NumpyVectorArray(U.components(self.components, ind), copy=False)

    def restricted(self, components):
        assert all(0 <= c < self.range.dim for c in components)
        source_components = self.components[components]
        return IdentityOperator(NumpyVectorSpace(len(source_components))), source_components


class IdentityOperator(OperatorBase):
    """The identity |Operator|.

    In other word ::

        op.apply(U) == U

    Parameters
    ----------
    space
        The |VectorSpace| the operator acts on.
    name
        Name of the operator.
    """

    linear = True

    def __init__(self, space, name=None):
        self.source = self.range = space
        self.name = name

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        return U.copy(ind=ind)


class ConstantOperator(OperatorBase):
    """A constant |Operator| always returning the same vector.

    Parameters
    ----------
    value
        A |VectorArray| of length 1 containing the vector which is
        returned by the operator.
    source
        Source |VectorSpace| of the operator.
    copy
        If `True`, store a copy of `vector` instead of `vector`
        itself.
    name
        Name of the operator.
    """

    linear = False

    def __init__(self, value, source, copy=True, name=None):
        assert isinstance(value, VectorArrayInterface)
        assert len(value) == 1
        self.source = source
        self.range = value.space
        self.name = name
        self._value = value.copy() if copy else value

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        count = len(U) if ind is None else 1 if isinstance(ind, Number) else len(ind)
        return self._value.copy(ind=([0] * count))


class VectorArrayOperator(OperatorBase):
    """Wraps a |VectorArray| as an |Operator|.

    If `transposed == False`, the operator maps from `NumpyVectorSpace(len(array))`
    to `array.space` by forming linear combinations of the vectors in the array
    with given coefficient arrays.

    If `transposed == True`, the operator maps from `array.space` to
    `NumpyVectorSpace(len(array))` by forming the scalar products of the arument
    with the vectors in the given array.

    Parameters
    ----------
    array
        The |VectorArray| which is to be treated as an operator.
    transposed
        See description above.
    copy
        If `True`, store a copy of `array` instead of `array` itself.
    name
        The name of the operator.
    """

    linear = True

    def __init__(self, array, transposed=False, copy=True, name=None):
        self._array = array.copy() if copy else array
        if transposed:
            self.source = array.space
            self.range = NumpyVectorSpace(len(array))
        else:
            self.source = NumpyVectorSpace(len(array))
            self.range = array.space
        self.transposed = transposed
        self.name = name

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        if not self.transposed:
            if ind is not None:
                U = U.copy(ind)
            return self._array.lincomb(U.data)
        else:
            return NumpyVectorArray(U.dot(self._array, ind=ind, pairwise=False), copy=False)

    def assemble_lincomb(self, operators, coefficients, name=None):

        transposed = operators[0].transposed
        if not all(isinstance(op, VectorArrayOperator) and op.transposed == transposed for op in operators):
            return None

        if coefficients[0] == 1:
            array = operators[0]._array.copy()
        else:
            array = operators[0]._array * coefficients[0]
        for op, c in izip(operators[1:], coefficients[1:]):
            array.axpy(c, op._array)
        return VectorArrayOperator(array, transposed=transposed, copy=False, name=name)

    def as_vector(self, mu=None):
        if len(self._array) != 1:
            raise TypeError('This operator does not represent a vector or linear functional.')
        else:
            return self._array.copy()


class VectorOperator(VectorArrayOperator):
    """Wrap a vector as a vector-like |Operator|.

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
    """

    linear = True
    source = NumpyVectorSpace(1)

    def __init__(self, vector, copy=True, name=None):
        assert isinstance(vector, VectorArrayInterface)
        assert len(vector) == 1
        super(VectorOperator, self).__init__(vector, transposed=False, copy=copy, name=name)

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        count = len(U) if ind is None else 1 if isinstance(ind, Number) else len(ind)
        R = self._array.copy(ind=([0] * count))
        for i, c in enumerate(U.data):
            R.scal(c[0], ind=i)
        return R


class VectorFunctional(VectorArrayOperator):
    """Wrap a vector as a linear |Functional|.

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
    """

    linear = True
    range = NumpyVectorSpace(1)

    def __init__(self, vector, product=None, copy=True, name=None):
        assert isinstance(vector, VectorArrayInterface)
        assert len(vector) == 1
        assert product is None or isinstance(product, OperatorInterface) and vector in product.source
        if product is None:
            super(VectorFunctional, self).__init__(vector, transposed=True, copy=copy, name=name)
        else:
            super(VectorFunctional, self).__init__(product.apply(vector), transposed=True, copy=False, name=name)


class FixedParameterOperator(OperatorBase):
    """Makes an |Operator| |Parameter|-independent by providing it a fixed |Parameter|.

    Parameters
    ----------
    operator
        The |Operator| to wrap.
    mu
        The fixed |Parameter| that will be fed to the
        :meth:`~pymor.operators.interfaces.OperatorInterface.apply` method
        of `operator`.
    """

    def __init__(self, operator, mu=None, name=None):
        assert isinstance(operator, OperatorInterface)
        assert operator.parse_parameter(mu) or True
        self.source = operator.source
        self.range = operator.range
        self.operator = operator
        self.mu = mu.copy()
        self.linear = operator.linear
        self.name = name

    def apply(self, U, ind=None, mu=None):
        return self.operator.apply(U, mu=self.mu)

    @property
    def invert_options(self):
        return self.operator.invert_options

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        self.operator.apply_inverse(U, ind=ind, mu=self.mu, options=options)
