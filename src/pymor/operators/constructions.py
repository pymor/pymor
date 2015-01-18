# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Module containing some constructions to obtain new operators from old ones."""

from __future__ import absolute_import, division, print_function

from numbers import Number
from itertools import izip

import numpy as np

from pymor.core.defaults import defaults_sid
from pymor.la.interfaces import VectorArrayInterface
from pymor.la.numpyvectorarray import NumpyVectorArray, NumpyVectorSpace
from pymor.operators.basic import OperatorBase
from pymor.operators.interfaces import OperatorInterface
from pymor.parameters.interfaces import ParameterFunctionalInterface
from pymor.parameters.functionals import ProjectionParameterFunctional


class LincombOperator(OperatorBase):
    """A generic |LincombOperator| representing a linear combination of arbitrary |Operators|.

    Parameters
    ----------
    operators
        List of |Operators| whose linear combination is formed.
    coefficients
        `None` or a list of linear coefficients. A linear coefficient can
        either be a fixed number or a |ParameterFunctional|.
    num_coefficients
        If `coefficients` is `None`, the number of linear coefficients (starting
        at index 0) which are given by the |Parameter| component with name
        `'coefficients_name'`. The missing coefficients are set to `1`.
    coefficients_name
        If `coefficients` is `None`, the name of the |Parameter| component providing
        the linear coefficients.
    name
        Name of the operator.
    """

    with_arguments = frozenset(('operators', 'coefficients', 'name'))

    def __init__(self, operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
        assert coefficients is None or len(operators) == len(coefficients)
        assert len(operators) > 0
        assert all(isinstance(op, OperatorInterface) for op in operators)
        assert coefficients is None or all(isinstance(c, (ParameterFunctionalInterface, Number)) for c in coefficients)
        assert all(op.source == operators[0].source for op in operators[1:])
        assert all(op.range == operators[0].range for op in operators[1:])
        assert coefficients is None or num_coefficients is None
        assert coefficients is None or coefficients_name is None
        assert coefficients is not None or coefficients_name is not None
        assert coefficients_name is None or isinstance(coefficients_name, str)
        self.source = operators[0].source
        self.range = operators[0].range
        self.operators = operators
        self.linear = all(op.linear for op in operators)
        if coefficients is None:
            num_coefficients = num_coefficients if num_coefficients is not None else len(operators)
            pad_coefficients = len(operators) - num_coefficients
            coefficients = [ProjectionParameterFunctional(coefficients_name, (num_coefficients,), i)
                            for i in range(num_coefficients)] + [1.] * pad_coefficients
        self.coefficients = coefficients
        self.name = name
        self.build_parameter_type(inherits=list(operators) +
                                  [f for f in coefficients if isinstance(f, ParameterFunctionalInterface)])
        self._try_assemble = not self.parametric

    def with_(self, **kwargs):
        assert set(kwargs.keys()) <= self.with_arguments
        operators = kwargs.get('operators', self.operators)
        coefficients = kwargs.get('coefficients', self.coefficients)
        assert len(operators) == len(self.operators)
        assert len(coefficients) == len(self.coefficients)
        return LincombOperator(operators, coefficients, name=kwargs.get('name', self.name))

    def evaluate_coefficients(self, mu):
        """Compute the linear coefficients of the linear combination for a given parameter.

        Parameters
        ----------
        mu
            |Parameter| for which to compute the linear coefficients.

        Returns
        -------
        List of linear coefficients.
        """
        mu = self.parse_parameter(mu)
        return np.array([c.evaluate(mu) if hasattr(c, 'evaluate') else c for c in self.coefficients])

    def apply(self, U, ind=None, mu=None):
        if hasattr(self, '_assembled_operator'):
            if self._defaults_sid == defaults_sid():
                return self._assembled_operator.apply(U, ind=ind)
            else:
                return self.assemble().apply(U, ind=ind)
        elif self._try_assemble:
            return self.assemble().apply(U, ind=ind)
        coeffs = self.evaluate_coefficients(mu)
        R = self.operators[0].apply(U, ind=ind, mu=mu)
        R.scal(coeffs[0])
        for op, c in izip(self.operators[1:], coeffs[1:]):
            R.axpy(c, op.apply(U, ind=ind, mu=mu))
        return R

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        if hasattr(self, '_assembled_operator'):
            if self._defaults_sid == defaults_sid():
                return self._assembled_operator.apply_adjoint(U, ind=ind, source_product=source_product,
                                                              range_product=range_product)
            else:
                return self.assemble().apply_adjoint(U, ind=ind, source_product=source_product,
                                                     range_product=range_product)
        elif self._try_assemble:
            return self.assemble().apply_adjoint(U, ind=ind, source_product=source_product,
                                                 range_product=range_product)
        coeffs = self.evaluate_coefficients(mu)
        R = self.operators[0].apply_adjoint(U, ind=ind, mu=mu, source_product=source_product,
                                            range_product=range_product)
        R.scal(coeffs[0])
        for op, c in izip(self.operators[1:], coeffs[1:]):
            R.axpy(c, op.apply_adjoint(U, ind=ind, mu=mu, source_product=source_product,
                                       range_product=range_product))
        return R

    def assemble(self, mu=None):
        if hasattr(self, '_assembled_operator'):
            if self._defaults_sid == defaults_sid():
                return self._assembled_operator
            else:
                self.logger.warn('Re-assembling since state of global defaults has changed.')
        operators = [op.assemble(mu) for op in self.operators]
        coefficients = self.evaluate_coefficients(mu)
        op = operators[0].assemble_lincomb(operators, coefficients, name=self.name + '_assembled')
        if not self.parametric:
            if op:
                self._assembled_operator = op
                self._defaults_sid = defaults_sid()
                return op
            else:
                self._try_assemble = False
                return self
        elif op:
            return op
        else:
            return LincombOperator(operators, coefficients, name=self.name + '_assembled')

    def jacobian(self, U, mu=None):
        if hasattr(self, '_assembled_operator'):
            if self._defaults_sid == defaults_sid():
                return self._assembled_operator.jacobian(U)
            else:
                return self.assemble().jacobian(U)
        elif self._try_assemble:
            return self.assemble().jacobian(U)
        jacobians = [op.jacobian(U, mu) for op in self.operators]
        coefficients = self.evaluate_coefficients(mu)
        jac = jacobians[0].assemble_lincomb(jacobians, coefficients, name=self.name + '_jacobian')
        if jac is None:
            return LincombOperator(jacobians, coefficients, name=self.name + '_jacobian')
        else:
            return jac

    def as_vector(self, mu=None):
        if hasattr(self, '_assembled_operator'):
            if self._defaults_sid == defaults_sid():
                return self._assembled_operator.as_vector()
            else:
                return self.assemble().as_vector()
        elif self._try_assemble:
            return self.assemble().as_vector()
        coefficients = self.evaluate_coefficients(mu)
        vectors = [op.as_vector(mu) for op in self.operators]
        R = vectors[0]
        R.scal(coefficients[0])
        for c, v in izip(coefficients[1:], vectors[1:]):
            R.axpy(c, v)
        return R

    def projected(self, source_basis, range_basis, product=None, name=None):
        if hasattr(self, '_assembled_operator'):
            if self._defaults_sid == defaults_sid():
                return self._assembled_operator.projected(source_basis, range_basis, product, name)
            else:
                return self.assemble().projected(source_basis, range_basis, product, name)
        elif self._try_assemble:
            return self.assemble().projected(source_basis, range_basis, product, name)
        proj_operators = [op.projected(source_basis=source_basis, range_basis=range_basis, product=product)
                          for op in self.operators]
        return self.with_(operators=proj_operators, name=name or self.name + '_projected')

    def projected_to_subbasis(self, dim_source=None, dim_range=None, name=None):
        """See :meth:`NumpyMatrixOperator.projected_to_subbasis`."""
        assert dim_source is None or dim_source <= self.source.dim
        assert dim_range is None or dim_range <= self.range.dim
        proj_operators = [op.projected_to_subbasis(dim_source=dim_source, dim_range=dim_range)
                          for op in self.operators]
        return self.with_(operators=proj_operators, name=name or '{}_projected_to_subbasis'.format(self.name))

    def __getstate__(self):
        d = self.__dict__.copy()
        if '_assembled_operator' in d:
            del d['_assembled_operator']
        return d


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

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        mu = self.parse_parameter(mu)
        return self.first.apply_adjoint(self.second.apply_adjoint(U, ind=ind, mu=mu, range_product=range_product),
                                        mu=mu, source_product=source_product)

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

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        assert U in self.range
        assert source_product is None or source_product.source == source_product.range == self.source
        assert range_product is None or range_product.source == range_product.range == self.range
        if range_product:
            PrU = range_product.apply(U, ind=ind)
        else:
            PrU = U.copy(ind=ind)
        if source_product:
            return source_product.apply_inverse(PrU)
        else:
            return PrU


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

    def projected(self, source_basis, range_basis, product=None, name=None):
        assert source_basis is None or source_basis in self.source
        assert range_basis is None or range_basis in self.range
        assert product is None or product.source == product.range == self.range
        if range_basis is not None:
            if product:
                projected_value = NumpyVectorArray(product.apply2(range_basis, self._value, pairwise=False).T,
                                                   copy=False)
            else:
                projected_value = NumpyVectorArray(range_basis.dot(self._value, pairwise=False).T, copy=False)
        else:
            projected_value = self._value
        if source_basis is None:
            return ConstantOperator(projected_value, self.source, copy=False,
                                    name=self.name + '_projected')
        else:
            return ConstantOperator(projected_value, NumpyVectorSpace(len(source_basis)), copy=False,
                                    name=self.name + '_projected')


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

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        assert U in self.range
        assert source_product is None or source_product.source == source_product.range == self.source
        assert range_product is None or range_product.source == range_product.range == self.range
        if not self.transposed:
            if range_product:
                ATPrU = NumpyVectorArray(range_product.apply2(self._array, U, U_ind=ind, pairwise=False).T, copy=False)
            else:
                ATPrU = NumpyVectorArray(self._array.dot(U, o_ind=ind, pairwise=False).T, copy=False)
            if source_product:
                return source_product.apply_inverse(ATPrU)
            else:
                return ATPrU
        else:
            if range_product:
                PrU = range_product.apply(U, ind=ind)
            else:
                PrU = U.copy(ind)
            ATPrU = self._array.lincomb(PrU.data)
            if source_product:
                return source_product.apply_inverse(ATPrU)
            else:
                return ATPrU

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
        return self.operator.apply(U, ind=ind, mu=self.mu)

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        return self.operator.apply_adjoint(U, ind=ind, mu=self.mu,
                                           source_product=source_product, range_product=range_product)

    @property
    def invert_options(self):
        return self.operator.invert_options

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        return self.operator.apply_inverse(U, ind=ind, mu=self.mu, options=options)
