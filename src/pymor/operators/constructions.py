# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Andreas Buhr <andreas@andreasbuhr.de>

"""Module containing some constructions to obtain new operators from old ones."""

from __future__ import absolute_import, division, print_function

from numbers import Number
from itertools import izip

import numpy as np

from pymor.core.defaults import defaults_sid, defaults
from pymor.core.interfaces import ImmutableInterface
from pymor.operators.basic import OperatorBase
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Parametric
from pymor.parameters.interfaces import ParameterFunctionalInterface
from pymor.vectorarrays.interfaces import VectorArrayInterface, VectorSpace
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace


class LincombOperator(OperatorBase):
    """An operator representing a linear combination of arbitrary |Operators|.

    Parameters
    ----------
    operators
        List of |Operators| whose linear combination is formed.
    coefficients
        A list of linear coefficients. A linear coefficient can
        either be a fixed number or a |ParameterFunctional|.
    name
        Name of the operator.
    """

    def __init__(self, operators, coefficients, name=None):
        assert len(operators) > 0
        assert len(operators) == len(coefficients)
        assert all(isinstance(op, OperatorInterface) for op in operators)
        assert all(isinstance(c, (ParameterFunctionalInterface, Number)) for c in coefficients)
        assert all(op.source == operators[0].source for op in operators[1:])
        assert all(op.range == operators[0].range for op in operators[1:])
        self.source = operators[0].source
        self.range = operators[0].range
        self.operators = operators
        self.linear = all(op.linear for op in operators)
        self.coefficients = coefficients
        self.name = name
        self.build_parameter_type(inherits=list(operators) +
                                  [f for f in coefficients if isinstance(f, ParameterFunctionalInterface)])
        self._try_assemble = not self.parametric

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

    def apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None):
        if hasattr(self, '_assembled_operator'):
            if self._defaults_sid == defaults_sid():
                return self._assembled_operator.apply2(V, U, V_ind=V_ind, U_ind=U_ind, product=product)
            else:
                return self.assemble().apply2(V, U, V_ind=V_ind, U_ind=U_ind, product=product)
        elif self._try_assemble:
            return self.assemble().apply2(V, U, V_ind=V_ind, U_ind=U_ind, product=product)
        coeffs = self.evaluate_coefficients(mu)
        R = self.operators[0].apply2(V, U, V_ind=V_ind, U_ind=U_ind, mu=mu, product=product)
        R *= coeffs[0]
        for op, c in izip(self.operators[1:], coeffs[1:]):
            R += c * op.apply2(V, U, V_ind=V_ind, U_ind=U_ind, mu=mu, product=product)
        return R

    def pairwise_apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None):
        if hasattr(self, '_assembled_operator'):
            if self._defaults_sid == defaults_sid():
                return self._assembled_operator.pairwise_apply2(V, U, V_ind=V_ind, U_ind=U_ind, product=product)
            else:
                return self.assemble().pairwise_apply2(V, U, V_ind=V_ind, U_ind=U_ind, product=product)
        elif self._try_assemble:
            return self.assemble().pairwise_apply2(V, U, V_ind=V_ind, U_ind=U_ind, product=product)
        coeffs = self.evaluate_coefficients(mu)
        R = self.operators[0].pairwise_apply2(V, U, V_ind=V_ind, U_ind=U_ind, mu=mu, product=product)
        R *= coeffs[0]
        for op, c in izip(self.operators[1:], coeffs[1:]):
            R += c * op.pairwise_apply2(V, U, V_ind=V_ind, U_ind=U_ind, mu=mu, product=product)
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

    def projected(self, range_basis, source_basis, product=None, name=None):
        if hasattr(self, '_assembled_operator'):
            if self._defaults_sid == defaults_sid():
                return self._assembled_operator.projected(range_basis, source_basis, product, name)
            else:
                return self.assemble().projected(range_basis, source_basis, product, name)
        elif self._try_assemble:
            return self.assemble().projected(range_basis, source_basis, product, name)
        proj_operators = [op.projected(range_basis=range_basis, source_basis=source_basis, product=product)
                          for op in self.operators]
        return self.with_(operators=proj_operators, name=name or self.name + '_projected')

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        """See :meth:`NumpyMatrixOperator.projected_to_subbasis`."""
        assert dim_source is None or dim_source <= self.source.dim
        assert dim_range is None or dim_range <= self.range.dim
        proj_operators = [op.projected_to_subbasis(dim_range=dim_range, dim_source=dim_source)
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
        self.name = name

    def apply(self, U, ind=None, mu=None):
        mu = self.parse_parameter(mu)
        return self.second.apply(self.first.apply(U, ind=ind, mu=mu), mu=mu)

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        mu = self.parse_parameter(mu)
        return self.first.apply_adjoint(self.second.apply_adjoint(U, ind=ind, mu=mu, range_product=range_product),
                                        mu=mu, source_product=source_product)

    def restricted(self, dofs):
        restricted_second, second_source_dofs = self.second.restricted(dofs)
        restricted_first, first_source_dofs = self.first.restricted(second_source_dofs)
        if isinstance(restricted_second, IdentityOperator):
            return restricted_first, first_source_dofs
        elif isinstance(restricted_first, IdentityOperator):
            return restricted_second, first_source_dofs
        else:
            return Concatenation(restricted_second, restricted_first), first_source_dofs

    def projected(self, range_basis, source_basis, product=None, name=None):
        if not self.parametric and self.linear:
            return super(Concatenation, self).projected(range_basis, source_basis, product=product, name=name)
        projected_first = self.first.projected(None, source_basis, product=None)
        if isinstance(projected_first, VectorArrayOperator) and not projected_first.transposed:
            return self.second.projected(range_basis, projected_first._array, product=product, name=name)
        else:
            projected_second = self.second.projected(range_basis, None, product=product)
            return Concatenation(projected_second, projected_first, name=name or self.name + '_projected')


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

    def restricted(self, dofs):
        assert all(0 <= c < self.range.dim for c in dofs)
        source_dofs = self.components[dofs]
        return IdentityOperator(NumpyVectorSpace(len(source_dofs))), source_dofs


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

    def jacobian(self, U, mu=None):
        assert U in self.source
        assert len(U) == 1
        return ZeroOperator(self.source, self.range, name=self.name + '_jacobian')

    def projected(self, range_basis, source_basis, product=None, name=None):
        assert source_basis is None or source_basis in self.source
        assert range_basis is None or range_basis in self.range
        assert product is None or product.source == product.range == self.range
        if range_basis is not None:
            if product:
                projected_value = NumpyVectorArray(product.apply2(range_basis, self._value).T, copy=False)
            else:
                projected_value = NumpyVectorArray(range_basis.dot(self._value).T, copy=False)
        else:
            projected_value = self._value
        if source_basis is None:
            return ConstantOperator(projected_value, self.source, copy=False,
                                    name=self.name + '_projected')
        else:
            return ConstantOperator(projected_value, NumpyVectorSpace(len(source_basis)), copy=False,
                                    name=self.name + '_projected')


class ZeroOperator(OperatorBase):
    """The |Operator| which maps every vector to zero.

    Parameters
    ----------
    source
        Source |VectorSpace| of the operator.
    range
        Range |VectorSpace| of the operator.
    name
        Name of the operator.
    """

    linear = True

    def __init__(self, source, range, name=None):
        assert isinstance(source, VectorSpace)
        assert isinstance(range, VectorSpace)
        self.source = source
        self.range = range
        self.name = name

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        count = len(U) if ind is None else 1 if isinstance(ind, Number) else len(ind)
        return self.range.zeros(count)

    def projected(self, range_basis, source_basis, product=None, name=None):
        assert source_basis is None or source_basis in self.source
        assert range_basis is None or range_basis in self.range
        assert product is None or product.source == product.range == self.range
        if source_basis is not None and range_basis is not None:
            return NumpyMatrixOperator(np.zeros((len(range_basis), len(source_basis))),
                                       name=self.name + '_projected')
        else:
            new_source = NumpyVectorSpace(len(source_basis)) if source_basis is not None else self.source
            new_range = NumpyVectorSpace(len(range_basis)) if range_basis is not None else self.source
            return ZeroOperator(new_source, new_range, name=self.name + '_projected')

    def assemble_lincomb(self, operators, coefficients, name=None):
        assert operators[0] is self
        if len(operators) > 1:
            return operators[1].assemble_lincomb(operators[1:], coefficients[1:], name=name)
        else:
            return self


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
            return NumpyVectorArray(U.dot(self._array, ind=ind), copy=False)

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        assert U in self.range
        assert source_product is None or source_product.source == source_product.range == self.source
        assert range_product is None or range_product.source == range_product.range == self.range
        if not self.transposed:
            if range_product:
                ATPrU = NumpyVectorArray(range_product.apply2(self._array, U, U_ind=ind).T, copy=False)
            else:
                ATPrU = NumpyVectorArray(self._array.dot(U, o_ind=ind).T, copy=False)
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

    def apply_inverse(self, V, ind=None, mu=None, options=None):
        return self.operator.apply_inverse(V, ind=ind, mu=self.mu, options=options)

    def jacobian(self, U, mu=None):
        return self.operator.jacobian(U, mu=self.mu)


class AdjointOperator(OperatorBase):
    """Represents the adjoint of a given |Operator|.

    See :meth:`~pymor.operators.interfaces.OperatorInterface.apply_adjoint`.

    Parameters
    ----------
    operator
        The |Operator| of which the adjoint is formed.
    source_product
        If not `None`, scalar product |Operator| for the source |VectorSpace|
        w.r.t. which to take the adjoint.
    range_product
        If not `None`, scalar product |Operator| for the range |VectorSpace|
        w.r.t. which to take the adjoint.
    name
        If not `None`, name of the operator.
    """

    linear = True

    def __init__(self, operator, source_product=None, range_product=None, name=None):
        assert isinstance(operator, OperatorInterface)
        assert operator.linear
        self.build_parameter_type(inherits=(operator,))
        self.source = operator.range
        self.range = operator.source
        self.operator = operator
        self.source_product = source_product
        self.range_product = range_product
        self.name = name or operator.name + '_adjoint'

    def apply(self, U, ind=None, mu=None):
        return self.operator.apply_adjoint(U, ind=ind, mu=mu,
                                           source_product=self.source_product, range_product=self.range_product)

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        if range_product != self.source_product:
            if range_product:
                U = range_product.apply(U, ind=ind)
                ind = None
            if self.source_product:
                U = self.source_product.apply_inverse(U, ind=ind)
                ind = None

        U = self.operator.apply(U, ind=ind, mu=mu)

        if source_product != self.range_product:
            if self.range_product:
                U = self.range_product.apply(U)
            if source_product:
                U = source_product.apply_inverse(U)

        return U

    def projected(self, range_basis, source_basis, product=None, name=None):
        if range_basis is not None:
            if product is not None:
                range_basis = product.apply(range_basis)
            if self.source_product:
                range_basis = self.source_product.apply_inverse(range_basis)

        if source_basis is not None and self.range_product:
            source_basis = self.range_product.apply(source_basis)

        operator = self.operator.projected(source_basis, range_basis)
        range_product = self.range_product if source_basis is None else None
        source_product = self.source_product if range_basis is None else None
        return AdjointOperator(operator, source_product=source_product, range_product=range_product,
                               name=name or self.name + '_projected')


class SelectionOperator(OperatorBase):
    """An |Operator| selecting one out of a list of |Operators|.

    operators[i] is used
    if parameterfunctional(mu) is less or equal than boundaries[i]
    and greater than boundaries[i-1]::

        -infty ------- boundaries[i] ---------- boundaries[i+1] ------- infty
                            |                        |
        --- operators[i] ---|---- operators[i+1] ----|---- operators[i+2]
                            |                        |

    Parameters
    ----------
    operators
        List of |Operators| from which one |Operator| is
        selected based on a parameter.
    parameter_functional
        A |ParameterFunctional| used for the selection of one |Operator|.
    boundaries
        The interval boundaries as defined above.
    name
        Name of the operator.

    """
    def __init__(self, operators, parameter_functional, boundaries, name=None):
        assert len(operators) > 0
        assert len(boundaries) == len(operators) - 1
        # check that boundaries are ascending:
        for i in range(len(boundaries)-1):
            assert boundaries[i] <= boundaries[i+1]
        assert all(isinstance(op, OperatorInterface) for op in operators)
        assert all(op.source == operators[0].source for op in operators[1:])
        assert all(op.range == operators[0].range for op in operators[1:])
        self.source = operators[0].source
        self.range = operators[0].range
        self.operators = tuple(operators)
        self.linear = all(op.linear for op in operators)

        self.name = name
        self.build_parameter_type(inherits=list(operators) + [parameter_functional])
        self._try_assemble = not self.parametric

        self.boundaries = tuple(boundaries)
        self.parameter_functional = parameter_functional

    def _get_operator_number(self, mu):
        value = self.parameter_functional.evaluate(mu)
        for i in range(len(self.boundaries)):
            if self.boundaries[i] >= value:
                return i
        return len(self.boundaries)

    def apply(self, U, ind=None, mu=None):
        mu = self.parse_parameter(mu)
        operator_number = self._get_operator_number(mu)
        return self.operators[operator_number].apply(U, ind=ind, mu=mu)

    def as_vector(self, mu=None):
        mu = self.parse_parameter(mu)
        operator_number = self._get_operator_number(mu)
        return self.operators[operator_number].as_vector(mu=mu)

    def projected(self, range_basis, source_basis, product=None, name=None):
        projected_operators = [op.projected(range_basis, source_basis, product=product, name=name)
                               for op in self.operators]
        return SelectionOperator(projected_operators, self.parameter_functional, self.boundaries,
                                 name or self.name + '_projected')


@defaults('raise_negative', 'tol')
def induced_norm(product, raise_negative=True, tol=1e-10, name=None):
    """The induced norm of a scalar product.

    The norm of a the vectors in a |VectorArray| U is calculated by
    calling ::

        product.pairwise_apply2(U, U, mu=mu)

    In addition, negative norm squares of absolute value smaller
    than `tol` are clipped to `0`.
    If `raise_negative` is `True`, a :exc:`ValueError` exception
    is raised if there are still negative norm squares afterwards.

    Parameters
    ----------
    product
        The scalar product |Operator| for which the norm is to be
        calculated.
    raise_negative
        If `True`, raise an exception if calcuated norm is negative.
    tol
        See above.

    Returns
    -------
    norm
        A function `norm(U, mu=None)` taking a |VectorArray| `U`
        as input together with the |Parameter| `mu` which is
        passed to the product.
    """
    return InducedNorm(product, raise_negative, tol, name)


class InducedNorm(ImmutableInterface, Parametric):
    """Instantiated by :func:`induced_norm`. Do not use directly."""

    def __init__(self, product, raise_negative, tol, name):
        self.product = product
        self.raise_negative = raise_negative
        self.tol = tol
        self.name = name
        self.build_parameter_type(inherits=(product,))

    def __call__(self, U, mu=None):
        norm_squared = self.product.pairwise_apply2(U, U, mu=mu)
        if self.tol > 0:
            norm_squared = np.where(np.logical_and(0 > norm_squared, norm_squared > - self.tol),
                                    0, norm_squared)
        if self.raise_negative and np.any(norm_squared < 0):
            raise ValueError('norm is negative (square = {})'.format(norm_squared))
        return np.sqrt(norm_squared)
