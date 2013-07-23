# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip
from numbers import Number

import numpy as np

from pymor.core import abstractmethod
from pymor.la.interfaces import VectorArrayInterface
from pymor.la.numpyvectorarray import NumpyVectorArray
from pymor.operators.interfaces import OperatorInterface, MatrixBasedOperatorInterface, LincombOperatorInterface
from pymor.parameters import ParameterFunctionalInterface


class DefaultOperator(OperatorInterface):

    def apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None, pairwise=True):
        mu = self.parse_parameter(mu)
        assert isinstance(V, VectorArrayInterface)
        assert isinstance(U, VectorArrayInterface)
        U_ind = None if U_ind is None else np.array(U_ind, copy=False, dtype=np.int, ndmin=1)
        V_ind = None if V_ind is None else np.array(V_ind, copy=False, dtype=np.int, ndmin=1)
        if pairwise:
            lu = len(U_ind) if U_ind is not None else len(U)
            lv = len(V_ind) if V_ind is not None else len(V)
            assert lu == lv
        AU = self.apply(U, ind=U_ind, mu=mu)
        if product is not None:
            AU = product.apply(AU)
        return V.dot(AU, ind=V_ind, pairwise=pairwise)

    def lincomb(operators, coefficients=None, name=None):
        return LincombOperator(operators, coefficients, name=None)

    def __add__(self, other):
        if isinstance(other, Number):
            assert other == 0.
            return self
        return self.lincomb([self, other], [1, 1])

    __radd__ = __add__

    def __mul__(self, other):
        assert isinstance(other, Number)
        return self.lincomb([self], [other])

    def __str__(self):
        return '{}: R^{} --> R^{}  (parameter type: {}, class: {})'.format(
            self.name, self.dim_source, self.dim_range, self.parameter_type,
            self.__class__.__name__)

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        raise NotImplementedError


class DefaultMatrixBasedOperator(MatrixBasedOperatorInterface, DefaultOperator):

    linear = True
    assembled = False
    sparse = None

    def as_vector_array(self, mu=None):
        if not self.assembled:
            return self.assemble(mu).as_vector_array()
        else:
            raise NotImplementedError

    @abstractmethod
    def _assemble(self, mu=None):
        pass

    def assemble(self, mu=None, force=False):
        if self.assembled:
            mu = self.parse_parameter(mu)
            return self
        elif self.parameter_type is None:
            mu = self.parse_parameter(mu)
            if force or not self._last_mat:
                self._last_mat = self._assemble(mu)
                return self._last_mat
            else:
                return self._last_mat
        else:
            mu_s = self.strip_parameter(mu)
            if not force and self._last_mu is not None and self._last_mu.allclose(mu_s):
                return self._last_mat
            else:
                self._last_mu = mu_s.copy()  # TODO: add some kind of log message here
                self._last_mat = self._assemble(mu)
                return self._last_mat

    def apply(self, U, ind=None, mu=None):
        if not self.assembled:
            return self.assemble(mu).apply(U, ind=ind)
        else:
            raise NotImplementedError

    _last_mu = None
    _last_mat = None


class DefaultLincombOperator(DefaultOperator, LincombOperatorInterface):

    def __init__(self, operators, coefficients=None, global_names=None, name=None):
        assert coefficients is None or len(operators) == len(coefficients)
        assert len(operators) > 0
        assert all(isinstance(op, OperatorInterface) for op in operators)
        assert coefficients is None or all(isinstance(c, (ParameterFunctionalInterface, Number)) for c in coefficients)
        assert all(op.dim_source == operators[0].dim_source for op in operators[1:])
        assert all(op.dim_range == operators[0].dim_range for op in operators[1:])
        assert all(op.type_source == operators[0].type_source for op in operators[1:])
        assert all(op.type_range == operators[0].type_range for op in operators[1:])
        assert coefficients is None or global_names is None
        self.dim_source = operators[0].dim_source
        self.dim_range = operators[0].dim_range
        self.type_source = operators[0].type_source
        self.type_range = operators[0].type_range
        self.operators = operators
        self.coefficients = coefficients
        self.linear = all(op.linear for op in operators)
        self.name = name
        if coefficients is None:
            self.build_parameter_type({'coefficients': len(operators)}, inherits=list(operators),
                                      global_names=global_names)
        else:
            self.build_parameter_type(inherits=list(operators) +
                                      [f for f in coefficients if isinstance(f, ParameterFunctionalInterface)])

    def evaluate_coefficients(self, mu):
        mu = self.parse_parameter(mu)
        if self.coefficients is None:
            return self.local_parameter(mu)['coefficients']
        else:
            return np.array([c.evaluate(mu) if hasattr(c, 'evaluate') else c for c in self.coefficients])

    def projected(self, source_basis, range_basis, product=None, name=None):
        from pymor.operators.constructions import project_operator
        proj_operators = [project_operator(op, source_basis, range_basis, product, name='{}_projected'.format(op.name))
                          for op in self.operators]
        name = name or '{}_projected'.format(self.name)
        return type(self)(operators=proj_operators, coefficients=self.coefficients,
                          global_names=self.parameter_global_names, name=name)


class LincombOperator(DefaultLincombOperator):

    def __init__(self, operators, coefficients=None, global_names=None, name=None):
        super(LincombOperator, self).__init__(operators=operators, coefficients=coefficients, global_names=global_names,
                                              name=name)
        self.lock()

    def apply(self, U, ind=None, mu=None):
        mu = self.parse_parameter(mu)
        coeffs = self.evaluate_coefficients(mu)
        Vs = [op.apply(U, ind=ind, mu=mu) for op in self.operators]
        R = Vs[0]
        R.scal(coeffs[0])
        for V, c in izip(Vs[1:], coeffs[1:]):
            R.axpy(c, V)
        return R


class ConstantOperator(DefaultOperator):

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
        mu = self.parse_parameter(mu)
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

    def as_vector_array(self):
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


class ComponentProjection(DefaultOperator):

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
        mu = self.parse_parameter(mu)
        assert isinstance(U, self.type_source)
        assert U.dim == self.dim_source
        return NumpyVectorArray(U.components(self.components, ind), copy=False)
