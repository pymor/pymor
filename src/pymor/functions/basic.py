# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number

from itertools import izip
import numpy as np

from pymor.functions.interfaces import FunctionInterface
from pymor.parameters import ParameterFunctionalInterface


class FunctionBase(FunctionInterface):

    def __add__(self, other):
        if isinstance(other, Number) and other == 0:
            return self
        elif not isinstance(other, FunctionInterface):
            other = np.array(other)
            assert other.shape == self.shape_range
            if np.all(other == 0.):
                return self
            other = ConstantFunction(other, dim_domain=self.dim_domain)
        return LincombFunction([self, other], [1., 1.])

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, FunctionInterface):
            return LincombFunction([self, other], [1., -1.])
        else:
            return self + (- np.array(other))

    def __mul__(self, other):
        assert isinstance(other, Number)
        return LincombFunction([self], [other])

    __rmul__ = __mul__

    def __neg__(self):
        return LincombFunction([self], [-1.])


class ConstantFunction(FunctionBase):
    '''A constant function ::

        f: R^d -> R^shape(c), f(x) = c

    Parameters
    ----------
    value
        The constant c.
    dim_domain
        The dimension d.
    name
        The name of the function.
    '''

    def __init__(self, value=np.array(1.0), dim_domain=1, name=None):
        assert dim_domain > 0
        assert isinstance(value, (Number, np.ndarray))
        super(ConstantFunction, self).__init__()
        value = np.array(value)
        self._value = value
        self.dim_domain = dim_domain
        self.shape_range = value.shape
        self.name = name

    def __str__(self):
        return ('{name}: x -> {value}').format(name=self.name, value=self._value)

    def evaluate(self, x, mu=None):
        assert self.check_parameter(mu)
        x = np.array(x, copy=False, ndmin=1)
        assert x.shape[-1] == self.dim_domain
        if x.ndim == 1:
            return np.array(self._value)
        else:
            return np.tile(self._value, x.shape[:-1] + (1,) * len(self.shape_range))


class GenericFunction(FunctionBase):
    '''A wrapper making an arbitrary python function a `Function`

    Parameters
    ----------
    mapping
        The function to wrap. If parameter_type is None, the function is of
        the form `mapping(x)` and is expected to vectorized. In particular::

            mapping(x).shape == x.shape[:-1] + shape_range

        If parameter_type is not None, the function has to have the signature
        `mapping(x, mu)`.
    dim_domain
        The dimension of the domain.
    shape_range
        The of the values returned by the mapping.
    parameter_type
        The type of the `Parameter` that mapping accepts.
    name
        The name of the function.
    '''

    def __init__(self, mapping, dim_domain=1, shape_range=tuple(), parameter_type=None, name=None):
        assert dim_domain > 0
        assert isinstance(shape_range, (Number, tuple))
        super(GenericFunction, self).__init__()
        self.dim_domain = dim_domain
        self.shape_range = shape_range if isinstance(shape_range, tuple) else (shape_range,)
        self.name = name
        self._mapping = mapping
        if parameter_type is not None:
            self.build_parameter_type(parameter_type, local_global=True)

    def __str__(self):
        return ('{name}: x -> {mapping}').format(name=self.name, mapping=self._mapping)

    def evaluate(self, x, mu=None):
        x = np.array(x, copy=False, ndmin=1)
        assert x.shape[-1] == self.dim_domain
        if self.parametric:
            mu = self.parse_parameter(mu)
            v = self._mapping(x, mu)
        else:
            assert self.check_parameter(mu)
            v = self._mapping(x)
        assert v.shape == x.shape[:-1] + self.shape_range

        return v


class LincombFunction(FunctionBase):

    def __init__(self, functions, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
        assert coefficients is None or len(functions) == len(coefficients)
        assert len(functions) > 0
        assert all(isinstance(f, FunctionInterface) for f in functions)
        assert coefficients is None or all(isinstance(c, (ParameterFunctionalInterface, Number)) for c in coefficients)
        assert all(f.dim_domain == functions[0].dim_domain for f in functions[1:])
        assert all(f.shape_range == functions[0].shape_range for f in functions[1:])
        assert coefficients is None or num_coefficients is None
        assert coefficients is None or coefficients_name is None
        assert coefficients is not None or coefficients_name is not None
        assert coefficients_name is None or isinstance(coefficients_name, str)
        self.dim_domain = functions[0].dim_domain
        self.shape_range = functions[0].shape_range
        self.functions = functions
        self.coefficients = coefficients
        self.coefficients_name = coefficients_name
        self.name = name
        if coefficients is None:
            self.num_coefficients = num_coefficients if num_coefficients is not None else len(functions)
            self.pad_coefficients = len(functions) - self.num_coefficients
            self.build_parameter_type({'coefficients': self.num_coefficients}, inherits=list(functions),
                                      global_names={'coefficients': coefficients_name})
        else:
            self.build_parameter_type(inherits=list(functions) +
                                      [f for f in coefficients if isinstance(f, ParameterFunctionalInterface)])

    def evaluate_coefficients(self, mu):
        mu = self.parse_parameter(mu)
        if self.coefficients is None:
            if self.pad_coefficients:
                return np.concatenate((self.local_parameter(mu)['coefficients'], np.ones(self.pad_coefficients)))
            else:
                return self.local_parameter(mu)['coefficients']

        else:
            return np.array([c.evaluate(mu) if hasattr(c, 'evaluate') else c for c in self.coefficients])

    def evaluate(self, x, mu=None):
        mu = self.parse_parameter(mu)
        coeffs = self.evaluate_coefficients(mu)
        return sum(c * f(x, mu) for c, f in izip(coeffs, self.functions))
