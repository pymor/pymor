# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from itertools import chain
from numbers import Number

import numpy as np

from pymor.functions.interfaces import FunctionInterface
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.parameters.interfaces import ParameterFunctionalInterface


class FunctionBase(FunctionInterface):
    """Base class for |Functions| providing some common functionality."""

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
        if isinstance(other, (Number, ParameterFunctionalInterface)):
            return LincombFunction([self], [other])
        if isinstance(other, FunctionInterface):
            return ProductFunction([self, other])
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self):
        return LincombFunction([self], [-1.])


class ConstantFunction(FunctionBase):
    """A constant |Function| ::

        f: R^d -> R^shape(c), f(x) = c

    Parameters
    ----------
    value
        The constant c.
    dim_domain
        The dimension d.
    name
        The name of the function.
    """

    def __init__(self, value=np.array(1.0), dim_domain=1, name=None):
        assert dim_domain > 0
        assert isinstance(value, (Number, np.ndarray))
        value = np.array(value)
        self._value = value
        self.dim_domain = dim_domain
        self.shape_range = value.shape
        self.name = name

    def __str__(self):
        return '{name}: x -> {value}'.format(name=self.name, value=self._value)

    def __repr__(self):
        return 'ConstantFunction({}, {})'.format(repr(self._value), self.dim_domain)

    def evaluate(self, x, mu=None):
        x = np.array(x, copy=False, ndmin=1)
        assert x.shape[-1] == self.dim_domain
        if x.ndim == 1:
            return np.array(self._value)
        else:
            return np.tile(self._value, x.shape[:-1] + (1,) * len(self.shape_range))


class GenericFunction(FunctionBase):
    """Wrapper making an arbitrary Python function between |NumPy arrays| a proper |Function|.

    Note that a :class:`GenericFunction` can only be :mod:`pickled <pymor.core.pickle>`
    if the function it is wrapping can be pickled (cf. :func:`~pymor.core.pickle.dumps_function`).
    For this reason, it is usually preferable to use :class:`ExpressionFunction`
    instead of :class:`GenericFunction`.

    Parameters
    ----------
    mapping
        The function to wrap. If `parameter_type` is `None`, the function is of
        the form `mapping(x)`. If `parameter_type` is not `None`, the function has
        to have the signature `mapping(x, mu)`. Moreover, the function is expected
        to be vectorized, i.e.::

            mapping(x).shape == x.shape[:-1] + shape_range.

    dim_domain
        The dimension of the domain.
    shape_range
        The shape of the values returned by the mapping.
    parameter_type
        The |ParameterType| the mapping accepts.
    name
        The name of the function.
    """

    def __init__(self, mapping, dim_domain=1, shape_range=(), parameter_type=None, name=None):
        assert dim_domain > 0
        assert isinstance(shape_range, (Number, tuple))
        self.dim_domain = dim_domain
        self.shape_range = shape_range if isinstance(shape_range, tuple) else (shape_range,)
        self.name = name
        self._mapping = mapping
        if parameter_type is not None:
            self.build_parameter_type(parameter_type)

    def __str__(self):
        return '{name}: x -> {mapping}'.format(name=self.name, mapping=self._mapping)

    def evaluate(self, x, mu=None):
        x = np.array(x, copy=False, ndmin=1)
        assert x.shape[-1] == self.dim_domain

        if self.parametric:
            mu = self.parse_parameter(mu)
            v = self._mapping(x, mu)
        else:
            v = self._mapping(x)

        if v.shape != x.shape[:-1] + self.shape_range:
            assert v.shape[:len(x.shape) - 1] == x.shape[:-1]
            v = v.reshape(x.shape[:-1] + self.shape_range)

        return v


class ExpressionFunction(GenericFunction):
    """Turns a Python expression given as a string into a |Function|.

    Some |NumPy| arithmetic functions like 'sin', 'log', 'min' are supported.
    For a full list see the `functions` class attribute.

    .. warning::
       :meth:`eval` is used to evaluate the given expression.
       Using this class with expression strings from untrusted sources will cause
       mayhem and destruction!

    Parameters
    ----------
    expression
        A Python expression of one variable `x` and a parameter `mu` given as
        a string.
    dim_domain
        The dimension of the domain.
    shape_range
        The shape of the values returned by the expression.
    parameter_type
        The |ParameterType| the expression accepts.
    values
        Dictionary of additional constants that can be used in `expression`
        with their corresponding value.
    name
        The name of the function.
    """

    functions = ExpressionParameterFunctional.functions

    def __init__(self, expression, dim_domain=1, shape_range=(), parameter_type=None, values=None, name=None):
        self.expression = expression
        self.values = values or {}
        code = compile(expression, '<expression>', 'eval')
        super().__init__(lambda x, mu={}: eval(code, dict(self.functions, **self.values), dict(mu, x=x, mu=mu)),
                         dim_domain, shape_range, parameter_type, name)

    def __repr__(self):
        return 'ExpressionFunction({}, {}, {}, {}, {})'.format(self.expression, repr(self.parameter_type),
                                                               self.shape_range, self.parameter_type,
                                                               self.values)

    def __reduce__(self):
        return (ExpressionFunction,
                (self.expression, self.dim_domain, self.shape_range, self.parameter_type, self.values,
                 getattr(self, '_name', None)))


class LincombFunction(FunctionBase):
    """A |Function| representing a linear combination of |Functions|.

    The linear coefficients can be provided either as scalars or as
    |ParameterFunctionals|.

    Parameters
    ----------
    functions
        List of |Functions| whose linear combination is formed.
    coefficients
        A list of linear coefficients. A linear coefficient can
        either be a fixed number or a |ParameterFunctional|.
    name
        Name of the function.

    Attributes
    ----------
    functions
    coefficients
    """

    def __init__(self, functions, coefficients, name=None):
        assert len(functions) > 0
        assert len(functions) == len(coefficients)
        assert all(isinstance(f, FunctionInterface) for f in functions)
        assert all(isinstance(c, (ParameterFunctionalInterface, Number)) for c in coefficients)
        assert all(f.dim_domain == functions[0].dim_domain for f in functions[1:])
        assert all(f.shape_range == functions[0].shape_range for f in functions[1:])
        self.dim_domain = functions[0].dim_domain
        self.shape_range = functions[0].shape_range
        self.functions = functions
        self.coefficients = coefficients
        self.name = name
        self.build_parameter_type(*chain(functions,
                                         (f for f in coefficients if isinstance(f, ParameterFunctionalInterface))))

    def evaluate_coefficients(self, mu):
        """Compute the linear coefficients for a given |Parameter| `mu`."""
        mu = self.parse_parameter(mu)
        return np.array([c.evaluate(mu) if hasattr(c, 'evaluate') else c for c in self.coefficients])

    def evaluate(self, x, mu=None):
        mu = self.parse_parameter(mu)
        coeffs = self.evaluate_coefficients(mu)
        return sum(c * f(x, mu) for c, f in zip(coeffs, self.functions))


class ProductFunction(FunctionBase):
    """A |Function| representing a product of |Functions|.

    Parameters
    ----------
    functions
        List of |Functions| whose product is formed.
    name
        Name of the function.

    Attributes
    ----------
    functions
    """

    def __init__(self, functions, name=None):
        assert len(functions) > 0
        assert all(isinstance(f, FunctionInterface) for f in functions)
        assert all(f.dim_domain == functions[0].dim_domain for f in functions[1:])
        assert all(f.shape_range == functions[0].shape_range for f in functions[1:])
        self.dim_domain = functions[0].dim_domain
        self.shape_range = functions[0].shape_range
        self.functions = functions
        self.name = name
        self.build_parameter_type(*functions)

    def evaluate(self, x, mu=None):
        mu = self.parse_parameter(mu)
        return np.prod([f(x, mu) for f in self.functions], axis=0)
