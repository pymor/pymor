# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.functions.interfaces import FunctionInterface
from pymor.parameters.interfaces import ParameterFunctionalInterface


class FunctionBase(FunctionInterface):
    """Base class for |Functions| providing some common functionality."""

    def __add__(self, other):
        """Returns a new :class:`LincombFunction` representing the sum of two functions, or
        of one function and a constant.
        """
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
        """Returns a new :class:`LincombFunction` representing the difference of two functions, or
        of one function and a constant.
        """
        if isinstance(other, FunctionInterface):
            return LincombFunction([self, other], [1., -1.])
        else:
            return self + (- np.array(other))

    def __mul__(self, other):
        """Returns a new :class:`LincombFunction` representing the product of a function by a scalar.
        """
        assert isinstance(other, Number)
        return LincombFunction([self], [other])

    __rmul__ = __mul__

    def __neg__(self):
        """Returns a new :class:`LincombFunction` representing the function scaled by -1."""
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

    Note that a GenericFunction can only be :mod:`~pymor.core.pickle`d
    if the function it is wrapping can be serialized. If normal pickling of the
    function fails, serialization using :func:`~pymor.core.pickle.dumps_function`
    will be tried as a last resort. For this reason, it is usually preferable to
    use ExpressionFunction instead, which always can be serialized.

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
            self.build_parameter_type(parameter_type, local_global=True)

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
        assert v.shape == x.shape[:-1] + self.shape_range

        return v


class ExpressionFunction(GenericFunction):
    """Turns a Python expression given as a string into a |Function|.

    Some |NumPy| arithmetic functions like 'sin', 'log', 'min' are supported.
    For a full list see the `functions` class attribute.

    .. warning::
       :meth:`eval` is used to evaluate the given expression. As a consequence,
       using this class with expression strings from untrusted sources will cause
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
    name
        The name of the function.
    """

    functions = {k: getattr(np, k) for k in {'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
                                             'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
                                             'exp', 'exp2', 'log', 'log2', 'log10', 'array',
                                             'min', 'minimum', 'max', 'maximum', 'pi', 'e',
                                             'sum', 'prod'}}

    def __init__(self, expression, dim_domain=1, shape_range=(), parameter_type=None, name=None):
        self.expression = expression
        code = compile(expression, '<expression>', 'eval')
        functions = self.functions
        mapping = lambda x, mu=None: eval(code, functions, {'x': x, 'mu': mu})
        super().__init__(mapping, dim_domain, shape_range, parameter_type, name)

    def __repr__(self):
        return 'ExpressionFunction({}, {}, {}, {})'.format(self.expression, repr(self.parameter_type),
                                                           self.shape_range, self.parameter_type)

    def __reduce__(self):
        return (ExpressionFunction,
                (self.expression, self.dim_domain, self.shape_range, self.parameter_type, getattr(self, '_name', None)))


class LincombFunction(FunctionBase):
    """A |Function| representing a linear combination of |Functions|.

    The linear coefficients can be provided as scalars or
    |ParameterFunctionals|. Alternatively, if no linear coefficients
    are given, the missing coefficients become part of the
    |Parameter| the functions expects.

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
        self.build_parameter_type(inherits=list(functions) +
                                  [f for f in coefficients if isinstance(f, ParameterFunctionalInterface)])

    def evaluate_coefficients(self, mu):
        """Compute the linear coefficients for a given |Parameter| `mu`."""
        mu = self.parse_parameter(mu)
        return np.array([c.evaluate(mu) if hasattr(c, 'evaluate') else c for c in self.coefficients])

    def evaluate(self, x, mu=None):
        mu = self.parse_parameter(mu)
        coeffs = self.evaluate_coefficients(mu)
        return sum(c * f(x, mu) for c, f in zip(coeffs, self.functions))
