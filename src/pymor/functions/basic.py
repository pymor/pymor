# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Laier <m_laie01@uni-muenster.de>

from __future__ import absolute_import, division, print_function

from numbers import Number

from itertools import izip
import numpy as np

from pymor.core.pickle import dumps, loads, dumps_function, loads_function, PicklingError
from pymor.functions.interfaces import FunctionInterface
from pymor.parameters import ParameterFunctionalInterface


class FunctionBase(FunctionInterface):
    """Base class for |Functions| providing some common functionality."""

    def __add__(self, other):
        """Returns a new :class:`LincombFunction` representing the sum of two functions, or
        one function and a constant.
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
        one function and a constant.
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
    """Wrapper making an arbitrary Python function between |NumPy arrays| a
    proper |Function|

    Parameters
    ----------
    mapping
        The function to wrap. If `parameter_type` is `None`, the function is of
        the form `mapping(x)` and is expected to vectorized. In particular::

            mapping(x).shape == x.shape[:-1] + shape_range.

        If `parameter_type` is not `None`, the function has to have the signature
        `mapping(x, mu)`.
    dim_domain
        The dimension of the domain.
    shape_range
        The shape of the values returned by the mapping.
    parameter_type
        The |ParameterType| the mapping accepts.
    name
        The name of the function.
    """

    def __init__(self, mapping, dim_domain=1, shape_range=tuple(), parameter_type=None, name=None):
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

    def __getstate__(self):
        s = self.__dict__.copy()
        try:
            pickled_mapping = dumps(self._mapping)
            picklable = True
        except PicklingError:
            self.logger.warn('Mapping not picklable, trying pymor.core.pickle.dumps_function.')
            pickled_mapping = dumps_function(self._mapping)
            picklable = False
        s['_mapping'] = pickled_mapping
        s['_picklable'] = picklable
        return s

    def __setstate__(self, state):
        if state.pop('_picklable'):
            state['_mapping'] = loads(state['_mapping'])
        else:
            state['_mapping'] = loads_function(state['_mapping'])
        self.__dict__.update(state)


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
        The Python expression of one variable `x` as a string.
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
                                             'exp', 'exp2', 'log', 'log2', 'log10',
                                             'min', 'minimum', 'max', 'maximum', }}

    def __init__(self, expression, dim_domain=1, shape_range=tuple(), parameter_type=None, name=None):
        self.expression = expression
        code = compile(expression, '<expression>', 'eval')
        functions = self.functions
        mapping = lambda x, mu=None: eval(code, functions, {'x':x, 'mu':mu})
        super(ExpressionFunction, self).__init__(mapping, dim_domain, shape_range, parameter_type, name)

    def __repr__(self):
        return 'ExpressionFunction({}, {}, {}, {})'.format(self.expression, repr(self.parameter_type),
                                                           self.shape_range, self.parameter_type)

    def __reduce__(self):
        return (ExpressionFunction,
                (self.expression, self.dim_domain, self.shape_range, self.parameter_type, self.name))


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
        `None` or a list of linear coefficients.
    num_coefficients
        If `coefficients` is `None`, the number of linear
        coefficients (starting at index 0) which are given by the
        |Parameter| component with name `'coefficients_name'`. The
        missing coefficients are set to `1`.
    coefficients_name
        If `coefficients` is `None`, the name of the |Parameter|
        component providing the linear coefficients.
    name
        Name of the function.

    Attributes
    ----------
    functions
    coefficients
    coefficients_name
    num_coefficients
    """

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
        """Compute the linear coefficients for a given |Parameter| `mu`."""
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
