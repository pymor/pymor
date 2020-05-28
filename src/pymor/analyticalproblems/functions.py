# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from itertools import chain
from numbers import Number

import numpy as np

from pymor.core.base import abstractmethod
from pymor.parameters.base import ParametricObject
from pymor.parameters.functionals import ParameterFunctional, ExpressionParameterFunctional


class Function(ParametricObject):
    """Interface for |Parameter| dependent analytical functions.

    Every |Function| is a map of the form ::

       f(μ): Ω ⊆ R^d -> R^(shape_range)

    The returned values are |NumPy arrays| of arbitrary (but fixed)
    shape. Note that NumPy distinguishes between one-dimensional
    arrays of length 1 (with shape `(1,)`) and zero-dimensional
    scalar arrays (with shape `()`). In pyMOR, we usually
    expect scalar-valued functions to have `shape_range == ()`.

    While the function might raise an error if it is evaluated
    for an argument not in the domain Ω, the exact behavior is left
    undefined.

    Functions are vectorized in the sense, that if `x.ndim == k`, then ::

       f(x, μ)[i0, i1, ..., i(k-2)] == f(x[i0, i1, ..., i(k-2)], μ).

    In particular, `f(x, μ).shape == x.shape[:-1] + shape_range`.

    Attributes
    ----------
    dim_domain
        The dimension d > 0.
    shape_range
        The shape of the function values.
    """

    @abstractmethod
    def evaluate(self, x, mu=None):
        """Evaluate the function for given argument `x` and |parameter values| `mu`."""
        pass

    def __call__(self, x, mu=None):
        """Shorthand for :meth:`~Function.evaluate`."""
        return self.evaluate(x, mu)

    def __add__(self, other):
        if isinstance(other, Number) and other == 0:
            return self
        elif not isinstance(other, Function):
            other = np.array(other)
            assert other.shape == self.shape_range
            if np.all(other == 0.):
                return self
            other = ConstantFunction(other, dim_domain=self.dim_domain)
        return LincombFunction([self, other], [1., 1.])

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Function):
            return LincombFunction([self, other], [1., -1.])
        else:
            return self + (- np.array(other))

    def __mul__(self, other):
        if isinstance(other, (Number, ParameterFunctional)):
            return LincombFunction([self], [other])
        if isinstance(other, Function):
            return ProductFunction([self, other])
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self):
        return LincombFunction([self], [-1.])


class ConstantFunction(Function):
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
        self.__auto_init(locals())
        self.shape_range = value.shape

    def __str__(self):
        return f'{self.name}: x -> {self.value}'

    def evaluate(self, x, mu=None):
        x = np.array(x, copy=False, ndmin=1)
        assert x.shape[-1] == self.dim_domain
        if x.ndim == 1:
            return np.array(self.value)
        else:
            return np.tile(self.value, x.shape[:-1] + (1,) * len(self.shape_range))


class GenericFunction(Function):
    """Wrapper making an arbitrary Python function between |NumPy arrays| a proper |Function|.

    Note that a :class:`GenericFunction` can only be :mod:`pickled <pymor.core.pickle>`
    if the function it is wrapping can be pickled (cf. :func:`~pymor.core.pickle.dumps_function`).
    For this reason, it is usually preferable to use :class:`ExpressionFunction`
    instead of :class:`GenericFunction`.

    Parameters
    ----------
    mapping
        The function to wrap. If `parameters` is `None`, the function is of
        the form `mapping(x)`. If `parameters` is not `None`, the function has
        to have the signature `mapping(x, mu)`. Moreover, the function is expected
        to be vectorized, i.e.::

            mapping(x).shape == x.shape[:-1] + shape_range.

    dim_domain
        The dimension of the domain.
    shape_range
        The shape of the values returned by the mapping.
    parameters
        The |Parameters| the mapping accepts.
    name
        The name of the function.
    """

    def __init__(self, mapping, dim_domain=1, shape_range=(), parameters={}, name=None):
        assert dim_domain > 0
        assert isinstance(shape_range, (Number, tuple))
        if not isinstance(shape_range, tuple):
            shape_range = (shape_range,)
        self.parameters_own = parameters
        self.__auto_init(locals())

    def __str__(self):
        return f'{self.name}: x -> {self.mapping}'

    def evaluate(self, x, mu=None):
        assert self.parameters.assert_compatible(mu)
        x = np.array(x, copy=False, ndmin=1)
        assert x.shape[-1] == self.dim_domain

        if self.parametric:
            v = self.mapping(x, mu)
        else:
            v = self.mapping(x)

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
    parameters
        The |Parameters| the expression accepts.
    values
        Dictionary of additional constants that can be used in `expression`
        with their corresponding value.
    name
        The name of the function.
    """

    functions = ExpressionParameterFunctional.functions

    def __init__(self, expression, dim_domain=1, shape_range=(), parameters={}, values=None, name=None):
        values = values or {}
        code = compile(expression, '<expression>', 'eval')
        super().__init__(lambda x, mu={}: eval(code, dict(self.functions, **values), dict(mu, x=x, mu=mu)),
                         dim_domain, shape_range, parameters, name)
        self.__auto_init(locals())

    def __reduce__(self):
        return (ExpressionFunction,
                (self.expression, self.dim_domain, self.shape_range, self.parameters, self.values,
                 getattr(self, '_name', None)))


class LincombFunction(Function):
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
        assert all(isinstance(f, Function) for f in functions)
        assert all(isinstance(c, (ParameterFunctional, Number)) for c in coefficients)
        assert all(f.dim_domain == functions[0].dim_domain for f in functions[1:])
        assert all(f.shape_range == functions[0].shape_range for f in functions[1:])
        self.__auto_init(locals())
        self.dim_domain = functions[0].dim_domain
        self.shape_range = functions[0].shape_range

    def evaluate_coefficients(self, mu):
        """Compute the linear coefficients for given |parameter values| `mu`."""
        assert self.parameters.assert_compatible(mu)
        return np.array([c.evaluate(mu) if hasattr(c, 'evaluate') else c for c in self.coefficients])

    def evaluate(self, x, mu=None):
        assert self.parameters.assert_compatible(mu)
        coeffs = self.evaluate_coefficients(mu)
        return sum(c * f(x, mu) for c, f in zip(coeffs, self.functions))


class ProductFunction(Function):
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
        assert all(isinstance(f, Function) for f in functions)
        assert all(f.dim_domain == functions[0].dim_domain for f in functions[1:])
        assert all(f.shape_range == functions[0].shape_range for f in functions[1:])
        self.__auto_init(locals())
        self.dim_domain = functions[0].dim_domain
        self.shape_range = functions[0].shape_range

    def evaluate(self, x, mu=None):
        assert self.parameters.assert_compatible(mu)
        return np.prod([f(x, mu) for f in self.functions], axis=0)


class BitmapFunction(Function):
    """Define a 2D |Function| via a grayscale image.

    Parameters
    ----------
    filename
        Path of the image representing the function.
    bounding_box
        Lower left and upper right coordinates of the domain of the function.
    range
        A pixel of value p is mapped to `(p / 255.) * range[1] + range[0]`.
    """

    dim_domain = 2
    shape_range = ()

    def __init__(self, filename, bounding_box=None, range=None):
        bounding_box = bounding_box or [[0., 0.], [1., 1.]]
        range = range or [0., 1.]
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL is needed for loading images. Try 'pip install pillow'")
        img = Image.open(filename)
        if not img.mode == "L":
            self.logger.warning("Image " + filename + " not in grayscale mode. Converting to grayscale.")
            img = img.convert('L')
        self.__auto_init(locals())
        self.bitmap = np.array(img).T[:, ::-1]
        self.lower_left = np.array(bounding_box[0])
        self.size = np.array(bounding_box[1] - self.lower_left)

    def evaluate(self, x, mu=None):
        indices = np.maximum(np.floor((x - self.lower_left) * np.array(self.bitmap.shape) / self.size).astype(int), 0)
        F = (self.bitmap[np.minimum(indices[..., 0], self.bitmap.shape[0] - 1),
                         np.minimum(indices[..., 1], self.bitmap.shape[1] - 1)]
             * ((self.range[1] - self.range[0]) / 255.)
             + self.range[0])
        return F
