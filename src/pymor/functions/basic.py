# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np

from .interfaces import FunctionInterface


class ConstantFunction(FunctionInterface):
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
        if not isinstance(value, np.ndarray):
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


class GenericFunction(FunctionInterface):
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
