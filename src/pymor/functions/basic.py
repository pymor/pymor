# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from .interfaces import FunctionInterface


class ConstantFunction(FunctionInterface):
    '''A constant function ::

        f: R^d -> R^r, f(x) = c

    Parameters
    ----------
    value
        The constant c.
    dim_domain
        The dimension d.
    dim_range
        The dimension r.
    name
        The name of the function.
    '''

    def __init__(self, value=1.0, dim_domain=1, dim_range=1, name=None):
        super(ConstantFunction, self).__init__()
        self.dim_domain = dim_domain
        self.dim_range = dim_range
        self.name = name
        value = np.array(value, copy=False)
        self._value = value.reshape((self.dim_range,))
        self.lock()

    def __str__(self):
        return ('{name}: x -> {value}').format(name=self.name, value=self._value)

    def evaluate(self, x, mu=None):
        assert mu is None
        x = np.array(x, copy=False, ndmin=1)
        if x.ndim == 1:
            assert x.shape[0] == self.dim_domain
            return np.array(self._value, ndmin=min(1, self.dim_range))
        else:
            assert x.shape[-1] == self.dim_domain
            return np.tile(self._value, x.shape[:-1] + (1,))


class GenericFunction(FunctionInterface):
    '''A wrapper making an arbitrary python function a `Function`

    Parameters
    ----------
    mapping
        The function to wrap. If parameter_type is None, the function is of
        the form `mapping(x)` and is expected to vectorized. If parameter_type
        is not None, the function has to have the form `mapping(x, mu)`.
    dim_domain
        The dimension of the domain.
    dim_range
        The dimension of the range.
    parameter_type
        The type of the `Parameter` that mapping accepts.
    name
        The name of the function.
    '''

    def __init__(self, mapping, dim_domain=1, dim_range=1, parameter_type=None, name=None):
        super(GenericFunction, self).__init__()
        self.dim_domain = dim_domain
        self.dim_range = dim_range
        self.name = name
        self._mapping = mapping
        if parameter_type is not None:
            self.build_parameter_type(parameter_type)
            self._with_mu = True
        else:
            self._with_mu = False
        self.lock()

    def __str__(self):
        return ('{name}: x -> {mapping}').format(name=self.name, mapping=self._mapping)

    def evaluate(self, x, mu=None):
        mu = self.map_parameter(mu)
        x = np.array(x, copy=False, ndmin=1)
        if self.dim_domain > 0:
            assert x.shape[-1] == self.dim_domain
        if self._with_mu:
            v = self._mapping(x, mu)
        else:
            v = self._mapping(x)
        if len(v.shape) < len(x.shape) and self.dim_range > 0:
            v = v[..., np.newaxis]
        return v
