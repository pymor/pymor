from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core import interfaces
from .interfaces import FunctionInterface


class ConstantFunction(FunctionInterface):

    def __init__(self, value=1.0, dim_domain=1, dim_range=1, name=None):
        '''
        here should be a contract to enforce that np.array(value, copy=False) is valid
        '''
        self.dim_domain = dim_domain
        self.dim_range = dim_range
        self.name = name
        value = np.array(value, copy=False)
        self._value = value.reshape((self.dim_range,))

    def __str__(self):
        return ('{name}: x -> {value}').format(name=self.name, value=self._value)

    def evaluate(self, x, mu={}):
        '''
        \todo    here should be a contract to enforce that np.array(x, copy=False, ndmin=1) is valid
        '''
        self.map_parameter(mu)   # ensure that there is no parameter ...
        x = np.array(x, copy=False, ndmin=1)
        if x.ndim == 1:
            assert x.shape[0] == self.dim_domain
            return np.array(self._value, ndmin=min(1, self.dim_range))
        else:
            assert x.shape[-1] == self.dim_domain
            return np.tile(self._value, x.shape[:-1] + (1,))


class GenericFunction(FunctionInterface):

    def __init__(self, mapping, dim_domain=1, dim_range=1, parameter_type=None, name=None):
        self.dim_domain = dim_domain
        self.dim_range = dim_range
        self.name = name
        self._mapping = mapping
        if parameter_type is not None:
            self.build_parameter_type(parameter_type)
            self._with_mu = True
        else:
            self._with_mu = False

    def __str__(self):
        return ('{name}: x -> {mapping}').format(name=self.name, mapping=self._mapping)

    def evaluate(self, x, mu={}):
        mu = self.map_parameter(mu)
        x = np.array(x, copy=False, ndmin=1)
        assert x.shape[-1] == self.dim_domain
        if self._with_mu:
            v = self._mapping(x, mu)
        else:
            v = self._mapping(x)
        if len(v.shape) < len(x.shape) and self.dim_range > 0:
            v = v[..., np.newaxis]
        return v
