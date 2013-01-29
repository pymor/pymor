from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from pymor.core import interfaces
from .interfaces import ParameterFunctionalInterface


class ProjectionParameterFunctional(ParameterFunctionalInterface):

    def __init__(self, parameter_type, component, coordinates=None, name=None):
        self.name = name
        self.set_parameter_type(parameter_type, local_global=True)
        assert component in self.parameter_type
        self.component = component
        if sum(self.parameter_type[component]) > 1:
            assert coordinates is not None and coordinates < self.parameter_type[component]
        self.coordinates = coordinates

    def evaluate(self, mu={}):
        '''
        \todo    here should be a contract to enforce that np.array(x, copy=False, ndmin=1) is valid
        '''
        mu = self.map_parameter(mu)
        if self.coordinates is None:
            return mu[self.component]
        else:
            return mu[self.component][self.coordinates]


class GenericParameterFunctional(ParameterFunctionalInterface):

    def __init__(self, parameter_type, mapping, name=None):
        self.name = name
        self._mapping = mapping
        self.set_parameter_type(parameter_type, local_global=True)

    def evaluate(self, mu={}):
        mu = self.map_parameter(mu)
        return self._mapping(mu)
