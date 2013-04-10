# pymor (http://www.pymor.org)
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from .interfaces import ParameterFunctionalInterface


class ProjectionParameterFunctional(ParameterFunctionalInterface):
    '''`ParameterFunctional` which returns a component of the parameter.

    Parameters
    ----------
    parameter_type
        The parameter type of the parameters the functional takes.
    component
        The component to return.
    coordinates
        If not `None` return `mu[component][coordinates]` instead of
        `mu[component]`.
    name
        Name of the functional.
    '''

    def __init__(self, parameter_type, component, coordinates=None, name=None):
        super(ProjectionParameterFunctional, self).__init__()
        self.name = name
        self.build_parameter_type(parameter_type, local_global=True)
        assert component in self.parameter_type
        self.component = component
        if sum(self.parameter_type[component]) > 1:
            assert coordinates is not None and coordinates < self.parameter_type[component]
        self.coordinates = coordinates

    def evaluate(self, mu={}):
        mu = self.map_parameter(mu)
        if self.coordinates is None:
            return mu[self.component]
        else:
            return mu[self.component][self.coordinates]


class GenericParameterFunctional(ParameterFunctionalInterface):
    '''A wrapper making an arbitrary python function a `ParameterFunctional`

    Parameters
    ----------
    parameter_type
        The parameter type of the parameters the functional takes.
    mapping
        The function to wrap. The function is of the form `mapping(mu)`.
    name
        The name of the functional.
    '''

    def __init__(self, parameter_type, mapping, name=None):
        super(ProjectionParameterFunctional, self).__init__()
        self.name = name
        self._mapping = mapping
        self.build_parameter_type(parameter_type, local_global=True)

    def evaluate(self, mu={}):
        mu = self.map_parameter(mu)
        return self._mapping(mu)
