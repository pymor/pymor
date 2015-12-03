# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np

from pymor.core.pickle import dumps, loads, dumps_function, loads_function, PicklingError
from pymor.parameters.interfaces import ParameterFunctionalInterface


class ProjectionParameterFunctional(ParameterFunctionalInterface):
    """|ParameterFunctional| returning a component of the given parameter.

    Parameters
    ----------
    component_name
        The name of the component to return.
    component_shape
        The shape of the component.
    coordinates
        If not `None`, return `mu[component_name][coordinates]` instead of
        `mu[component_name]`.
    name
        Name of the functional.
    """

    def __init__(self, component_name, component_shape, coordinates=None, name=None):
        self.name = name
        if isinstance(component_shape, Number):
            component_shape = tuple() if component_shape == 0 else (component_shape,)
        self.build_parameter_type({component_name: component_shape}, local_global=True)
        self.component_name = component_name
        if sum(component_shape) > 1:
            assert coordinates is not None and coordinates < component_shape
        self.coordinates = coordinates

    def evaluate(self, mu=None):
        mu = self.parse_parameter(mu)
        if self.coordinates is None:
            return mu[self.component_name]
        else:
            return mu[self.component_name][self.coordinates]


class GenericParameterFunctional(ParameterFunctionalInterface):
    """A wrapper making an arbitrary Python function a |ParameterFunctional|

    Note that a GenericParameterFunctional can only be :mod:`~pymor.core.pickle`d
    if the function it is wrapping can be serialized. If normal pickling of the
    function fails, serialization using :func:`~pymor.core.pickle.dumps_function`
    will be tried as a last resort. For this reason, it is usually preferable to
    use ExpressionParameterFunctional instead, which always can be serialized.

    Parameters
    ----------
    parameter_type
        The |ParameterType| of the |Parameters| the functional takes.
    mapping
        The function to wrap. The function has signature `mapping(mu)`.
    name
        The name of the functional.
    """

    def __init__(self, mapping, parameter_type, name=None):
        self.name = name
        self._mapping = mapping
        self.build_parameter_type(parameter_type, local_global=True)

    def evaluate(self, mu=None):
        mu = self.parse_parameter(mu)
        return self._mapping(mu)

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


class ExpressionParameterFunctional(GenericParameterFunctional):
    """Turns a Python expression given as a string into a |ParameterFunctional|.

    Some |NumPy| arithmetic functions like 'sin', 'log', 'min' are supported.
    For a full list see the `functions` class attribute.

    .. warning::
       :meth:`eval` is used to evaluate the given expression. As a consequence,
       using this class with expression strings from untrusted sources will cause
       mayhem and destruction!

    Parameters
    ----------
    expression
        The Python expression for the functional as a string.
    parameter_type
        The |ParameterType| of the |Parameters| the functional takes.
    """

    functions = {k: getattr(np, k) for k in {'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
                                             'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
                                             'exp', 'exp2', 'log', 'log2', 'log10',
                                             'min', 'minimum', 'max', 'maximum', 'pi', 'e', }}

    def __init__(self, expression, parameter_type, name=None):
        self.expression = expression
        code = compile(expression, '<expression>', 'eval')
        functions = self.functions
        mapping = lambda mu: eval(code, functions, mu)
        super(ExpressionParameterFunctional, self).__init__(mapping, parameter_type, name)

    def __repr__(self):
        return 'ExpressionParameterFunctional({}, {})'.format(self.expression, repr(self.parameter_type))

    def __reduce__(self):
        return (ExpressionParameterFunctional,
                (self.expression, self.parameter_type, getattr(self, '_name', None)))
