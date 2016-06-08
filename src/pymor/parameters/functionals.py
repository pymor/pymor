# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.parameters.interfaces import ParameterFunctionalInterface


class ProjectionParameterFunctional(ParameterFunctionalInterface):
    """|ParameterFunctional| returning a component of the given parameter.

    For given parameter `mu`, this functional evaluates to ::

        mu[component_name][coordinates]


    Parameters
    ----------
    component_name
        The name of the parameter component to return.
    component_shape
        The shape of the parameter component.
    coordinates
        See above.
    name
        Name of the functional.
    """

    def __init__(self, component_name, component_shape, coordinates=tuple(), name=None):
        self.name = name
        if isinstance(component_shape, Number):
            component_shape = tuple() if component_shape == 0 else (component_shape,)
        self.build_parameter_type({component_name: component_shape}, local_global=True)
        self.component_name = component_name
        self.coordinates = coordinates
        assert len(coordinates) == len(component_shape)
        assert not component_shape or coordinates < component_shape

    def evaluate(self, mu=None):
        mu = self.parse_parameter(mu)
        return mu[self.component_name].item(self.coordinates)


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
        value = self._mapping(mu)
        # ensure that we return a number not an array
        if isinstance(value, np.ndarray):
            return value.item()
        else:
            return value


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
                                             'exp', 'exp2', 'log', 'log2', 'log10', 'array',
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


class ProductParameterFunctional(ParameterFunctionalInterface):
    """Forms the product of a list of |ParameterFunctionals| or numbers.

    Parameters
    ----------
    factors
        A list of |ParameterFunctionals| or numbers.
    name
        Name of the functional.
    """

    def __init__(self, factors, name=None):
        assert len(factors) > 0
        assert all(isinstance(f, (ParameterFunctionalInterface, Number)) for f in factors)
        self.name = name
        self.factors = tuple(factors)
        self.build_parameter_type(inherits=[f for f in factors if isinstance(f, ProductParameterFunctional)])

    def evaluate(self, mu=None):
        mu = self.parse_parameter(mu)
        return np.array([f.evaluate(mu) if hasattr(f, 'evaluate') else f for f in self.factors]).prod()
