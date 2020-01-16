# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number
from itertools import chain

import numpy as np

from pymor.core.base import ImmutableObject, abstractmethod
from pymor.parameters.base import Parametric


class ParameterFunctional(ImmutableObject, Parametric):
    """Interface for |Parameter| functionals.

    A parameter functional is simply a function mapping a |Parameter| to
    a number.
    """

    @abstractmethod
    def evaluate(self, mu=None):
        """Evaluate the functional for the given |Parameter| `mu`."""
        pass

    @abstractmethod
    def d_mu(self, component, index=()):
        """Return the functionals's derivative with respect to an index of a parameter component.

        Parameters
        ----------
        component
            Parameter component
        index
            index in the parameter component

        Returns
        -------
        New |Parameter| functional representing the partial derivative.
        """
        pass

    def __call__(self, mu=None):
        return self.evaluate(mu)

    def __mul__(self, other):
        from pymor.parameters.functionals import ProductParameterFunctional
        if not isinstance(other, (Number, ParameterFunctional)):
            return NotImplemented
        return ProductParameterFunctional([self, other])

    __rmul__ = __mul__

    def __neg__(self):
        return self * (-1.)

    def _check_and_parse_input(self, component, index):
        # check whether component is in parameter_type
        if component not in self.parameter_type:
            return False, None
        # check whether index has the correct shape
        if isinstance(index, Number):
            index = (index,)
        index = tuple(index)
        for idx in index:
            assert isinstance(idx, Number)
        shape = self.parameter_type[component]
        for i,idx in enumerate(index):
            assert idx < shape[i], 'wrong input `index` given'
        return True, index


class ProjectionParameterFunctional(ParameterFunctional):
    """|ParameterFunctional| returning a component of the given parameter.

    For given parameter `mu`, this functional evaluates to ::

        mu[component_name][index]


    Parameters
    ----------
    component_name
        The name of the parameter component to return.
    component_shape
        The shape of the parameter component.
    index
        See above.
    name
        Name of the functional.
    """

    def __init__(self, component_name, component_shape, index=(), name=None):
        if isinstance(component_shape, Number):
            component_shape = () if component_shape == 0 else (component_shape,)
        assert len(index) == len(component_shape)
        assert not component_shape or index < component_shape

        self.__auto_init(locals())
        self.build_parameter_type({component_name: component_shape})

    def evaluate(self, mu=None):
        mu = self.parse_parameter(mu)
        return mu[self.component_name].item(self.index)

    def d_mu(self, component, index=()):
        check, index = self._check_and_parse_input(component, index)
        if check:
            if component == self.component_name and index == self.index:
                return ConstantParameterFunctional(1, name=self.name + '_d_mu')
        return ConstantParameterFunctional(0, name=self.name + '_d_mu')


class GenericParameterFunctional(ParameterFunctional):
    """A wrapper making an arbitrary Python function a |ParameterFunctional|

    Note that a GenericParameterFunctional can only be :mod:`pickled <pymor.core.pickle>`
    if the function it is wrapping can be pickled. For this reason, it is usually
    preferable to use :class:`ExpressionParameterFunctional` instead of
    :class:`GenericParameterFunctional`.

    Parameters
    ----------
    mapping
        The function to wrap. The function has signature `mapping(mu)`.
    parameter_type
        The |ParameterType| of the |Parameters| the functional expects.
    name
        The name of the functional.
    derivative_mappings
        A dict containing all partial derivatives of each component and index in the
        |ParameterType| with the signature `derivative_mappings[component][index](mu)`
    second_derivative_mappings
        A dict containing all second order partial derivatives of each component and index in the
        |ParameterType| with the signature `second_derivative_mappings[component_i][index_i][component_j][index_j](mu)`
    """

    def __init__(self, mapping, parameter_type, name=None, derivative_mappings=None, second_derivative_mappings=None):
        self.__auto_init(locals())
        self.build_parameter_type(parameter_type)

    def evaluate(self, mu=None):
        mu = self.parse_parameter(mu)
        value = self.mapping(mu)
        # ensure that we return a number not an array
        if isinstance(value, np.ndarray):
            return value.item()
        else:
            return value

    def d_mu(self, component, index=()):
        check, index = self._check_and_parse_input(component, index)
        if check:
            if self.derivative_mappings is None:
                raise ValueError('You must provide a dict of expressions for all \
                                  partial derivatives in self.parameter_type')
            else:
                if component in self.derivative_mappings:
                    if self.second_derivative_mappings is None:
                        return GenericParameterFunctional(self.derivative_mappings[component][index],
                                                        self.parameter_type, name=self.name + '_d_mu')
                    else:
                        if component in self.second_derivative_mappings:
                            return GenericParameterFunctional(self.derivative_mappings[component][index],
                                                    self.parameter_type, name=self.name + '_d_mu',
                                                    derivative_mappings=self.second_derivative_mappings[component][index])
                        else:
                            return GenericParameterFunctional(self.derivative_mappings[component][index],
                                                    self.parameter_type, name=self.name + '_d_mu',
                                                    derivative_mappings={})
                else:
                    raise ValueError('derivative expressions do not contain item {}'.format(component))
        return ConstantParameterFunctional(0, name=self.name + '_d_mu')


class ExpressionParameterFunctional(GenericParameterFunctional):
    """Turns a Python expression given as a string into a |ParameterFunctional|.

    Some |NumPy| arithmetic functions like `sin`, `log`, `min` are supported.
    For a full list see the `functions` class attribute.

    .. warning::
       :meth:`eval` is used to evaluate the given expression.
       Using this class with expression strings from untrusted sources will cause
       mayhem and destruction!

    Parameters
    ----------
    expression
        A Python expression in the parameter components of the given `parameter_type`.
    parameter_type
        The |ParameterType| of the |Parameters| the functional expects.
    name
        The name of the functional.
    derivative_expressions
        A dict containing a Python expression for the partial derivatives of each
        parameter component.
    second_derivative_expressions
        A dict containing a list of dicts of Python expressions for all second order partial derivatives of each
        parameter component i and j.
    """

    functions = {k: getattr(np, k) for k in {'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2',
                                             'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
                                             'exp', 'exp2', 'log', 'log2', 'log10', 'sqrt', 'array',
                                             'min', 'minimum', 'max', 'maximum', 'pi', 'e',
                                             'sum', 'prod', 'abs', 'sign', 'zeros', 'ones'}}
    functions['norm'] = np.linalg.norm
    functions['polar'] = lambda x: (np.linalg.norm(x, axis=-1), np.arctan2(x[..., 1], x[..., 0]) % (2*np.pi))
    functions['np'] = np

    def __init__(self, expression, parameter_type, name=None, derivative_expressions=None, second_derivative_expressions=None):
        self.expression = expression
        code = compile(expression, '<expression>', 'eval')
        functions = self.functions

        def get_lambda(exp_code):
            return lambda mu: eval(exp_code, functions, mu)

        exp_mapping = get_lambda(code)
        if derivative_expressions is not None:
            derivative_mappings = derivative_expressions.copy()
            for (key,exp) in derivative_mappings.items():
                exp_array = np.array(exp, dtype=object)
                for exp in np.nditer(exp_array, op_flags=['readwrite'], flags= ['refs_ok']):
                    exp_code = compile(str(exp), '<expression>', 'eval')
                    mapping = get_lambda(exp_code)
                    exp[...] = mapping
                derivative_mappings[key] = exp_array
        else:
            derivative_mappings = None
        if second_derivative_expressions is not None:
            second_derivative_mappings = second_derivative_expressions.copy()
            for (key_i,key_dicts) in second_derivative_mappings.items():
                key_dicts_array = np.array(key_dicts, dtype=object)
                for key_dict in np.nditer(key_dicts_array, op_flags=['readwrite'], flags= ['refs_ok']):
                    for (key_j, exp) in key_dict[()].items():
                        exp_array = np.array(exp, dtype=object)
                        for exp in np.nditer(exp_array, op_flags=['readwrite'], flags= ['refs_ok']):
                            exp_code = compile(str(exp), '<expression>', 'eval')
                            mapping = get_lambda(exp_code)
                            exp[...] = mapping
                        key_dict[()][key_j] = exp_array
                second_derivative_mappings[key_i] = key_dicts_array
        else:
            second_derivative_mappings = None
        super().__init__(exp_mapping, parameter_type, name, derivative_mappings, second_derivative_mappings)
        self.__auto_init(locals())

    def __reduce__(self):
        return (ExpressionParameterFunctional,
                (self.expression, self.parameter_type, getattr(self, '_name', None),
                 self.derivative_expressions, self.second_derivative_expressions))


class ProductParameterFunctional(ParameterFunctional):
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
        assert all(isinstance(f, (ParameterFunctional, Number)) for f in factors)
        self.__auto_init(locals())
        self.build_parameter_type(*(f for f in factors if isinstance(f, ParameterFunctional)))

    def evaluate(self, mu=None):
        mu = self.parse_parameter(mu)
        return np.array([f.evaluate(mu) if hasattr(f, 'evaluate') else f for f in self.factors]).prod()

    def d_mu(self, component, index=()):
        raise NotImplementedError


class ConjugateParameterFunctional(ParameterFunctional):
    """Conjugate of a given |ParameterFunctional|

    Evaluates a given |ParameterFunctional| and returns the complex
    conjugate of the value.

    Parameters
    ----------
    functional
        The |ParameterFunctional| of which the complex conjuate is
        taken.
    name
        Name of the functional.
    """

    def __init__(self, functional, name=None):
        self.functional = functional
        self.name = name or f'{functional.name}_conj'
        self.build_parameter_type(functional)

    def evaluate(self, mu=None):
        mu = self.parse_parameter(mu)
        return np.conj(self.functional.evaluate(mu))

    def d_mu(self, component, index=()):
        raise NotImplementedError


class ConstantParameterFunctional(ParameterFunctional):
    """|ParameterFunctional| returning a constant value for each parameter.

    Parameters
    ----------
    constant_value
        value of the functional
    name
        Name of the functional.
    """

    def __init__(self, constant_value, name=None):
        self.constant_value = constant_value
        self.__auto_init(locals())

    def evaluate(self, mu=None):
        return self.constant_value

    def d_mu(self, component, index=()):
        return self.with_(constant_value=0, name=self.name + '_d_mu')


class MinThetaParameterFunctional(ParameterFunctionalInterface):
    """|ParameterFunctional| implementing the min-theta approach from [Haa17]_ (Proposition 2.35).


    Parameters
    ----------
    thetas
        List or tuple of |ParameterFunctional|s

    mu_bar
        see [Haa17]_ (Proposition 2.35)

    alpha_mu_bar
        see [Haa17]_ (Proposition 2.35)

    name
        Name of the functional.
    """

    def __init__(self, thetas, mu_bar, alpha_mu_bar=1., name=None):
        assert isinstance(thetas, (list, tuple))
        assert all([isinstance(theta, (Number, ParameterFunctionalInterface)) for theta in thetas])
        thetas = [ConstantParameterFunctional(theta) if not isinstance(theta, ParameterFunctionalInterface) else theta
                  for theta in thetas]
        assert all([isinstance(f, ParameterFunctionalInterface) for f in thetas])
        self.build_parameter_type(*chain(thetas))
        mu_bar = self.parse_parameter(mu_bar)
        thetas_mu_bar = np.array([theta(mu_bar) for theta in thetas])
        assert np.all(thetas_mu_bar > 0)
        assert isinstance(alpha_mu_bar, Number)
        assert alpha_mu_bar > 0
        self.__auto_init(locals())
        self.thetas_mu_bar = thetas_mu_bar # why is this required after __auto_init?

    def evaluate(self, mu=None):
        mu = self.parse_parameter(mu)
        thetas_mu = np.array([theta(mu) for theta in self.thetas])
        assert np.all(thetas_mu > 0)
        return self.alpha_mu_bar * np.min(thetas_mu / self.thetas_mu_bar)

    def d_mu(self, component, index=()):
        raise NotImplementedError


class MaxThetaParameterFunctional(ParameterFunctionalInterface):
    """|ParameterFunctional| implementing the max-theta approach from [Haa17]_ (Exercise 5.12).


    Parameters
    ----------
    thetas
        List or tuple of |ParameterFunctional|s

    mu_bar
        see [Haa17]_ (Proposition 2.35)

    gamma_mu_bar
        see [Haa17]_ (Proposition 2.35)

    name
        Name of the functional.
    """

    def __init__(self, thetas, mu_bar, gamma_mu_bar=1., name=None):
        assert isinstance(thetas, (list, tuple))
        thetas = [ConstantParameterFunctional(f) if not isinstance(f, ParameterFunctionalInterface) else f
                  for f in thetas]
        assert all([isinstance(f, ParameterFunctionalInterface) for f in thetas])
        self.build_parameter_type(*chain(thetas))
        mu_bar = self.parse_parameter(mu_bar)
        thetas_mu_bar = np.array([theta(mu_bar) for theta in thetas])
        assert np.all(thetas_mu_bar > 0)
        assert isinstance(gamma_mu_bar, Number)
        assert gamma_mu_bar > 0
        self.__auto_init(locals())
        self.thetas_mu_bar = thetas_mu_bar # why is this required after __auto_init?

    def evaluate(self, mu=None):
        mu = self.parse_parameter(mu)
        thetas_mu = np.array([theta(mu) for theta in self.thetas])
        assert np.all(thetas_mu > 0)
        return self.gamma_mu_bar * np.max(thetas_mu / self.thetas_mu_bar)

    def d_mu(self, component, index=()):
        raise NotImplementedError
