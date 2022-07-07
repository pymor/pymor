# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.analyticalproblems.expressions import parse_expression
from pymor.core.base import abstractmethod
from pymor.parameters.base import Mu, ParametricObject, Parameters
from pymor.tools.floatcmp import float_cmp


class ParameterFunctional(ParametricObject):
    """Interface for |Parameter| functionals.

    A parameter functional is simply a function mapping |Parameters| to
    a number.
    """

    @abstractmethod
    def evaluate(self, mu=None):
        """Evaluate the functional for given |parameter values| `mu`."""
        pass

    def d_mu(self, parameter, index=0):
        """Return the functionals's derivative with respect to a given parameter.

        Parameters
        ----------
        parameter
            The |Parameter| w.r.t. which to return the derivative.
        index
            Index of the |Parameter|'s component w.r.t which to return the derivative.

        Returns
        -------
        New |ParameterFunctional| representing the partial derivative.
        """
        if parameter not in self.parameters:
            return ConstantParameterFunctional(0, name=f'{self.name}_d_{parameter}_{index}')
        else:
            raise NotImplementedError

    def __call__(self, mu=None):
        return self.evaluate(mu)

    def _add_sub(self, other, sign):
        if not isinstance(other, (ParameterFunctional, Number)):
            return NotImplemented

        if isinstance(other, Number):
            if other == 0:
                return self
            other = ConstantParameterFunctional(other)

        if self.name != 'LincombParameterFunctional' or not isinstance(self, LincombParameterFunctional):
            if other.name == 'LincombParameterFunctional' and isinstance(other, LincombParameterFunctional):
                functionals = (self,) + other.functionals
                coefficients = (1.,) + (other.coefficients if sign == 1. else tuple(-c for c in other.coefficients))
            else:
                functionals, coefficients = (self, other), (1., sign)
        elif other.name == 'LincombParameterFunctional' and isinstance(other, LincombParameterFunctional):
            functionals = self.functionals + other.functionals
            coefficients = self.coefficients + (other.coefficients if sign == 1.
                                                else tuple(-c for c in other.coefficients))
        else:
            functionals, coefficients = self.functionals + (other,), self.coefficients + (sign,)

        return LincombParameterFunctional(functionals, coefficients)

    def _radd_sub(self, other, sign):
        assert not isinstance(other, ParameterFunctional)  # handled by __add__/__sub__
        if not isinstance(other, Number):
            return NotImplemented

        if other == 0:
            return self
        other = ConstantParameterFunctional(other)

        if self.name != 'LincombParameterFunctional' or not isinstance(self, LincombParameterFunctional):
            functionals, coefficients = (other, self), (1., sign)
        else:
            functionals = (other,) + self.functionals
            coefficients = (1.,) + (self.coefficients if sign == 1. else tuple(-c for c in self.coefficients))

        return LincombParameterFunctional(functionals, coefficients)

    def __add__(self, other):
        return self._add_sub(other, 1.)

    def __sub__(self, other):
        return self._add_sub(other, -1.)

    def __radd__(self, other):
        return self._radd_sub(other, 1.)

    def __rsub__(self, other):
        return self._radd_sub(other, -1.)

    def __mul__(self, other):
        if not isinstance(other, (Number, ParameterFunctional)):
            return NotImplemented
        if self.name != 'ProductParameterFunctional' or not isinstance(self, ProductParameterFunctional):
            if isinstance(other, ProductParameterFunctional) and other.name == 'ProductParameterFunctional':
                return other.with_(factors=(self,) + other.factors)
            else:
                return ProductParameterFunctional((self, other))
        elif isinstance(other, ProductParameterFunctional) and other.name == 'ProductParameterFunctional':
            factors = self.factors + other.factors
            return ProductParameterFunctional(factors)
        else:
            return self.with_(factors=self.factors + (other,))

    __rmul__ = __mul__

    def __neg__(self):
        return self * (-1.)


class ProjectionParameterFunctional(ParameterFunctional):
    """|ParameterFunctional| returning a component value of the given parameter.

    For given parameter map `mu`, this functional evaluates to ::

        mu[parameter][index]


    Parameters
    ----------
    parameter
        The name of the parameter to return.
    size
        The size of the parameter.
    index
        See above.
    name
        Name of the functional.
    """

    def __init__(self, parameter, size=1, index=None, name=None):
        assert isinstance(size, Number)
        if index is None and size == 1:
            index = 0
        assert isinstance(index, Number)
        assert 0 <= index < size

        self.__auto_init(locals())
        self.parameters_own = {parameter: size}

    def evaluate(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        return mu[self.parameter].item(self.index)

    def d_mu(self, parameter, index=0):
        if parameter == self.parameter:
            assert 0 <= index < self.size
            if index == self.index:
                return ConstantParameterFunctional(1, name=f'{self.name}_d_{parameter}_{index}')
        return ConstantParameterFunctional(0, name=f'{self.name}_d_{parameter}_{index}')


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
    parameters
        The |Parameters| the functional depends on.
    name
        The name of the functional.
    derivative_mappings
        A dict containing all partial derivatives of each |Parameter| and index
        with the signature `derivative_mappings[parameter][index](mu)`
    second_derivative_mappings
        A dict containing all second order partial derivatives of each |Parameter| and index
        with the signature
        `second_derivative_mappings[parameter_i][index_i][parameter_j][index_j](mu)`
    """

    def __init__(self, mapping, parameters, name=None, derivative_mappings=None, second_derivative_mappings=None):
        self.__auto_init(locals())
        self.parameters_own = parameters

    def evaluate(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        value = self.mapping(mu)
        # ensure that we return a number not an array
        if isinstance(value, np.ndarray):
            return value.item()
        else:
            return value

    def d_mu(self, parameter, index=0):
        if parameter in self.parameters:
            assert 0 <= index < self.parameters[parameter]
            if self.derivative_mappings is None:
                raise ValueError('You must provide a dict of expressions for all \
                                  partial derivatives in self.parameters')
            else:
                if parameter in self.derivative_mappings:
                    if self.second_derivative_mappings is None:
                        return GenericParameterFunctional(
                            self.derivative_mappings[parameter][index],
                            self.parameters, name=f'{self.name}_d_{parameter}_{index}'
                        )
                    else:
                        if parameter in self.second_derivative_mappings:
                            return GenericParameterFunctional(
                                self.derivative_mappings[parameter][index],
                                self.parameters, name=f'{self.name}_d_{parameter}_{index}',
                                derivative_mappings=self.second_derivative_mappings[parameter][index]
                            )
                        else:
                            return GenericParameterFunctional(
                                self.derivative_mappings[parameter][index],
                                self.parameters, name=f'{self.name}_d_{parameter}_{index}',
                                derivative_mappings={}
                            )
                else:
                    raise ValueError('derivative expressions do not contain item {}'.format(parameter))
        return ConstantParameterFunctional(0, name=f'{self.name}_d_{parameter}_{index}')


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
        A Python expression in the parameter components of the given `parameters`.
    parameters
        The |Parameters| the functional depends on.
    name
        The name of the functional.
    derivative_expressions
        A dict containing a Python expression for the partial derivatives of each
        parameter component.
    second_derivative_expressions
        A dict containing a list of dicts of Python expressions for all second order partial
        derivatives of each parameter component i and j.
    """

    def __init__(self, expression, parameters, name=None, derivative_expressions=None,
                 second_derivative_expressions=None):
        self.expression_obj = parse_expression(expression, parameters=parameters)
        exp_mapping = self.expression_obj.to_numpy([])

        if derivative_expressions is not None:
            derivative_mappings = derivative_expressions.copy()
            for key, exp in derivative_mappings.items():
                if isinstance(exp, str):
                    exp = [exp]
                exp_array = np.array(exp, dtype=object)
                for exp in np.nditer(exp_array, op_flags=['readwrite'], flags=['refs_ok']):
                    exp_obj = parse_expression(str(exp), parameters=parameters)
                    exp[...] = exp_obj.to_numpy([])
                derivative_mappings[key] = exp_array
        else:
            derivative_mappings = None

        if second_derivative_expressions is not None:
            second_derivative_mappings = second_derivative_expressions.copy()
            for key_i, key_dicts in second_derivative_mappings.items():
                if isinstance(key_dicts, dict):
                    key_dicts = [key_dicts]
                key_dicts_array = np.array(key_dicts, dtype=object)
                for key_dict in np.nditer(key_dicts_array, op_flags=['readwrite'], flags=['refs_ok']):
                    for key_j, exp in key_dict[()].items():
                        if isinstance(exp, str):
                            exp = [exp]
                        exp_array = np.array(exp, dtype=object)
                        for exp in np.nditer(exp_array, op_flags=['readwrite'], flags=['refs_ok']):
                            exp_obj = parse_expression(str(exp), parameters=parameters)
                            exp[...] = exp_obj.to_numpy([])
                        key_dict[()][key_j] = exp_array
                second_derivative_mappings[key_i] = key_dicts_array
        else:
            second_derivative_mappings = None
        super().__init__(exp_mapping, parameters, name, derivative_mappings, second_derivative_mappings)
        self.__auto_init(locals())

    def __reduce__(self):
        return (ExpressionParameterFunctional,
                (self.expression, self.parameters, getattr(self, '_name', None),
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
        factors = tuple(factors)
        self.__auto_init(locals())

    def evaluate(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        return np.array([f.evaluate(mu) if hasattr(f, 'evaluate') else f for f in self.factors]).prod()

    def d_mu(self, parameter, index=0):
        summands = []
        for i, f in enumerate(self.factors):
            if hasattr(f, 'evaluate'):
                f_d_mu = f.d_mu(parameter, index)
                if isinstance(f_d_mu, ConstantParameterFunctional) and f_d_mu() == 0:
                    continue
            else:
                continue
            summands.append(
                ProductParameterFunctional([f_d_mu if j == i else g for j, g in enumerate(self.factors)])
            )
        if not summands:
            return ConstantParameterFunctional(0, name=f'{self.name}_d_{parameter}_{index}')
        else:
            return LincombParameterFunctional(summands, [1] * len(summands), name=f'{self.name}_d_{parameter}_{index}')


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

    def evaluate(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        return np.conj(self.functional.evaluate(mu))


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

    def d_mu(self, parameter, index=0):
        return self.with_(constant_value=0, name=f'{self.name}_d_{parameter}_{index}')


class LincombParameterFunctional(ParameterFunctional):
    """A |ParameterFunctional| representing a linear combination of |ParameterFunctionals|.

    The coefficients must be provided as scalars.

    Parameters
    ----------
    functionals
        List of |ParameterFunctionals| whose linear combination is formed.
    coefficients
        A list of scalar coefficients.
    name
        Name of the functional.

    Attributes
    ----------
    functionals
    coefficients
    """

    def __init__(self, functionals, coefficients, name=None):
        assert len(functionals) > 0
        assert len(functionals) == len(coefficients)
        assert all(isinstance(f, ParameterFunctional) for f in functionals)
        assert all(isinstance(c, Number) for c in coefficients)
        functionals = tuple(functionals)
        coefficients = tuple(coefficients)
        self.__auto_init(locals())

    def evaluate(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        return sum(c * f(mu) for c, f in zip(self.coefficients, self.functionals))

    def d_mu(self, parameter, index=0):
        functionals_d_mu = [f.d_mu(parameter, index) for f in self.functionals]
        return self.with_(functionals=functionals_d_mu, name=f'{self.name}_d_{parameter}_{index}')


class MinThetaParameterFunctional(ParameterFunctional):
    """|ParameterFunctional| implementing the min-theta approach from :cite:`Haa17` (Prop. 2.35).

    Let V denote a Hilbert space and let a: V x V -> K denote a parametric coercive bilinear form
    with affine decomposition ::

      a(u, v, mu) = sum_{q = 1}^Q theta_q(mu) a_q(u, v),

    for Q positive coefficient |ParameterFunctional| theta_1, ..., theta_Q and positive
    semi-definite component bilinear forms a_1, ..., a_Q: V x V -> K. Let mu_bar be a
    parameter with respect to which the coercivity constant
    of a(., ., mu_bar) is known, i.e. we known alpha_mu_bar > 0, s.t. ::

      alpha_mu_bar |u|_V^2 <= a(u, u, mu=mu_bar).

    The min-theta approach from :cite:`Haa17` (Proposition 2.35) allows to obtain a computable
    bound for the coercivity constant of a(., ., mu) for arbitrary parameters mu, since ::

      a(u, u, mu=mu) >= min_{q = 1}^Q theta_q(mu)/theta_q(mu_bar) a(u, u, mu=mu_bar).

    Given a list of the thetas, the |parameter values| mu_bar and the constant alpha_mu_bar, this
    functional thus evaluates to ::

      alpha_mu_bar * min_{q = 1}^Q theta_q(mu)/theta_q(mu_bar)


    Parameters
    ----------
    thetas
        List or tuple of |ParameterFunctional|
    mu_bar
        Parameter associated with alpha_mu_bar.
    alpha_mu_bar
        Known coercivity constant.
    name
        Name of the functional.
    """

    def __init__(self, thetas, mu_bar, alpha_mu_bar=1., name=None):
        assert isinstance(thetas, (list, tuple))
        assert len(thetas) > 0
        assert all([isinstance(theta, (Number, ParameterFunctional)) for theta in thetas])
        thetas = tuple(ConstantParameterFunctional(theta) if not isinstance(theta, ParameterFunctional) else theta
                       for theta in thetas)
        if not isinstance(mu_bar, Mu):
            mu_bar = Parameters.of(thetas).parse(mu_bar)
        assert Parameters.of(thetas).assert_compatible(mu_bar)
        thetas_mu_bar = np.array([theta(mu_bar) for theta in thetas])
        assert np.all(thetas_mu_bar > 0)
        assert isinstance(alpha_mu_bar, Number)
        assert alpha_mu_bar > 0
        self.__auto_init(locals())
        self.thetas_mu_bar = thetas_mu_bar

    def evaluate(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        thetas_mu = np.array([theta(mu) for theta in self.thetas])
        assert np.all(thetas_mu > 0)
        return self.alpha_mu_bar * np.min(thetas_mu / self.thetas_mu_bar)


class BaseMaxThetaParameterFunctional(ParameterFunctional):
    """Implements a generalization of the max-theta approach from :cite:`Haa17` (Exercise 5.12).

    Let V denote a Hilbert space and let a: V x V -> K denote a continuous bilinear form or
    l: V -> K a continuous linear functional, either with affine decomposition ::

      a(u, v, mu) = sum_{q = 1}^Q theta_q(mu) a_q(u, v)

    or ::

      l(v, mu) = sum_{q = 1}^Q theta_q(mu) l_q(v)

    for Q coefficient |ParameterFunctional| theta_1, ..., theta_Q and continuous bilinear forms
    a_1, ..., a_Q: V x V -> K or continuous linear functionals l_q: V -> K. Let mu_bar be a
    parameter with respect to which the continuity constant of a(., ., mu_bar) or l(., mu_bar)
    is known, i.e. we known gamma_mu_bar > 0, s.t. ::

      a(u, v, mu_bar) <= gamma_mu_bar |u|_V |v|_V

    or::

      l(v, mu_bar) <= gamma_mu_bar |v|_V.

    The max-theta approach (in its generalized form) from :cite:`Haa17` (Exercise 5.12) allows
    to obtain a computable bound for the continuity constant of another bilinear form
    a_prime(., ., mu) or linear form l_prime(., mu) with the same
    affine decomposition but different theta_prime_q for arbitrary parameters mu, since ::

      a_prime(u, v, mu=mu) <= |max_{q = 1}^Q theta_prime_q(mu)/theta_q(mu_bar)| |a(u, v, mu=mu_bar)|

    or ::

      l_prime(v, mu=mu) <= |max_{q = 1}^Q theta_prime_q(mu)/theta_q(mu_bar)| |l(v, mu=mu_bar)|,

    if all theta_q(mu_bar) != 0.

    Given a list of the thetas, the |parameter values| mu_bar and the constant gamma_mu_bar,
    this functional thus evaluates to ::

      |gamma_mu_bar * max_{q = 1}^Q theta_prime_q(mu)/theta_q(mu_bar)|

    Note that we also get an upper bound if theta_prime_q(mu) == 0 for any q. However, if
    theta_prime_q(mu) == 0 for at least one q, we need to use the absolute value in the denominator,
    i.e. ::

      |gamma_mu_bar * max_{q = 1}^Q theta_prime_q(mu)/|theta_q(mu_bar)| |

    Parameters
    ----------
    thetas
        List or tuple of |ParameterFunctional| of the base bilinear form a which is used for
        estimation.
    thetas_prime
        List or tuple of |ParameterFunctional| of the bilinear form a_prime for the numerator of the
        MaxThetaParameterFunctional.
    mu_bar
        Parameter associated with gamma_mu_bar.
    gamma_mu_bar
        Known continuity constant of the base bilinear form a.
    name
        Name of the functional.
    """

    def __init__(self, thetas_prime, thetas, mu_bar, gamma_mu_bar=1., name=None):
        assert isinstance(thetas_prime, (list, tuple))
        assert isinstance(thetas, (list, tuple))
        assert len(thetas) > 0
        assert len(thetas) == len(thetas_prime)
        assert all([isinstance(theta, (Number, ParameterFunctional)) for theta in thetas])
        assert all([isinstance(theta, (Number, ParameterFunctional)) for theta in thetas_prime])
        thetas = tuple(ConstantParameterFunctional(f) if not isinstance(f, ParameterFunctional) else f
                       for f in thetas)
        thetas_prime = tuple(ConstantParameterFunctional(f) if not isinstance(f, ParameterFunctional) else f
                             for f in thetas_prime)
        if not isinstance(mu_bar, Mu):
            mu_bar = Parameters.of(thetas).parse(mu_bar)
        assert Parameters.of(thetas).assert_compatible(mu_bar)
        thetas_mu_bar = np.array([theta(mu_bar) for theta in thetas])
        assert not np.any(float_cmp(thetas_mu_bar, 0))
        assert isinstance(gamma_mu_bar, Number)
        assert gamma_mu_bar > 0
        self.__auto_init(locals())
        self.thetas_mu_bar = thetas_mu_bar
        self.theta_mu_bar_has_negative = True if np.any(thetas_mu_bar < 0) else False
        if self.theta_mu_bar_has_negative:
            # If 0 is in theta_prime(mu), we need to use the absolute value to ensure
            # that the bound is still valid (and not zero)
            self.abs_thetas_mu_bar = np.array([np.abs(theta(mu_bar)) for theta in thetas])

    def evaluate(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        thetas_prime_mu = np.array([theta(mu) for theta in self.thetas_prime])
        if np.all(np.logical_or(thetas_prime_mu < 0, thetas_prime_mu > 0)) or not self.theta_mu_bar_has_negative:
            return self.gamma_mu_bar * np.abs(np.max(thetas_prime_mu / self.thetas_mu_bar))
        else:
            # special case
            return self.gamma_mu_bar * np.abs(np.max(thetas_prime_mu / self.abs_thetas_mu_bar))

    def d_mu(self, component, index=()):
        raise NotImplementedError


class MaxThetaParameterFunctional(BaseMaxThetaParameterFunctional):
    """|ParameterFunctional| implementing the max-theta approach from :cite:`Haa17` (Exercise 5.12).

    This is a specialized version of BaseMaxThetaParameterFunctional which allows to obtain a
    computable bound for the continuity constant of the actual a(., ., mu) or l(., mu) for
    arbitrary parameters mu, since ::

      a(u, v, mu=mu) <= |max_{q = 1}^Q theta_q(mu)/theta_q(mu_bar)|  |a(u, v, mu=mu_bar)|

    or ::

      l(v, mu=mu) <= |max_{q = 1}^Q theta_q(mu)/theta_q(mu_bar)| |l(v, mu=mu_bar)|,

    if all theta_q(mu_bar) != 0.

    Given a list of the thetas, the |parameter values| mu_bar and the constant gamma_mu_bar, this
    functional thus evaluates to ::

     |gamma_mu_bar * max{q = 1}^Q theta_q(mu)/theta_q(mu_bar)|

    Parameters
    ----------
    thetas
        List or tuple of |ParameterFunctional|
    mu_bar
        Parameter associated with gamma_mu_bar.
    gamma_mu_bar
        Known continuity constant.
    name
        Name of the functional.
    """

    def __init__(self, thetas, mu_bar, gamma_mu_bar=1., name=None):
        super().__init__(thetas, thetas, mu_bar, gamma_mu_bar, name)
