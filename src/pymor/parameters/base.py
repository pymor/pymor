# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module contains the implementation of pyMOR's parameter handling facilities.

A |Parameter| in pyMOR is basically a `dict` of |NumPy Arrays|. Each item of the
dict is called a parameter component. The |ParameterType| of a |Parameter| is the dict
of the shapes of the parameter components, i.e. ::

    mu.parameter_type['component'] == mu['component'].shape

Classes which represent mathematical objects depending on parameters, e.g. |Functions|,
|Operators|, |Models| derive from the |Parametric| mixin. Each |Parametric|
object has a :attr:`~Parametric.parameter_type` attribute holding the |ParameterType|
of the |Parameters| the object's `evaluate`, `apply`, `solve`, etc. methods expect.
Note that the |ParameterType| of the given |Parameter| is allowed to be a
superset of the object's |ParameterType|.

The |ParameterType| of a |Parametric| object is determined in its :meth:`__init__`
method by calling :meth:`~Parametric.build_parameter_type` which computes the
|ParameterType| as the union of the |ParameterTypes| of the objects given to the
method. This way, e.g., an |Operator| can inherit the |ParameterTypes| of the
data functions it depends upon.

A |Parametric| object can have a |ParameterSpace| assigned to it by setting the
:attr:`~Parametric.parameter_space` attribute (the |ParameterType| of the space
has to agree with the |ParameterType| of the object). The
:meth:`~Parametric.parse_parameter` method parses a user input according to
the object's |ParameterType| to make it a |Parameter| (e.g. if the |ParameterType|
consists of a single one-dimensional component, the user can simply supply a list
of numbers of the right length). Moreover, when given a |Parameter|,
:meth:`~Parametric.parse_parameter` ensures the |Parameter| has an appropriate
|ParameterType|.
"""

from collections import OrderedDict
from numbers import Number

import numpy as np

from pymor.tools.floatcmp import float_cmp_all
from pymor.tools.pprint import format_array


class Parameters(OrderedDict):
    """Class representing a parameter type.

    A parameter type is simply a dictionary with strings as keys and tuples of
    natural numbers as values. The keys are the names of the parameter components
    and the tuples their expected shape (compare :class:`Parameter`).

    Apart from checking the correct format of its values, the only difference
    between a |ParameterType| and an ordinary `dict` is, that |ParameterType|
    orders its keys alphabetically.

    Parameters
    ----------
    t
        If `t` is an object with a `parameter_type` attribute, a copy of this
        |ParameterType| is created. Otherwise, `t` can be anything from which
        a `dict` can be constructed.
    """

    def __init__(self, t):
        if t is None:
            t = {}
        elif isinstance(t, Parameters):
            pass
        elif hasattr(t, 'parameter_type'):
            assert isinstance(t.parameter_type, Parameters)
            t = t.parameter_type
        else:
            t = dict(t)
            for k, v in t.items():
                if not isinstance(v, tuple):
                    assert isinstance(v, Number)
                    t[k] = () if v == 0 else (v,)
        super().__init__(sorted(t.items()))
        self.clear = self.__setitem__ = self.__delitem__ = self.pop = self.popitem = self.update = self._is_immutable

    def _is_immutable(*args, **kwargs):
        raise ValueError('ParameterTypes cannot be modified')

    def copy(self):
        return Parameters(self)

    def fromkeys(self, S, v=None):
        raise NotImplementedError

    def __str__(self):
        return '{' + ', '.join(f'{k}: {v}' for k, v in self.items()) + '}'

    def __repr__(self):
        return 'ParameterType(' + str(self) + ')'

    def __le__(self, mu):
        if mu is not None and not isinstance(mu, Mu):
            raise TypeError('mu is not a Parameter. (Use parameter_type.parse?)')
        return not self or \
            mu is not None and all(getattr(mu.get(k), 'shape') == v for k, v in self.items())

    def why_incompatible(self, mu):
        if mu is not None and not isinstance(mu, Mu):
            return 'mu is not a Parameter. (Use parameter_type.parse?)'
        assert self
        if mu is None:
            mu = {}
        failing_components = {}
        for k, v in self.items():
            if k not in mu:
                failing_components[k] = f'missing != {v}'
            elif mu[k].shape != v:
                failing_components[k] = f'{mu[k].shape} != {v}'
        assert failing_components
        return f'Incompatible components: {failing_components}'

    def parse(self, mu):
        """Takes a user input `mu` and interprets it as a |Parameter| according to the given
        |ParameterType|.

        Depending on the |ParameterType|, `mu` can be given as a |Parameter|, dict, tuple,
        list, array or scalar.

        Parameters
        ----------
        mu
            The user input which shall be interpreted as a |Parameter|.

        Returns
        -------
        The resulting |Parameter|.

        Raises
        ------
        ValueError
            Is raised if `mu` cannot be interpreted as a |Parameter| of |ParameterType|
            `parameter_type`.
        """
        if not self:
            assert mu is None or mu == {}
            return Mu({})

        if isinstance(mu, Mu):
            assert mu.parameter_type == self
            return mu

        if not isinstance(mu, dict):
            if isinstance(mu, (tuple, list)):
                if len(self) == 1 and len(mu) != 1:
                    mu = (mu,)
            else:
                mu = (mu,)
            if len(mu) != len(self):
                raise ValueError('Parameter length does not match.')
            mu = dict(zip(sorted(self), mu))
        elif set(mu.keys()) != set(self.keys()):
            raise ValueError(f'Provided parameter with keys {list(mu.keys())} does not match '
                             f'parameter type {self}.')

        def parse_value(k, v):
            if not isinstance(v, np.ndarray):
                v = np.array(v)
                try:
                    v = v.reshape(self[k])
                except ValueError:
                    raise ValueError(f'Shape mismatch for parameter component {k}: got {v.shape}, '
                                     f'expected {self[k]}')
            if v.shape != self[k]:
                raise ValueError(f'Shape mismatch for parameter component {k}: got {v.shape}, '
                                 f'expected {self[k]}')
            return v

        return Mu({k: parse_value(k, v) for k, v in mu.items()})

    def __reduce__(self):
        return (Parameters, (dict(self),))

    def __hash__(self):
        return hash(tuple(self.items()))


class Mu(dict):
    """Class representing a parameter.

    A |Parameter| is simply a `dict` where each key is a string and each value
    is a |NumPy array|. We call an item of the dictionary a *parameter component*.

    A |Parameter| differs from an ordinary `dict` in the following ways:

        - It is ensured that each value is a |NumPy array|.
        - We overwrite :meth:`copy` to ensure that not only the `dict`
          but also the |NumPy arrays| are copied.
        - The :meth:`allclose` method allows to compare |Parameters| for
          equality in a mathematically meaningful way.
        - We override :meth:`__str__` to ensure alphanumerical ordering of the keys
          and pretty printing of the values.
        - The :attr:`parameter_type` property can be used to obtain the |ParameterType|
          of the parameter.
        - Use :meth:`from_parameter_type` to construct a |Parameter| from a |ParameterType|
          and user supplied input.

    Parameters
    ----------
    v
        Anything that :class:`dict` accepts for the construction of a dictionary.

    Attributes
    ----------
    parameter_type
        The |ParameterType| of the |Parameter|.
    """

    def __init__(self, v):
        if v is None:
            v = {}
        i = iter(v.items()) if hasattr(v, 'items') else v
        dict.__init__(self, {k: np.array(v) if not isinstance(v, np.ndarray) else v for k, v in i})

    def allclose(self, mu):
        """Compare two |Parameters| using :meth:`~pymor.tools.floatcmp.float_cmp_all`.

        Parameters
        ----------
        mu
            The |Parameter| with which to compare.

        Returns
        -------
        `True` if both |Parameters| have the same |ParameterType| and all parameter
        components are almost equal, else `False`.
        """
        assert isinstance(mu, Mu)
        if list(self.keys()) != list(mu.keys()):
            return False
        elif not all(float_cmp_all(v, mu[k]) for k, v in self.items()):
            return False
        else:
            return True

    def copy(self):
        c = Mu({k: v.copy() for k, v in self.items()})
        return c

    def __setitem__(self, key, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        dict.__setitem__(self, key, value)

    def __eq__(self, mu):
        if not isinstance(mu, Mu):
            mu = Mu(mu)
        if list(self.keys()) != list(mu.keys()):
            return False
        elif not all(np.array_equal(v, mu[k]) for k, v in self.items()):
            return False
        else:
            return True

    def fromkeys(self, S, v=None):
        raise NotImplementedError

    def pop(self, k, d=None):
        raise NotImplementedError

    def popitem(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def parameter_type(self):
        return Parameters({k: v.shape for k, v in self.items()})

    def __str__(self):
        np.set_string_function(format_array, repr=False)
        s = '{'
        for k in sorted(self.keys()):
            v = self[k]
            if v.ndim > 1:
                v = v.ravel()
            if s == '{':
                s += f'{k}: {v}'
            else:
                s += f', {k}: {v}'
        s += '}'
        np.set_string_function(None, repr=False)
        return s

    def __getstate__(self):
        return dict(self)


class Parametric:
    """Mixin class for objects representing mathematical entities depending on a |Parameter|.

    Each such object has a |ParameterType| stored in the :attr:`parameter_type` attribute,
    which should be set by the implementor during :meth:`__init__` using the
    :meth:`build_parameter_type` method. Methods expecting the |Parameter| (typically
    `evaluate`, `apply`, `solve`, etc. ..) should accept an optional argument `mu` defaulting
    to `None`. This argument `mu` should then be fed into :meth:`parse_parameter` to obtain a
    |Parameter| of correct |ParameterType| from the (user supplied) input `mu`.

    Attributes
    ----------
    parameter_type
        The |ParameterType| of the |Parameters| the object expects.
    parameter_space
        |ParameterSpace| the parameters are expected to lie in or `None`.
    parametric:
        `True` if the object really depends on a parameter, i.e. :attr:`parameter_type`
        is not empty.
    """

    parameter_type = Parameters({})

    @property
    def parameter_space(self):
        return self._parameter_space

    @parameter_space.setter
    def parameter_space(self, ps):
        assert ps is None or self.parameter_type == ps.parameter_type
        self._parameter_space = ps

    @property
    def parametric(self):
        return bool(self.parameter_type)

    def build_parameter_type(self, *args, provides=None, **kwargs):
        """Builds the |ParameterType| of the object. Should be called by :meth:`__init__`.

        The |ParameterType| of a |Parametric| object is determined by the parameter components
        the object itself requires for evaluation, and by the parameter components
        required by the objects the object depends upon for evaluation.

        All parameter components (directly specified or inherited by the |ParameterType|
        of a given |Parametric| object) with the same name are treated as identical and
        are thus required to have the same shapes. The object's |ParameterType| is then
        made up by the shapes of all parameter components appearing.

        Additionally components of the resulting |ParameterType| can be removed by
        specifying them via the `provides` parameter. The idea is that the object itself
        may provide parameter components to the inherited objects which thus should
        not become part of the object's own parameter type. (A typical application
        would be |InstationaryModel|, which provides a time parameter
        component to its (time-dependent) operators during time-stepping.)

        Parameters
        ----------
        args
            Each positional argument must either be a dict of parameter components and shapes or
            a |Parametric| object whose :attr:`~Parametric.parameter_type` is added.
        kwargs
            Each keyword argument is interpreted as parameter component with corresponding shape.
        provides
            `Dict` of parameter component names and shapes which are provided by the object
            itself. The parameter components listed here will not become part of the object's
            |ParameterType|.
        """
        provides = provides or {}
        my_parameter_type = {}

        def check_shapes(shape1, shape2):
            if type(shape1) is not tuple:
                assert isinstance(shape1, Number)
                shape1 = () if shape1 == 0 else (shape1,)
            if type(shape2) is not tuple:
                assert isinstance(shape2, Number)
                shape2 = () if shape2 == 0 else (shape2,)
            assert shape1 == shape2, \
                (f'Dimension mismatch for parameter component {component} '
                 f'(got {my_parameter_type[component]} and {shape})')
            return True

        for arg in args:
            if hasattr(arg, 'parameter_type'):
                arg = arg.parameter_type
            if arg is None:
                continue
            for component, shape in arg.items():
                assert component not in my_parameter_type or check_shapes(my_parameter_type[component], shape)
                my_parameter_type[component] = shape

        for component, shape in kwargs.items():
            assert component not in my_parameter_type or check_shapes(my_parameter_type[component], shape)
            my_parameter_type[component] = shape

        for component, shape in provides.items():
            assert component not in my_parameter_type or check_shapes(my_parameter_type[component], shape)
            my_parameter_type.pop(component, None)

        self.parameter_type = Parameters(my_parameter_type)
