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


class ParameterType(OrderedDict):
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
        elif isinstance(t, ParameterType):
            pass
        elif hasattr(t, 'parameter_type'):
            assert isinstance(t.parameter_type, ParameterType)
            t = t.parameter_type
        else:
            t = dict(t)
            for k, v in t.items():
                if not isinstance(v, tuple):
                    assert isinstance(v, Number)
                    t[k] = () if v == 0 else (v,)
        super().__init__(sorted(t.items()))
        self.clear = self.__setitem__ = self.__delitem__ = self.pop = self.popitem = self.update = self._is_immutable

    def parse_parameter(self, mu):
        """Interpret a user supplied parameter `mu` as a |Parameter|.

        If `mu` is not already a |Parameter|, :meth:`Parameter.from_parameter_type`
        is used, to make `mu` a parameter of the correct |ParameterType|. If `mu`
        is already a |Parameter|, it is checked if its |ParameterType| matches our own.
        (It is actually allowed that the |ParameterType| of `mu` is a superset of
        our own |ParameterType| in the obvious sense.)

        Parameters
        ----------
        mu
            The input to parse as a |Parameter|.
        """
        if mu is None:
            assert not self, \
                f'Given parameter is None but expected parameter of type {self}'
            return {}

        if not self:
            assert mu is None or mu == {}
            return {}

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

        return {k: parse_value(k, v) for k, v in mu.items()}

    def _is_immutable(*args, **kwargs):
        raise ValueError('ParameterTypes cannot be modified')

    def copy(self):
        return ParameterType(self)

    def fromkeys(self, S, v=None):
        raise NotImplementedError

    def __str__(self):
        return '{' + ', '.join(f'{k}: {v}' for k, v in self.items()) + '}'

    def __repr__(self):
        return 'ParameterType(' + str(self) + ')'

    def __le__(self, mu):

        def check_component(c, shape):
            if len(shape) == 0:
                return isinstance(c, Number) or isinstance(c, np.ndarray) and c.shape == ()
            elif len(shape) == 1:
                return isinstance(c, (list, tuple, np.ndarray)) and len(c) == shape[0]
            else:
                return isinstance(c, np.ndarray) and c.shape == shape

        return not self or \
            isinstance(mu, dict) and all(check_component(mu.get(k), v) for k, v in self.items())

    def __reduce__(self):
        return (ParameterType, (dict(self),))

    def __hash__(self):
        return hash(tuple(self.items()))


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

    parameter_type = None

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

    def parse_parameter(self, mu):
        """Interpret a user supplied parameter `mu` as a |Parameter|.

        If `mu` is not already a |Parameter|, :meth:`Parameter.from_parameter_type`
        is used, to make `mu` a parameter of the correct |ParameterType|. If `mu`
        is already a |Parameter|, it is checked if its |ParameterType| matches our own.
        (It is actually allowed that the |ParameterType| of `mu` is a superset of
        our own |ParameterType| in the obvious sense.)

        Parameters
        ----------
        mu
            The input to parse as a |Parameter|.
        """
        parameter_type = self.parameter_type or ParameterType({})
        if not isinstance(mu, dict):
            return parameter_type.parse_parameter(mu)
        assert mu >= parameter_type
        return mu

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

        self.parameter_type = ParameterType(my_parameter_type)
