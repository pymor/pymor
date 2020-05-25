# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module contains the implementation of pyMOR's parameter handling facilities.

A |Parameter| in pyMOR is basically a `dict` of |NumPy Arrays|. Each item of the
dict is called a parameter component. The |ParameterType| of a |Parameter| is the dict
of the shapes of the parameter components, i.e. ::

    mu.parameters['component'] == mu['component'].shape

Classes which represent mathematical objects depending on parameters, e.g. |Functions|,
|Operators|, |Models| derive from the |Parametric| mixin. Each |Parametric|
object has a :attr:`~Parametric.parameters` attribute holding the |ParameterType|
of the |Parameters| the object's `evaluate`, `apply`, `solve`, etc. methods expect.
Note that the |ParameterType| of the given |Parameter| is allowed to be a
superset of the object's |ParameterType|.

The |ParameterType| of a |Parametric| object is determined in its :meth:`__init__`
method by calling :meth:`~Parametric.build_parameter_type` which computes the
|ParameterType| as the union of the |ParameterTypes| of the objects given to the
method. This way, e.g., an |Operator| can inherit the |ParameterTypes| of the
data functions it depends upon.

The :meth:`~Parametric.parse_parameter` method parses a user input according to
the object's |ParameterType| to make it a |Parameter| (e.g. if the |ParameterType|
consists of a single one-dimensional component, the user can simply supply a list
of numbers of the right length). Moreover, when given a |Parameter|,
:meth:`~Parametric.parse_parameter` ensures the |Parameter| has an appropriate
|ParameterType|.
"""

from itertools import product
from numbers import Number

import numpy as np

from pymor.core.base import ImmutableObject
from pymor.tools.floatcmp import float_cmp_all
from pymor.tools.frozendict import FrozenDict
from pymor.tools.pprint import format_array
from pymor.tools.random import get_random_state


class Parameters(FrozenDict):
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
        If `t` is an object with a `parameters` attribute, a copy of this
        |ParameterType| is created. Otherwise, `t` can be anything from which
        a `dict` can be constructed.
    """

    def _post_init(self):
        assert all(type(k) is str and type(v) is int and 0 <= v
                   for k, v in self.items())

    def __str__(self):
        return '{' + ', '.join(f'{k}: {v}' for k, v in sorted(self.items())) + '}'

    def __repr__(self):
        return 'Parameters(' + str(self) + ')'

    def __le__(self, mu):
        if isinstance(mu, Parameters):
            return all(mu.get(k) == v for k, v in self.items())
        else:
            if mu is not None and not isinstance(mu, Mu):
                raise TypeError('mu is not a Parameter. (Use parameters.parse?)')
            return not self or \
                mu is not None and all(getattr(mu.get(k), 'size', None) == v for k, v in self.items())

    def assert_compatible(self, mu):
        assert mu >= self, self.why_incompatible(mu)
        return True

    def why_incompatible(self, mu):
        if mu is not None and not isinstance(mu, Mu):
            return 'mu is not a Parameter. (Use parameters.parse?)'
        assert self
        if mu is None:
            mu = {}
        failing_components = {}
        for k, v in self.items():
            if k not in mu:
                failing_components[k] = f'missing != {v}'
            elif mu[k].shape != v:
                failing_components[k] = f'{mu[k].size} != {v}'
        assert failing_components
        return f'Incompatible components: {failing_components}'

    @classmethod
    def of(cls, *args):
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
            a |Parametric| object whose :attr:`~Parametric.parameters` is added.
        """
        parameters = {}

        def check_shapes(component, shape1, shape2):
            assert isinstance(shape2, int) and shape2 >= 0, f'Component {component} not an int or negative'
            assert shape1 is None or shape1 == shape2, \
                f'Dimension mismatch for parameter {component} (got {shape1} and {shape2})'
            return True

        def traverse(obj):
            if obj is None:
                return
            elif isinstance(obj, ParametricObject):
                assert all(check_shapes(component, parameters.get(component), shape)
                           for component, shape in obj.parameters.items())
                parameters.update(obj.parameters)
            elif isinstance(obj, (list, tuple)):
                for o in obj:
                    traverse(o)
            elif isinstance(obj, (dict, FrozenDict)):
                for o in obj.values():
                    traverse(o)
            elif isinstance(obj, np.ndarray) and obj.dtype == object:
                for o in obj.flat:
                    traverse(o)

        for arg in args:
            traverse(arg)

        return cls(parameters)

    def __or__(self, other):
        assert all(k not in self or self[k] == v
                   for k, v in other.items())
        return Parameters(dict(self, **other))

    def __sub__(self, other):
        assert all(k not in self or self[k] == v
                   for k, v in other.items())
        return Parameters({k: v for k, v in self.items() if k not in other})

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
            `parameters`.
        """

        def fail(msg):
            if isinstance(mu, dict):
                mu_str = '{' + ', '.join([f'{k}: {v}' for k, v in mu.items()]) + '}'
            else:
                mu_str = str(mu)
            raise ValueError(f'{mu_str} is incompatible with Parameters {self} ({msg})')

        if not self:
            mu is None or mu == {} or fail('must be None or empty dict')
            return Mu({})

        elif isinstance(mu, Mu):
            mu == self or fail(self.why_incompatible(mu))
            set(mu) == set(self) or fail(f'additional parameters {set(mu) - set(self)}')
            return mu

        elif isinstance(mu, Number):
            1 == sum(v for v in self.values()) or fail('need more than one number')
            return Mu({next(iter(self)): np.array([mu])})

        elif isinstance(mu, (tuple, list, np.ndarray)):
            if isinstance(mu, np.ndarray):
                mu = mu.ravel()
            all(isinstance(v, Number) for v in mu) or fail('not every element a number')
            len(mu) == sum(v for v in self.values()) or fail('wrong size')
            parsed_mu = {}
            for k, v in self.items():
                p, mu = mu[:v], mu[v:]
                parsed_mu[k] = p
            return Mu(parsed_mu)

        elif isinstance(mu, dict):
            set(mu.keys()) == set(self.keys()) or fail('components not matching')

            def parse_value(k, v):
                isinstance(v, (Number, tuple, list, np.ndarray)) or fail(f"invalid type '{type(v)}' of parameter {k}")
                if isinstance(v, Number):
                    v = np.array([v])
                elif isinstance(v, np.ndarray):
                    v = v.ravel()
                len(v) == self[k] or fail('wrong size of parameter {k}')
                return v

            return Mu({k: parse_value(k, v) for k, v in mu.items()})

    def space(self, *ranges):
        return ParameterSpace(self, *ranges)

    def __hash__(self):
        return hash(tuple(self.items()))


class Mu(FrozenDict):
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
        - The :attr:`parameters` property can be used to obtain the |ParameterType|
          of the parameter.
        - Use :meth:`from_parameter_type` to construct a |Parameter| from a |ParameterType|
          and user supplied input.

    Parameters
    ----------
    v
        Anything that :class:`dict` accepts for the construction of a dictionary.

    Attributes
    ----------
    parameters
        The |ParameterType| of the |Parameter|.
    """

    def __new__(cls, *args, **kwargs):
        mu = super().__new__(cls,
                             ((k, np.array(v, copy=False, ndmin=1))
                              for k, v in dict(*args, **kwargs).items()))
        assert all(v.ndim == 1 for v in mu.values())
        # only make elements immutable when running without optimization
        assert not any(v.setflags(write=False) for v in mu.values())
        return mu

    def with_(self, **kwargs):
        return Mu(self, **kwargs)

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
        return self.keys() == mu.keys() and all(float_cmp_all(v, mu[k]) for k, v in self.items())

    def copy(self):
        return self

    def __eq__(self, mu):
        if not isinstance(mu, Mu):
            try:
                mu = Mu(mu)
            except:
                return False
        return self.keys() == mu.keys() and all(np.array_equal(v, mu[k]) for k, v in self.items())

    @property
    def parameters(self):
        return Parameters({k: v.size for k, v in self.items()})

    def __str__(self):
        return '{' + ', '.join(f'{k}: {format_array(v)}' for k, v in sorted(self.items())) + '}'

    def __repr__(self):
        return f'Mu({self})'


class ParametricObject(ImmutableObject):
    """Mixin class for objects representing mathematical entities depending on a |Parameter|.

    Each such object has a |ParameterType| stored in the :attr:`parameters` attribute,
    which should be set by the implementor during :meth:`__init__` using the
    :meth:`build_parameter_type` method. Methods expecting the |Parameter| (typically
    `evaluate`, `apply`, `solve`, etc. ..) should accept an optional argument `mu` defaulting
    to `None`. This argument `mu` should then be fed into :meth:`parse_parameter` to obtain a
    |Parameter| of correct |ParameterType| from the (user supplied) input `mu`.

    Attributes
    ----------
    parameters
        The |ParameterType| of the |Parameters| the object expects.
    parametric:
        `True` if the object really depends on a parameter, i.e. :attr:`parameters`
        is not empty.
    """

    @property
    def parameters(self):
        if self._parameters is not None:
            return self._parameters
        assert self._locked, 'parameters attribute can only be accessed after class initialization'
        params = Parameters.of(*(getattr(self, arg) for arg in self._init_arguments))
        if self.parameters_own:
            params = params | self.parameters_own
        if self.parameters_internal:
            params = params - self.parameters_internal
        self._parameters = params
        return params

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = Parameters(parameters)
        assert self.__check_parameter_consistency()

    @property
    def parameters_own(self):
        return self._parameters_own or Parameters({})

    @parameters_own.setter
    def parameters_own(self, parameters_own):
        self._parameters_own = Parameters(parameters_own)
        assert self.__check_parameter_consistency()

    @property
    def parameters_internal(self):
        return self._parameters_internal or Parameters({})

    @parameters_internal.setter
    def parameters_internal(self, parameters_internal):
        self._parameters_internal = Parameters(parameters_internal)
        assert self.__check_parameter_consistency()

    @property
    def parameters_inherited(self):
        return self.parameters - self.parameters_own

    @property
    def parametric(self):
        return bool(self.parameters)

    def __check_parameter_consistency(self):
        if self._parameters_internal is not None:
            if self._parameters is not None:
                assert self._parameters.keys().isdisjoint(self._parameters_internal)
            if self._parameters_own is not None:
                assert self._parameters_own.keys().isdisjoint(self._parameters_internal)
        if self._parameters_own is not None:
            if self._parameters is not None:
                assert self._parameters >= self._parameters_own
        return True

    _parameters = None
    _parameters_own = None
    _parameters_internal = None


class ParameterSpace(ParametricObject):

    def __init__(self, parameters, *ranges):
        assert isinstance(parameters, Parameters)
        assert 1 <= len(ranges) <= 2
        if len(ranges) == 1:
            ranges = ranges[0]
        if isinstance(ranges, (tuple, list)):
            assert len(ranges) == 2
            ranges = {k: ranges for k in parameters}
        assert isinstance(ranges, dict)
        assert all(k in ranges
                   and len(ranges[k]) == 2
                   and all(isinstance(v, Number) for v in ranges[k])
                   and ranges[k][0] <= ranges[k][1]
                   for k in parameters)
        self.parameters = parameters
        self.ranges = FrozenDict((k, tuple(v)) for k, v in ranges.items())

    def sample_uniformly(self, counts):
        """Uniformly sample |Parameters| from the space."""
        if isinstance(counts, dict):
            pass
        elif isinstance(counts, (tuple, list, np.ndarray)):
            counts = {k: c for k, c in zip(self.parameters, counts)}
        else:
            counts = {k: counts for k in self.parameters}

        linspaces = tuple(np.linspace(self.ranges[k][0], self.ranges[k][1], num=counts[k]) for k in self.parameters)
        iters = tuple(product(ls, repeat=max(1, np.zeros(sps).size))
                      for ls, sps in zip(linspaces, self.parameters.values()))
        return [Mu((k, np.array(v)) for k, v in zip(self.parameters, i))
                for i in product(*iters)]

    def sample_randomly(self, count=None, random_state=None, seed=None):
        """Randomly sample |Parameters| from the space.

        Parameters
        ----------
        count
            `None` or number of random parameters (see below).
        random_state
            :class:`~numpy.random.RandomState` to use for sampling.
            If `None`, a new random state is generated using `seed`
            as random seed, or the :func:`default <pymor.tools.random.default_random_state>`
            random state is used.
        seed
            If not `None`, a new radom state with this seed is used.

        Returns
        -------
        If `count` is `None`, an inexhaustible iterator returning random
        |Parameters|.
        Otherwise a list of `count` random |Parameters|.
        """
        assert not random_state or seed is None
        random_state = get_random_state(random_state, seed)
        get_param = lambda: Mu(((k, random_state.uniform(self.ranges[k][0], self.ranges[k][1], size))
                               for k, size in self.parameters.items()))
        if count is None:
            def param_generator():
                while True:
                    yield get_param()
            return param_generator()
        else:
            return [get_param() for _ in range(count)]

    def contains(self, mu):
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        if not mu >= self.parameters:
            return False
        return all(np.all(self.ranges[k][0] <= mu[k]) and np.all(mu[k] <= self.ranges[k][1])
                   for k in self.parameters)
