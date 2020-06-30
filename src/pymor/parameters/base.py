# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module contains the implementation of pyMOR's parameter handling facilities.

Use the |ParametricObject| base class to define immutable (mathematical) objects that
depend on some |Parameters|. Each |Parameter| in pyMOR has a name and a fixed dimension
(number of scalar components of the parameter vector). In particular, scalar parameters
are treated as parameter vectors of dimension 1. Mappings of |Parameters| to
|parameter values| are stored in pyMOR using dedicated :class:`Mu` objects.
To sample |parameter values| within a given range, |ParameterSpace| objects can be used.
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
    """Immutable dict mapping parameter names to parameter dimensions.

    Each key of a |Parameters| dict is a string specifying the
    name of a parameter. The corresponding value is a non-negative `int`
    specifying the dimension (number of scalar components) of the parameter.
    """

    def _post_init(self):
        assert all(type(k) is str and type(v) is int and 0 <= v
                   for k, v in self.items())

    @classmethod
    def of(cls, *args):
        """Computes the total set of |Parameters| a collection of objects depends on.

        If two objects depend on a parameter with the same name, both parameters must have
        the same dimension.

        Parameters
        ----------
        args
            Each positional argument must either be `None`, a |ParametricObject| or
            lists, tuples, dicts or |NumPy arrays| of such objects. The latter will be
            traversed recursively.
        """
        parameters = {}

        def check_dims(param, dim1, dim2):
            assert isinstance(dim2, int) and dim2 >= 0, f'Dimension of parameter {param} not an int or negative'
            assert dim1 is None or dim1 == dim2, \
                f'Dimension mismatch for parameter {param} (got {dim1} and {dim2})'
            return True

        def traverse(obj):
            if obj is None:
                return
            elif isinstance(obj, ParametricObject):
                assert all(check_dims(param, parameters.get(param), dim)
                           for param, dim in obj.parameters.items())
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

    @property
    def dim(self):
        """The sum of the dimensions of all parameters."""
        return sum(self.values())

    def parse(self, mu):
        """Takes a user input `mu` and interprets it as set of |parameter values|
        according to the given |Parameters|.

        Depending on the |Parameters|, `mu` can be given as a dict, list,
        tuple, |NumPy array| or scalar. In the latter cases, multiple parameters
        will be concatenated by alphabetical ordering. E.g.::

            Parameters(b=2, a=1).parse([1,2,3])

        will assign to parameter `a` the value `[1]` and to parameter `b` the
        values `[2, 3]`.

        Parameters
        ----------
        mu
            The user input which shall be interpreted as |parameter values|.

        Returns
        -------
        The resulting object of |parameter values|.

        Raises
        ------
        ValueError
            Is raised if `mu` cannot be interpreted as |parameter values| for the
            given |Parameters|.
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
            for k, v in sorted(self.items()):
                p, mu = mu[:v], mu[v:]
                parsed_mu[k] = p
            return Mu(parsed_mu)

        elif isinstance(mu, dict):
            set(mu.keys()) == set(self.keys()) or fail('parameters not matching')

            def parse_value(k, v):
                isinstance(v, (Number, tuple, list, np.ndarray)) or fail(f"invalid value type '{type(v)}' for parameter {k}")
                if isinstance(v, Number):
                    v = np.array([v])
                elif isinstance(v, np.ndarray):
                    v = v.ravel()
                len(v) == self[k] or fail('wrong dimension of parameter value {k}')
                return v

            return Mu({k: parse_value(k, v) for k, v in mu.items()})

    def space(self, *ranges):
        """Create a |ParameterSpace| with given ranges.

        This is a shorthand for ::

            ParameterSpace(self, *range)

        See |ParameterSpace| for allowed range arguments.
        """
        return ParameterSpace(self, *ranges)

    def assert_compatible(self, mu):
        """Assert that |parameter values| are compatible with the given |Parameters|.

        Each of the parameter must be contained in  `mu` and the dimensions have to match,
        i.e. ::

            mu[parameter].size == self[parameter]

        Otherwise, an `AssertionError` will be raised.
        """
        assert self.is_compatible(mu), self.why_incompatible(mu)
        return True

    def is_compatible(self, mu):
        """Check if |parameter values| are compatible with the given |Parameters|.

        Each of the parameter must be contained in  `mu` and the dimensions have to match,
        i.e. ::

            mu[parameter].size == self[parameter]
        """
        if mu is not None and not isinstance(mu, Mu):
            raise TypeError('mu is not a Mu instance. (Use parameters.parse?)')
        return not self or \
            mu is not None and all(getattr(mu.get(k), 'size', None) == v for k, v in self.items())

    def why_incompatible(self, mu):
        if mu is not None and not isinstance(mu, Mu):
            return 'mu is not a Mu instance. (Use parameters.parse?)'
        assert self
        if mu is None:
            mu = {}
        failing_params = {}
        for k, v in self.items():
            if k not in mu:
                failing_params[k] = f'missing != {v}'
            elif mu[k].shape != v:
                failing_params[k] = f'{mu[k].size} != {v}'
        assert failing_params
        return f'Incompatible parameters: {failing_params}'

    def __or__(self, other):
        assert all(k not in self or self[k] == v
                   for k, v in other.items())
        return Parameters(dict(self, **other))

    def __sub__(self, other):
        assert all(k not in self or self[k] == v
                   for k, v in other.items())
        return Parameters({k: v for k, v in self.items() if k not in other})

    def __le__(self, mu):
        """Check if |parameter values| are compatible with the given |Parameters|.

        Each of the parameter must be contained in  `mu` and the dimensions have to match,
        i.e. ::

            mu[parameter].size == self[parameter]
        """
        if isinstance(mu, Parameters):
            return all(mu.get(k) == v for k, v in self.items())
        else:
            return NotImplemented

    def __str__(self):
        return '{' + ', '.join(f'{k}: {v}' for k, v in sorted(self.items())) + '}'

    def __repr__(self):
        return 'Parameters(' + str(self) + ')'

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class Mu(FrozenDict):
    """Immutable mapping of |Parameter| names to parameter values.

    Parameters
    ----------
    Anything that dict accepts for the construction of a dictionary.
    Values are automatically converted to immutable one-dimensional |NumPy arrays|,
    unless the Python interpreter runs with the `-O` flag.

    Attributes
    ----------
    parameters
        The |Parameters| to which the mapping assigns values.
    """

    def __new__(cls, *args, **kwargs):
        mu = super().__new__(cls,
                             ((k, np.array(v, copy=False, ndmin=1))
                              for k, v in dict(*args, **kwargs).items()))
        assert all(type(k) is str and v.ndim == 1 for k, v in mu.items())
        # only make elements immutable when running without optimization
        assert not any(v.setflags(write=False) for v in mu.values())
        return mu

    def with_(self, **kwargs):
        return Mu(self, **kwargs)

    @property
    def parameters(self):
        return Parameters({k: v.size for k, v in self.items()})

    def allclose(self, mu):
        """Compare two dicts of |parameter values| using :meth:`~pymor.tools.floatcmp.float_cmp_all`.

        Parameters
        ----------
        mu
            The |parameter values| with which to compare.

        Returns
        -------
        `True` if both |parameter value| dicts contain values for the same |Parameters| and all
        components of the parameter values are almost equal, else `False`.
        """
        assert isinstance(mu, Mu)
        return self.keys() == mu.keys() and all(float_cmp_all(v, mu[k]) for k, v in self.items())

    def to_numpy(self):
        """All parameter values as a NumPy array, ordered alphabetically."""
        return np.hstack([v for k, v in sorted(self.items())])

    def copy(self):
        return self

    def __eq__(self, mu):
        if not isinstance(mu, Mu):
            try:
                mu = Mu(mu)
            except Exception:
                return False
        return self.keys() == mu.keys() and all(np.array_equal(v, mu[k]) for k, v in self.items())

    def __str__(self):
        return '{' + ', '.join(f'{k}: {format_array(v)}' for k, v in sorted(self.items())) + '}'

    def __repr__(self):
        return f'Mu({self})'


class ParametricObject(ImmutableObject):
    """Base class for immutable mathematical entities depending on some |Parameters|.

    Each |ParametricObject| lists the |Parameters| it depends on in the :attr:`parameters`
    attribute. Usually, these |Parameters| are automatically derived as the union of all
    |Parameters| of the object's `__init__` arguments.

    Additional |Parameters| introduced by the object itself can be specified by setting the
    :attr:`parameters_own` attribute in `__init__`. In case the object fixes some |Parameters|
    it's child objects depend on to concrete values, those |Parameters| can be removed from
    the :attr:`parameters` attribute by setting :attr:`parameters_internal`.

    Alternatively, :attr:`parameters` can be initialized manually in `__init__`.

    Attributes
    ----------
    parameters
        The |Parameters| the object depends on.
    parameters_own
        The |Parameters| the object depends on which are not inherited from a child
        object the object depends on. Each item of :attr:`parameters_own` is also an
        item of :attr:`parameters`.
    parameters_inherited
        The |Parameters| the object depends on because some child object depends on them.
        Each item of :attr:`parameters_own` is also an item of :attr:`parameters`.
    parameters_internal
        The |Parameters| some of the object's child objects may depend on, but which are
        fixed to a concrete value by this object. All items of :attr:`parameters_internal`
        are removed from :attr:`parameters` and :attr:`parameters_inherited`. When
        initializing :attr:`parameters_own` and :attr:`parameters_internal`, it has to be
        ensured that both dicts are disjoint.
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
    """A set of |Parameters| with allowed ranges for their values.

    |ParameterSpaces| are mostly used to create sample set of
    |parameter values| for given |Parameters| within a specified
    range.

    Parameters
    ----------
    parameters
        The |Parameters| which are part of the space.
    ranges
        Allowed ranges for the |parameter values|. Either:

        - two numbers specifying the lower and upper bound
          for all parameter value components,
        - a list/tuple of two numbers specifying these bounds,
        - or a dict of those tuples, specifying upper and lower
          bounds individually for each parameter of the space.
    """

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
        """Uniformly sample |parameter values| from the space.

        Parameters
        ----------
        counts
            Number of samples to take per parameter and component
            of the parameter. Either a dict of counts per |Parameter|
            or a single count that is taken for all |Parameters|

        Returns
        -------
        List of |parameter value| dicts.
        """
        if isinstance(counts, dict):
            pass
        else:
            counts = {k: counts for k in self.parameters}

        linspaces = tuple(np.linspace(self.ranges[k][0], self.ranges[k][1], num=counts[k]) for k in self.parameters)
        iters = tuple(product(ls, repeat=max(1, np.zeros(sps).size))
                      for ls, sps in zip(linspaces, self.parameters.values()))
        return [Mu((k, np.array(v)) for k, v in zip(self.parameters, i))
                for i in product(*iters)]

    def sample_randomly(self, count=None, random_state=None, seed=None):
        """Randomly sample |parameter values| from the space.

        Parameters
        ----------
        count
            `None` or number of random samples (see below).
        random_state
            :class:`~numpy.random.RandomState` to use for sampling.
            If `None`, a new random state is generated using `seed`
            as random seed, or the :func:`default <pymor.tools.random.default_random_state>`
            random state is used.
        seed
            If not `None`, a new random state with this seed is used.

        Returns
        -------
        If `count` is `None`, an inexhaustible iterator returning random
        |parameter value| dicts.
        Otherwise a list of `count` random |parameter value| dicts.
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
        if not self.parameters.is_compatible(mu):
            return False
        return all(np.all(self.ranges[k][0] <= mu[k]) and np.all(mu[k] <= self.ranges[k][1])
                   for k in self.parameters)
