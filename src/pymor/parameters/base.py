# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''This module contains the implementation of pyMOR's parameter handling facilities.

A |Parameter| in pyMOR is basically a `dict` of |NumPy Arrays|. Each item of the
dict is called a parameter component. The |ParameterType| of a |Parameter| is a dict
of the shapes of the parameter components, i.e. ::

    mu.parameter_type['component'] == mu['component'].shape

Classes which represent mathematical objects depending on parameters, e.g. |Functions|,
|Operators|, |Discretizations| derive from the |Parametric| mixin. Each |Parametric|
object has a :attr:`~Parametric.parameter_type` attribute holding the |ParameterType|
of the |Parameters| the object's `evaluate`, `apply`, `solve`, etc. methods expect.
Note that the |ParameterType| of the given |Parameter| is allowed to actually be a
superset (in the obvious sense) of the object's |ParameterType|.

The object's |ParameterType| is determined in its :meth:`__init__` method by calling
:meth:`~Parametric.build_parameter_type` which computes the |ParameterType| as the
union of the |ParameterTypes| of the objects given to the method. This way, an, e.g.,
|Operator| can inherit the |ParameterTypes| of the data functions it depends upon.

A |Parametric| object can have a |ParameterSpace| assigned to it by setting the
:attr:`~Parametric.parameter_space` attribute. (The |ParameterType| of the space
has to agree with the |ParameterType| of the object.) The
:meth:`~Parametric.parse_parameter` method parses a user input according to
the object's |ParameterType| to make it a |Parameter| (e.g. if the |ParameterType|
consists of a single one-dimensional component, the user can simply supply a list
of numbers of the right length). Moreover, if given a |Parameter|, it checks whether
the |Parameter| has an appropriate |ParameterType|. Thus :meth:`~Parametric.parse_parameter`
should always be called by the implementor for any given parameter argument. The
:meth:`~Parametric.local_parameter` method is used to extract the local parameter
components of the given |Parameter| and performs some name mapping. (See the
documentation of :meth:`~Parametric.build_parameter_type` for details.)
'''

from __future__ import absolute_import, division, print_function

from itertools import izip
from numbers import Number

import numpy as np

from pymor import defaults
from pymor.tools import float_cmp_all


class ParameterType(dict):
    '''Class representing a parameter type.

    A parameter type is simply a dictionary with strings as keys and tuples of
    natural numbers as values. The keys are the names of the parameter components
    and the tuples their expected shape. (Compare :class:`Parameter`.)

    Apart from checking the correct format of its values, the only difference
    between a ParameterType and an ordinary `dict` is, that |ParameterType|
    orders its keys alphabetically.

    Parameters
    ----------
    t
        If `t` is an object with a `parameter_type` attribute, a copy of this
        |ParameterType| is created. Otherwise, `t` can be anything from which
        a `dict` can be constructed.
    '''

    __keys = None

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
            for k, v in t.iteritems():
                if not isinstance(v, tuple):
                    assert isinstance(v, Number)
                    t[k] = tuple() if v == 0 else (v,)
        # calling dict.__init__ breaks multiple inheritance but is faster than
        # the super() call
        dict.__init__(self, t)

    def clear(self):
        dict.clear(self)
        self.__keys = None

    def copy(self):
        c = ParameterType(self)
        if self.__keys is not None:
            c.__keys = list(self.__keys)
        return c

    def __setitem__(self, key, value):
        if not isinstance(value, tuple):
            assert isinstance(value, Number)
            value = tuple() if value == 0 else (value,)
        assert all(isinstance(v, Number) for v in value)
        if key not in self:
            self.__keys = None
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self.__keys = None

    def __iter__(self):
        if self.__keys is None:
            self.__keys = sorted(dict.keys(self))
        return iter(self.__keys)

    def keys(self):
        if self.__keys is None:
            self.__keys = sorted(dict.keys(self))
        return list(self.__keys)

    def iterkeys(self):
        return iter(self)

    def items(self):
        return list(self.iteritems())

    def iteritems(self):
        for k in self:
            yield k, self[k]

    def values(self):
        return list(self.itervalues())

    def itervalues(self):
        for k in self:
            yield self[k]

    def fromkeys(self, S, v=None):
        raise NotImplementedError

    def pop(self, k, d=None):
        raise NotImplementedError

    def popitem(self):
        raise NotImplementedError

    def update(self, d):
        self.__keys = None
        if isinstance(d, ParameterType):
            dict.update(self, d)
        else:
            for k, v in d.iteritems():
                self[k] = v

    def __str__(self):
        if self.__keys is None:
            self.__keys = sorted(self.keys())
        return '{' + ', '.join('{}: {}'.format(k, self[k]) for k in self.__keys) + '}'


class Parameter(dict):
    '''Class representing a parameter.

    A |Parameter| is simply a `dict` where each key is a string and each value
    is a |NumPy array|. We call an item of the dictionary a *parameter component*.

    |Parameter| differs from an ordinary `dict` in the following ways:

        - It is ensured that each value is a |NumPy array|.
        - We overwrite :meth:`copy` to ensure that not only the `dict`
          but also the |NumPy arrays| are copied.
        - The :meth:`allclose` method allows to compare |Parameters| for
          equality in a mathematically meaningful way.
        - Each |Parameter| has a :attr:`~Parameter.sid` property providing a
          unique state id. (See :mod:`pymor.core.interfaces`.)
        - We override :meth:`__str__` to ensure alphanumerical ordering of the keys
          and more or less pretty printing of the values.
        - The :attr:`parameter_type` property can be used to obtain the |ParameterType|
          of the parameter.
        - Use :meth:`from_parameter_type` to construct a |Parameter| from a |ParameterType|
          and user supplied input.

    Parameters
    ----------
    v
        Anything that `dict` accepts for the construction of a dictionary.

    Attributes
    ----------
    parameter_type
        The |ParameterType| of the |Parameter|.
    sid
        The state id of the |Parameter|. (See :mod:`pymor.core.interfaces`.)
    '''

    __keys = None

    def __init__(self, v):
        if v is None:
            v = {}
        i = v.iteritems() if hasattr(v, 'iteritems') else v
        dict.__init__(self, {k: np.array(v) if not isinstance(v, np.ndarray) else v for k, v in i})

    @classmethod
    def from_parameter_type(cls, mu, parameter_type=None):
        '''Takes a user input `mu` and interprets it as a |Parameter| according to the given
        |ParameterType|.

        Depending on the |ParameterType|, `mu` can be given as a |Parameter|, dict, tuple,
        list, array or scalar. For the excact details, please refer to the source code.

        Parameters
        ----------
        mu
            The user input which shall be interpreted as a |Parameter|.
        parameter_type
            The |ParameterType| w.r.t. which `mu` is to be interpreted.

        Returns
        -------
        The resulting |Parameter|.

        Raises
        ------
        ValueError
            Is raised if `mu` cannot be interpreted as a |Parameter| of |ParameterType|
            `parameter_type`.
        '''
        if not parameter_type:
            assert mu is None or mu == {}
            return None

        if isinstance(mu, Parameter):
            assert mu.parameter_type == parameter_type
            return mu

        if not isinstance(mu, dict):
            if isinstance(mu, (tuple, list)):
                if len(parameter_type) == 1 and len(mu) != 1:
                    mu = (mu,)
            else:
                mu = (mu,)
            if len(mu) != len(parameter_type):
                raise ValueError('Parameter length does not match.')
            mu = dict(izip(parameter_type, mu))
        elif set(mu.keys()) != set(parameter_type.keys()):
            raise ValueError('Provided parameter with keys {} does not match parameter type {}.'
                             .format(mu.keys(), parameter_type))
        for k, v in mu.iteritems():
            if not isinstance(v, np.ndarray):
                v = np.array(v)
                try:
                    v = v.reshape(parameter_type[k])
                except ValueError:
                    raise ValueError('Shape mismatch for parameter component {}: got {}, expected {}'
                                     .format(k, v.shape, parameter_type[k]))
                mu[k] = v
            if v.shape != parameter_type[k]:
                raise ValueError('Shape mismatch for parameter component {}: got {}, expected {}'
                                 .format(k, v.shape, parameter_type[k]))
        return cls(mu)

    def allclose(self, mu):
        '''Compare to |Parameters| using :meth:`~pymor.tools.floatcmp.float_cmp_all`.

        Parameters
        ----------
        mu
            The |Parameter| with which to compare.

        Returns
        -------
        `True` if both |Parameters| have the same |ParameterType| and all parameter
        components are almost equal, else `False`.
        '''
        assert isinstance(mu, Parameter)
        if self.viewkeys() != mu.viewkeys():
            return False
        elif not all(float_cmp_all(v, mu[k]) for k, v in self.iteritems()):
            return False
        else:
            return True

    def clear(self):
        dict.clear(self)
        self.__keys = None

    def copy(self):
        c = Parameter({k: v.copy() for k, v in self.iteritems()})
        if self.__keys is not None:
            c.__keys = list(self.__keys)
        return c

    def __setitem__(self, key, value):
        if key not in self:
            self.__keys = None
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self.__keys = None

    def __eq__(self, mu):
        if not isinstance(mu, Parameter):
            mu = Parameter(mu)
        if self.viewkeys() != mu.viewkeys():
            return False
        elif not all(np.array_equal(v, mu[k]) for k, v in self.iteritems()):
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
        return ParameterType({k: v.shape for k, v in self.iteritems()})

    @property
    def sid(self):
        if self.__keys is None:
            self.__keys = sorted(self.keys())
        return 'Parameter_' + '_'.join('{}-{}-{}'.format(k, self[k].shape, self[k].tostring()) for k in self.__keys)

    def __str__(self):

        def format_array(array):
            def format_element(e):
                if e > 1e15:
                    return '%(n).2e' % {'n': e}
                elif e == np.floor(e):
                    return '%(n).0f' % {'n': e}
                elif e - np.floor(e) > 0.01 or e < 1000:
                    return '%(n).2f' % {'n': e}
                else:
                    return '%(n).2e' % {'n': e}

            if array.ndim == 0:
                return str(array.item())
            elif len(array) == 0:
                return ''
            elif len(array) == 1:
                if defaults.compact_print:
                    return '[' + format_element(array[0]) + ']'
                else:
                    return '[{}]'.format(array[0])
            s = '['
            for ii in np.arange(len(array) - 1):
                if defaults.compact_print:
                    s += format_element(array[ii]) + ', '
                else:
                    s += '{}, '.format(array[ii])
            if defaults.compact_print:
                s += format_element(array[-1]) + ']'
            else:
                s += '{}]'.format(array[-1])
            return s

        np.set_string_function(format_array, repr=False)
        if self.__keys is None:
            self.__keys = sorted(self.keys())
        s = '{'
        for k in self.__keys:
            v = self[k]
            if v.ndim > 1:
                v = v.ravel()
            if s == '{':
                s += '{}: {}'.format(k, v)
            else:
                s += ', {}: {}'.format(k, v)
        s += '}'
        np.set_string_function(None, repr=False)
        return s


class Parametric(object):
    '''Mixin class for objects representing mathematical entities depending on a |Parameter|.

    Each such object has a |ParameterType| stored in the :attr:`parameter_type` attribute,
    which should be calculated by the implementor during :meth:`__init__` using the
    :meth:`build_parameter_type` method. Methods expecting the |Parameter| (typically
    `evaluate`, `apply`, `solve`, etc. ..) should accept an optional argument `mu` defaulting
    to `None`. This `mu` should then be fed into :meth:`parse_parameter` to obtain a
    |Parameter| of correct |ParameterType| from the (user supplied) input `mu`. The local
    parameter components (see :meth:`build_parameter_type`) can be extracted using
    :meth:`local_type`.

    Attributes
    ----------
    parameter_type
        The |ParameterType| of the |Parameters| the object expects or `None`,
        if the object actually is parameter independent.
    parameter_local_type
        The |ParameterType| of the parameter components which are introduced
        by the object itself and are not inherited by other objects it
        depends on. `None` if there are no such component.
        (See :meth:`build_parameter_type`.)
    parameter_space
        |ParameterSpace| the parameter is expected to lie in or `None`.
    parametric:
        `True` if the object really depends on a parameter, i.e. :attr:`parameter_type`
        is not `None`.
    '''

    parameter_type = None
    parameter_local_type = None

    _parameter_global_names = None
    _parameter_space = None

    @property
    def parameter_space(self):
        return self._parameter_space

    @parameter_space.setter
    def parameter_space(self, ps):
        assert ps is None or self.parameter_type == ps.parameter_type
        self._parameter_space = ps

    @property
    def parametric(self):
        return self.parameter_type is not None

    def parse_parameter(self, mu):
        '''Interpret a user supplied parameter `mu` as a |Parameter|.

        If `mu` is not already a |Parameter|, :meth:`Parameter.from_parameter_type`
        is used, to make `mu` a parameter of the correct |ParameterType|. If `mu`
        is already a |Parameter|, it is checked if its |ParameterType| matches our own.
        (It is actually allowed that the |ParameterType| of `mu` is a superset of
        our own |ParameterType| in the obvious sense.)

        Parameters
        ----------
        mu
            The input to parse as a |Parameter|.
        '''
        if mu is None:
            assert self.parameter_type is None, \
                'Given parameter is None but expected parameter of type {}'.format(self.parameter_type)
            return None
        if mu.__class__ is not Parameter:
            mu = Parameter.from_parameter_type(mu, self.parameter_type)
        assert self.parameter_type is None or all(getattr(mu.get(k, None), 'shape', None) == v
                                                  for k, v in self.parameter_type.iteritems()), \
            ('Given parameter of type {} does not match expected parameter type {}'
             .format(mu.parameter_type, self.parameter_type))
        return mu

    def check_parameter(self, mu):
        '''Wrapper around :meth:`parse_parameter` returning `True`.

        This method is intended to be used in `assert` statements, if one is, for
        some reason, not interested in the |Parameter| itself, e.g. if the object
        is known to not be :attr:`parametric`. This way one can check if the
        provided `mu` can be parsed correctly (if not an exception will be raised),
        and the check can be disabled for speedup using Python's `-O` command line
        argument.

        Parameters
        ----------
        mu
            The input to be checked.
        '''
        self.parse_parameter(mu)
        return True

    def local_parameter(self, mu):
        '''Extract the local parameter components with their local names from a given |Parameter|.

        See :meth:`build_parameter_type` for details.
        '''
        assert mu.__class__ is Parameter
        return (None if self.parameter_local_type is None
                else {k: mu[v] for k, v in self._parameter_global_names.iteritems()})

    def strip_parameter(self, mu):
        '''Remove all components of the |Parameter| `mu` which are not part of the object's |ParameterType|.

        Otherwise :meth:`strip_parameter` behaves like :meth:`parse_parameter`.

        This method is mainly useful for caching where the key should only contain the
        relevant parameter components.
        '''
        if mu.__class__ is not Parameter:
            mu = Parameter.from_parameter_type(mu, self.parameter_type)
        assert self.parameter_type is None \
            or all(getattr(mu.get(k, None), 'shape', None) == v for k, v in self.parameter_type.iteritems())
        return None if self.parameter_type is None else Parameter({k: mu[k] for k in self.parameter_type})

    def build_parameter_type(self, local_type=None, global_names=None, local_global=False,
                             inherits=None, provides=None):
        '''Builds the |ParameterType| of the object. Should be called by :meth:`__init__`.

        The |ParameterType| of a |Parametric| object is determined by the parameter components
        the object by itself requires for evaluation, and by the parameter components
        required by the objects the object depends upon for evaluation. We speak of local
        and inherited parameter components. The |ParameterType| of the local parameter
        components are provided via the `local_type` parameter, whereas the |Parametric|
        objects from which parameter components are inherited are provided as the `inherits`
        parameter.

        Since the implementor does not necessarily know the future use of the object,
        a mapping between the names of the local parameter components and their intended
        global names (from the user perspective) can be provided via the `global_names`
        parameter. This mapping of names will be usually provided by the user when
        instantiating the class. (E.g. a |Function| evaluating x->x^a could depend
        on a local parameter component named 'exponent', whereas the user wishes to name
        the component 'decay_rate'.) If such a mapping is not desired, the `local_global`
        parameter must be set to `True`. (To later extract the local parameter components
        with their local names from a given |Parameter| use the :meth:`local_parameter`
        method.)

        After the name mapping, all parameter components (local or inherited by one of the
        objects provided via `inherits`) with the same name are treated as identical and
        are thus required to have the same shapes. The object's |ParameterType| is then
        made up by the shapes of all parameter components appearing.

        Additionally components of the resulting |ParameterType| can be removed by
        specifying them via the `provides` `dict`. The idea is that the object itself
        may provide parameter components to the inherited objects which thus should
        not become part of the object's own parameter type. (A typical application
        would be |InstationaryDiscretization|, which provides a time parameter
        component to its (time-dependent) operators during time-stepping.)

        .. note::
           As parameter components of the |ParameterTypes| of different objects are
           treated as equal if they have the same name, renaming a local parameter
           component is not merely a convenience feature but can also have a semantic
           meaning by identifying local parameter components with inherited ones.


        Parameters
        ----------
        local_type
            |ParameterType| of the local parameter components.
        global_names
            A `dict` of the form `{'localname': 'globalname', ...}` defining a name mapping
            specifying global parameter component names for each key of `local_type`. If `None`
            and `local_type` is not `None`, `local_global` must be set to `True`.
        local_global
            If `True,` treat the names of the local parameter components as global names of these
            components. In this case, `global_names` must be `None`.
        inherits
            Iterable where each entry is a |Parametric| object whose |ParameterType| shall become
            part of the built |ParameterType|.
        provides
            `Dict` of parameter component names and their shapes which are provided by the object
            itself to the objects in the `inherited` list. The parameter components listed here
            will thus not become part of the object's |ParameterType|.
        '''
        assert not local_global or global_names is None
        assert inherits is None or all(op is None or isinstance(op, Parametric) for op in inherits)

        local_type = ParameterType(local_type)
        if local_global and local_type is not None:
            global_names = {k: k for k in local_type}

        def check_local_type(local_type, global_names):
            assert not local_type or global_names, 'Must specify a global name for each key of local_type'
            for k in local_type:
                assert k in global_names, 'Must specify a global name for {}'.format(k)
            return True

        assert check_local_type(local_type, global_names)

        global_type = (local_type.copy() if local_global
                       else ParameterType({global_names[k]: v for k, v in local_type.iteritems()}))
        provides = ParameterType(provides)

        def check_op(op, global_type, provides):
            for name, shape in op.parameter_type.iteritems():
                assert name not in global_type or global_type[name] == shape,\
                    ('Component dimensions of global name {} do not match ({} and {})'
                     .format(name, global_type[name], shape))
                assert name not in provides or provides[name] == shape,\
                    'Component dimensions of provided name {} do not match'.format(name)
            return True

        if inherits:
            for op in (o for o in inherits if getattr(o, 'parametric', False)):
                assert check_op(op, global_type, provides)
                global_type.update({k: v for k, v in op.parameter_type.iteritems() if k not in provides})

        self.parameter_type = global_type or None
        self.parameter_local_type = local_type or None
        self._parameter_global_names = global_names
