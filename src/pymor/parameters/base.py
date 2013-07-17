# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip

import numpy as np

from pymor.tools import float_cmp_all


class ParameterType(object):

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise ValueError('ParameterType accepts at most one positional argument')
        self._dict = {}
        self._keys = []
        self.update(*args, **kwargs)

    def __setitem__(self, key, value):
        if not key in self._dict:
            self._keys.append(key)
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __eq__(self, other):
        if other.__class__ != ParameterType:
            return False
        else:
            return self._dict == other._dict

    def __ne__(self, other):
        if other.__class__ != ParameterType:
            return True
        else:
            return self._dict != other._dict

    def __ge__(self, other):
        if other is None:
            return True
        return all(shape == other.get(name, None) for name, shape in self._dict.iteritems())

    def __nonzero__(self):
        return bool(self._keys)

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def update(self, *args, **kwargs):
        if len(args) > 1:
            raise ValueError('update accepts at most one positional argument')
        if len(args) > 0:
            if hasattr(args[0], 'keys'):
                for k in args[0]:
                    self[k] = args[0][k]
            else:
                for k, v in args[0]:
                    self[k] = v
        for k in kwargs:
            self[k] = kwargs[k]

    def copy(self):
        return ParameterType(self)

    def get(self, k, d):
        return self._dict.get(k, d)

    def keys(self):
        return list(self._keys)

    def values(self):
        for k in self._keys:
            yield self._dict[k]

    def iteritems(self):
        for k in self._keys:
            yield (k, self._dict[k])

    def __str__(self):
        s = 'ParameterType(('
        for k, v in self.iteritems():
            s += '(\'{}\', {}),'.format(k, v)
        s += '))'
        return s

    def __repr__(self):
        s = 'ParameterType(('
        for k, v in self.iteritems():
            s += '({}, {}),'.format(repr(k), repr(v))
        s += '))'
        return s


class Parameter(dict):
    '''Class representing a parameter.

    A parameter is simply a dict of numpy arrays together with
    a `ParameterType`. We overwrite copy() to ensure that not only
    the dict but also the arrays are copied. Moreover an
    allclose() method is provided to compare parameters for equality.

    Note that for performance reasons we do not check if the provided
    `ParameterType` actually fits the parameter, so be very careful
    when modifying `Parameter` objects.
    '''

    def __init__(self, v, order=None):
        # calling dict.__init__ breaks multiple inheritance but is faster than
        # the super() call
        dict.__init__(self, v)
        if order is not None:
            self._keys = list(order)
            assert set(dict.keys(self)) == set(self._keys)
        elif hasattr(v, 'keys'):
            self._keys = v.keys()
        else:
            self._keys = [k for k, v in v]
            assert set(dict.keys(self)) == set(self._keys)

    def allclose(self, mu):
        assert isinstance(mu, Parameter)
        if set(self.keys()) != set(mu.keys()):
            return False
        if not all(float_cmp_all(v, mu[k]) for k, v in self.iteritems()):
            return False
        else:
            return True

    def clear(self):
        dict.clear(self)
        self._keys = []

    def copy(self):
        c = Parameter({},order =[])
        for k in self._keys:
            c[k] = self[k].copy()
        return c

    def iterkeys(self):
        return iter(self._keys)

    def itervalues(self):
        for k in self._keys:
            yield self[k]

    def values(self):
        return list(self.itervalues())

    def __setitem__(self, key, value):
        if key not in self:
            self._keys.append(key)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self._keys.remove(key)

    def __iter__(self):
        return self._keys.__iter__()

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
        return ParameterType((k, v.shape) for k in self._keys)

    def __str__(self):
        s = '{'
        for k in self._keys:
            v = self[k]
            if v.ndim > 1:
                v = v.ravel()
            if s == '{':
                s += '{}: {}'.format(k, v)
            else:
                s += ', {}: {}'.format(k, v)
        s += '}'
        return s

def parse_parameter(mu, parameter_type=None):
    '''Takes a parameter specification `mu` and makes it a `Parameter` according to `parameter_type`.

    Depending on the `parameter_type`, `mu` can be given as a `Parameter`, dict, tuple,
    list, array or scalar.

    Parameters
    ----------
    mu
        The parameter specification.
    parameter_type
        The parameter type w.r.t. which `mu` is to be interpreted.

    Returns
    -------
    The corresponding `Parameter`.

    Raises
    ------
    ValueError
        Is raised if `mu` cannot be interpreted as a `Paramter` of `parameter_type`.
    '''
    if not parameter_type:
        assert mu is None
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
        mu = dict(izip(parameter_type._keys, mu))
    elif set(mu.keys()) != set(parameter_type.keys()):
        raise ValueError('Components do not match')
    for k, v in mu.iteritems():
        if not isinstance(v, np.ndarray):
            mu[k] = np.array(v)
    if not all(mu[k].shape == parameter_type[k] for k in mu):
        raise ValueError('Component dimensions do not match')
    return Parameter(mu, order=parameter_type._keys)


def parse_parameter_type(parameter_type):
    '''Takes a parameter type specification and makes it a `ParameterType`.

    A `ParameterType` is an ordered dict whose values are tuples of natural numbers
    defining the shape of the corresponding parameter component.

    Parameters
    ----------
    parameter_type
        The parameter type specification. Can be a dict or OrderedDict, in which case
        scalar values are made tuples of length 1, or a `ParameterSpace` whose
        parameter_type is taken.

    Returns
    -------
    The corresponding parameter type.
    '''

    from pymor.parameters.interfaces import ParameterSpaceInterface
    if parameter_type is None:
        return None
    if isinstance(parameter_type, ParameterSpaceInterface):
        return ParameterType(parameter_type.parameter_type)
    parameter_type = ParameterType(parameter_type)
    for k, v in parameter_type.iteritems():
        if not isinstance(v, tuple):
            if v == 0 or v == 1:
                parameter_type[k] = tuple()
            else:
                parameter_type[k] = tuple((v,))
    return parameter_type


class Parametric(object):
    '''Mixin class for objects whose evaluations depend on a parameter.

    Parameters
    ----------
    parameter_type
        The parameter type of the parameters the object takes.
    global_parameter_type
        The parameter type without any renamings by the user.
    local_parameter_type
        The parameter type of the parameter components which are introduced
        by the object itself and are not inherited by other objects it
        depends on.
    parameter_space
        If not `None` the `ParameterSpace` the parameter is expected to lie in.
    parametric:
        Is True if the object has a nontrivial parameter type.
    '''

    parameter_type = None
    parameter_local_type = None
    parameter_global_names = None
    parameter_provided = None
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
        if mu is None:
            assert self.parameter_type is None
            return None
        if mu.__class__ is not Parameter:
            mu = parse_parameter(mu, self.parameter_type)
        assert self.parameter_type is None or all(getattr(mu.get(k, None), 'shape', None) == v for k, v in self.parameter_type.iteritems())
        return mu

    def local_parameter(self, mu):
        assert mu.__class__ is Parameter
        return None if self.parameter_local_type is None else {k: mu[v] for k, v in self.parameter_global_names.iteritems()}

    def strip_parameter(self, mu):
        if not isinstance(mu, Parameter):
            mu_ = parse_parameter(mu, self.parameter_type)
        assert self.parameter_type is None or all(getattr(mu.get(k, None), 'shape', None) == v for k, v in self.parameter_type.iteritems())
        return None if self.parameter_type is None else Parameter({k: mu[k] for k in self.parameter_type},
                                                                  order=self.parameter_type._keys)

    def build_parameter_type(self, local_type=None, global_names=None, local_global=False, inherits=None, provides=None):
        '''Builds the parameter type of the object. To be called by __init__.

        Parameters
        ----------
        local_type
            Parameter type for the parameter components introduced by the object itself.
        global_names
            A dict of the form `{'localname': 'globalname', ...}` defining a name mapping specifying global
            parameter names for the keys of local_type
        local_global
            If True, use the identity mapping `{'localname': 'localname', ...}` as global_names, i.e. each local
            parameter name should be treated as a global parameter name.
        inherits
            List where each entry is a Parametric object whose parameter type shall become part of the
            built parameter type.
        provides
            Dict where the keys specify parameter names and the values are corresponding shapes. The
            parameters listed in `provides` will not become part of the parameter type. Instead they
            have to be provided by the class implementor.

        Returns
        -------
        The parameter type of the object.
        '''
        assert not local_global or global_names is None
        local_type = parse_parameter_type(local_type)
        if local_global and local_type is not None:
            global_names = {k: k for k in local_type}
        if local_type and not (global_names and all(k in global_names for k in local_type)):
            if not global_names:
                raise ValueError('Must specify a global name for each key of local_type')
            else:
                for k in local_type:
                    if not k in global_names:
                        raise ValueError('Must specify a global name for {}'.format(k))

        parameter_maps = {}
        parameter_types = {}

        global_type = ParameterType()
        if local_type:
            for n in global_names:
                global_type[n] = local_type[n]

        provides = parse_parameter_type(provides)
        provides = provides or {}

        if inherits:
            for op in (o for o in inherits if o.parametric):
                for name, shape in op.parameter_type.iteritems():
                    if name in global_type and global_type[name] != shape:
                        raise ValueError('Component dimensions of global name {} do not match ({} and {})'.format(name,
                            global_type[name], shape))
                    if name in provides:
                        if provides[global_name] != shape:
                            raise ValueError('Component dimensions of provided name {} do not match'.format(name))
                    else:
                        global_type[name] = shape

        self.parameter_type = global_type or None
        self.parameter_local_type = local_type
        self.parameter_global_names = global_names
        self.parameter_provided = provides or None

    def parameter_info(self):
        '''Return an info string about the object's parameter type and how it is built.'''

        if not self.parametric:
            return 'The parameter_type is None\n'
        else:
            return 'The parameter_type is: {}\n\n'.format(self.parameter_type)
