# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip
from numbers import Number

import numpy as np

from pymor.tools import float_cmp_all
from pymor import defaults


class ParameterType(dict):

    __keys = None

    def __init__(self, t):
        if t is None:
            t = {}
        elif isinstance(t, ParameterType):
            pass
        elif hasattr(t, 'parameter_type'):
            assert isinstance(t.parameter_type, ParameterType)
            t = parameter_type.parameter_type
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
        return '{' +  ', '.join('{}: {}'.format(k, self[k]) for k in self.__keys) + '}'


class Parameter(dict):
    '''Class representing a parameter.

    A parameter is simply a dict of numpy arrays. We overwrite copy() to
    ensure that not only the dict but also the arrays are copied. Moreover
    an allclose() method is provided to compare parameters for equality.
    Finally __str__() ensures an alphanumerical ordering of the keys. This
    is not true, however, for keys() or iteritems().
    '''

    __keys = None

    def __init__(self, v):
        if v is None:
            v = {}
        i = v.iteritems() if hasattr(v, 'iteritems') else v
        dict.__init__(self, {k: np.array(v) if not isinstance(v, np.ndarray) else v for k, v in i})

    @classmethod
    def from_parameter_type(cls, mu, parameter_type=None):
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

            if len(array) == 0:
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
    '''Mixin class for objects whose evaluations depend on a parameter.

    Parameters
    ----------
    parameter_type
        The parameter type of the parameters the object takes.
    parameter_local_type
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
    parameter_provided = None
    parameter_global_names = None

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
        self.parse_parameter(mu)
        return True

    def local_parameter(self, mu):
        assert mu.__class__ is Parameter
        return None if self.parameter_local_type is None else {k: mu[v] for k, v in self.parameter_global_names.iteritems()}

    def strip_parameter(self, mu):
        if mu.__class__ is not Parameter:
            mu = Parameter.from_parameter_type(mu, self.parameter_type)
        assert self.parameter_type is None or all(getattr(mu.get(k, None), 'shape', None) == v for k, v in self.parameter_type.iteritems())
        return None if self.parameter_type is None else Parameter({k: mu[k] for k in self.parameter_type})

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

        global_type = local_type.copy() if local_global else ParameterType({global_names[k]: v for k, v in local_type.iteritems()})
        provides = ParameterType(provides)

        def check_op(op, global_type, provides):
            for name, shape in op.parameter_type.iteritems():
                assert name not in global_type or global_type[name] == shape,\
                    'Component dimensions of global name {} do not match ({} and {})'.format(
                    name, global_type[name], shape)
                assert name not in provides or provides[name] == shape,\
                    'Component dimensions of provided name {} do not match'.format(name)
            return True

        if inherits:
            for op in (o for o in inherits if getattr(o, 'parametric', False)):
                assert check_op(op, global_type, provides)
                global_type.update(op.parameter_type)

        self.parameter_type = global_type or None
        self.parameter_local_type = local_type or None
        self.parameter_provided = provides or None
        self.parameter_global_names = global_names
