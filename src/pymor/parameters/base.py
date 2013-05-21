# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

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

    def keys(self):
        return iter(self._keys)

    def values(self):
        for k in self._keys:
            yield self._dict[k]

    def iteritems(self):
        for k in self._keys:
            yield (k, self._dict[k])

    def __str__(self):
        s = 'ParameterType(('
        for k, v in self.iteritems():
            s += '({}, {}),'.format(k, v)
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

    def __init__(self, parameter_type, *args, **kwargs):
        super(Parameter, self).__init__(*args, **kwargs)
        self.parameter_type = parameter_type

    def allclose(self, mu):
        assert isinstance(mu, Parameter)
        if set(self.keys()) != set(mu.keys()):
            return False
        if not all(float_cmp_all(v, mu[k]) for k, v in self.iteritems()):
            return False
        else:
            return True

    def copy(self):
        c = Parameter(self.parameter_type)
        for k, v in self.iteritems():
            c[k] = v.copy()
        return c

    def __str__(self):
        s = '{'
        for k, v in self.iteritems():
            if s == '{':
                s += '{}: {}'.format(k, v.ravel())
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
        mu = Parameter(parameter_type, zip(parameter_type.keys(), mu))
    elif set(mu.keys()) != set(parameter_type.keys()):
        raise ValueError('Components do not match')
    else:
        mu = Parameter(parameter_type, mu)
    for k, v in mu.iteritems():
        if not isinstance(v, np.ndarray):
            mu[k] = np.array(v)
    if not all(mu[k].shape == parameter_type[k] for k in mu):
        raise ValueError('Component dimensions do not match')
    return mu


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
    _parameter_space = None

    @property
    def parameter_space(self):
        return self._parameter_space

    @parameter_space.setter
    def parameter_space(self, ps):
        assert self.parameter_type == ps.parameter_type
        self._parameter_space = ps

    @property
    def parametric(self):
        return self.parameter_type is not None

    def parse_parameter(self, mu):
        if not self.parameter_type:
            assert mu is None
            return None
        if mu.__class__ == Parameter and mu.parameter_type == self.parameter_type:
            return mu
        else:
            return parse_parameter(mu, self.parameter_type)

    def map_parameter(self, mu, target='self', component=None, provide=None):
        '''Maps a parameter to the local parameter type or to the parameter types of the objects from which
        parameter components are inherited.

        This method will be called by the implementor of the class.

        Parameters
        ----------
        mu
            The parameter to map. Can be of the types accepted by `parse_parameter`.
        target
            If 'self', map to the local parameter type provided by `local_type` in
            `build_parameter_type`. Otherwise `target` has to be a key of the `inherits`
            parameter passed to `build_parameter_type` and `mu` is mapped to the corresponding
            parameter type.
        component
            If `target` consists of a tuple of `Parametric`s, this specifies the index of the
            `Parametric` to map to. Otherwise must be `None`.
        provide
            Dict of additionaly provided parameters which are not part of `self.parameter_type` or
            `None`. Note that the value of each parameter has to be a numpy array of the correct
            shape. No automatic conversion is performed.

        Returns
        -------
        The mapped `Parameter`.
        '''

        if self.parameter_type is None:
            assert mu is None
            return None

        if mu.__class__ != Parameter or mu.parameter_type != self.parameter_type:
            mu = parse_parameter(mu, self.parameter_type)

        parameter_type = self.parameter_types[target] if component is None else self.parameter_types[target][component]
        if parameter_type is None:
            return None

        if provide is not None:
            assert all(provide[k].shape == self.parameter_provided[k] for k in provide)
            mu.update(provide)

        if component is None:
            parameter_map = self.parameter_maps_renamed[target]
            mu_mapped = {k: mu[v] for k, (v, m) in parameter_map.iteritems()}
        else:
            parameter_map = self.parameter_maps_renamed[target][component]
            mu_mapped = {k: mu[v][component, ...] if m else mu[v] for k, (v, m) in parameter_map.iteritems()}

        return Parameter(parameter_type, mu_mapped)

    def build_parameter_type(self, local_type=None, inherits=None, local_global=False, provides=None):
        '''Builds the parameter type of the object. To be called by __init__.

        Parameters
        ----------
        local_type
            Parameter type for the parameter components introduced by the object itself.
        inherits
            Ordered dict where each key is a string indentifier for an object from which parameter components
            are inherited and where the value is the parameter type of the corresponding object.
        local_global
            If True, treat the components of local_type as global components.
        provides
            Dict where the keys specify parameter names and the values are corresponding shapes. The
            parameters listed in `provides` will not become part of the parameter type. Instead they
            have to be provided using the `provides` parameter of the `map_parameter` method.

        Returns
        -------
        The parameter type of the object. It is built according to the following rules:
            1. If local_global is False, a '.' is prepended to each key of `local_type`.
            2. For each key `obj` of `inherits`:
                    If a key of `inherits['obj']` begins with a '.', '.obj' is prepended to the key.
            3. The parameter type is built by concatenating `local_type` with the values of `inherits`
               ignoring duplicate keys. (The values of duplicate keys must be equal.)
        '''
        local_type = parse_parameter_type(local_type)
        self.local_parameter_type = local_type
        parameter_maps = {}
        parameter_types = {}
        if not local_type:
            global_type = ParameterType()
            parameter_maps['self'] = None
            parameter_types['self'] = None
        elif local_global:
            global_type = local_type.copy()
            parameter_maps['self'] = {k: (k, False) for k in local_type.keys()}
            parameter_types['self'] = local_type
        else:
            parameter_map = {k: ('.{}'.format(k), False) for k in local_type.keys()}
            global_type = ParameterType(('.{}'.format(k), v) for k, v in local_type.iteritems())
            parameter_maps['self'] = parameter_map
            parameter_types['self'] = local_type

        provides = parse_parameter_type(provides)
        provides = provides or {}

        if inherits:
            for n in inherits:
                if isinstance(inherits[n], tuple) or isinstance(inherits[n], list):
                    merged_param_map = {}
                    if all(op.parameter_type for op in inherits[n]):
                        for name, shape in inherits[n][0].parameter_type.iteritems():
                            if (name.startswith('.')
                                    and all(name in op.parameter_type for op in inherits[n])
                                    and all(op.parameter_type[name] == shape for op in inherits[n])):
                                global_name = '.{}{}'.format(n, name)
                                if global_name in provides:
                                    if provides[global_name][1:] != shape or len(inherits[n]) != provides[global_name][0]:
                                        raise ValueError('Component dimensions of provided name {} do not match'
                                                         .format(name))
                                else:
                                    global_type[global_name] = shape
                                merged_param_map[name] = (global_name, True)
                    parameter_map_list = list()
                    for i, op in enumerate(inherits[n]):
                        if op.parameter_type:
                            parameter_map = merged_param_map.copy()
                            for name, shape in op.parameter_type.iteritems():
                                if name in merged_param_map:
                                    continue
                                if name.startswith('.'):
                                    global_name = '.{}_{}{}'.format(n, i, name)
                                else:
                                    if name in global_type and global_type[name] != shape:
                                        raise ValueError('Dimensions of global name {} do not match'.format(name))
                                    global_name = name
                                if global_name in provides:
                                    if provides[global_name] != shape:
                                        raise ValueError('Component dimensions of provided name {} do not match'
                                                         .format(name))
                                else:
                                    global_type[global_name] = shape
                                parameter_map[name] = (global_name, False)
                            parameter_map_list.append(parameter_map)
                        else:
                            parameter_map_list.append(None)
                    parameter_maps[n] = parameter_map_list
                    parameter_types[n] = [op.parameter_type.copy() if op.parameter_type is not None else None
                                          for op in inherits[n]]
                elif inherits[n].parameter_type:
                    parameter_map = {}
                    for name, shape in inherits[n].parameter_type.iteritems():
                        if name.startswith('.'):
                            global_name = '.{}{}'.format(name, shape)
                        else:
                            if name in global_type and global_type[name] != shape:
                                raise ValueError('Component dimensions of global name {} do not match'.format(name))
                            global_name = name

                        if global_name in provides:
                            if provides[global_name] != shape:
                                raise ValueError('Component dimensions of provided name {} do not match'.format(name))
                        else:
                            global_type[global_name] = shape
                        parameter_map[name] = (global_name, False)

                    parameter_maps[n] = parameter_map
                    parameter_types[n] = inherits[n].parameter_type.copy()
                else:
                    parameter_maps[n] = None
                    parameter_types[n] = None

        self.global_parameter_type = global_type or None
        self.parameter_type = global_type or None
        self.parameter_types = parameter_types
        self.parameter_provided = provides or None
        self.parameter_maps = parameter_maps
        self.parameter_maps_renamed = parameter_maps
        self.parameter_name_map = {}

    def rename_parameter(self, name_map):
        '''Rename a parameter component of the object's parameter type.

        This method can be called by the object's owner to rename local parameter components
        (whose name begins with '.') to global parameter components. This should be called
        directly after instantiation.

        Parameters
        ----------
        name_map
            A dict of the form `{'.oldname1': 'newname1', ...}` defining the name mapping.
        '''
        assert self.parametric
        for k, v in name_map.iteritems():
            if not k.startswith('.'):
                raise ValueError('Can only rename parameters with local name (starting with ".")')
            if not k in self.global_parameter_type:
                raise ValueError('There is no parameter named {}'.format(k))
            if k in self.parameter_name_map:
                raise ValueError('{} has already been renamed to {}'.format(k, self.parameter_name_map[k]))
            self.parameter_name_map[k] = v

        # rebuild parameter_maps_renamed
        def rename_map(m):
            if m is None:
                return None
            return {k: (self.parameter_name_map[v], b) if v in self.parameter_name_map else (v, b)
                    for k, (v, b) in m.iteritems()}

        self.parameter_maps_renamed = {k: map(rename_map, m) if isinstance(m, list) else rename_map(m)
                                       for k, m in self.parameter_maps.iteritems()}

        # rebuild parameter_type
        parameter_type = ParameterType()
        for k, v in self.global_parameter_type.iteritems():
            if k in self.parameter_name_map:
                k = self.parameter_name_map[k]
            if k in parameter_type and parameter_type[k] != v:
                raise ValueError('Mismatching shapes for parameter {}: {} and {}'.format(k, parameter_type[k], v))
            parameter_type[k] = v
        self.parameter_type = parameter_type

    def parameter_info(self):
        '''Return an info string about the object's parameter type and how it is built.'''

        if not self.parametric:
            return 'The parameter_type is None\n'

        msg = 'The parameter_type is: {}\n\n'.format(self.parameter_type)
        msg += 'We have the following parameter-maps:\n\n'
        for n, mp in self.parameter_maps.iteritems():
            if not mp:
                msg += '{}: None'.format(n)
            elif not isinstance(mp, list):
                pad = ' ' * (len(n) + 2)
                msg += '{}: '.format(n)
                for k, (v, f) in mp.iteritems():
                    msg += '{} <- {}'.format(k, v)
                    if v in self.parameter_name_map:
                        msg += ' <- {}'.format(self.parameter_name_map[k])
                    msg += '\n' + pad
            else:
                for i, m in mp.enumerate():
                    tag = '{}[{}]: '.format(n, i)
                    pad = ' ' * len(tag)
                    msg += tag
                    if not m:
                        msg += 'None\n'
                    else:
                        for k, (v, f) in m.iteritems():
                            msg += '{} <- {}'.format(k, v)
                            if f:
                                msg += '[{}, ...]'.format(i)
                            if v in self.parameter_name_map:
                                msg += ' <- {}'.format(self.parameter_name_map[k])
                            msg += '\n' + pad
            msg += '\n'
        return msg
