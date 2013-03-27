from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import numpy as np

from pymor.tools import float_cmp_all


class Parameter(OrderedDict):
    '''Class representing a parameter.

    A parameter is simply an ordered dict of numpy arrays.
    We overwrite copy() to ensure that not only the dict but
    also the arrays are copied. Moreover an allclose() method
    is provided to compare parameters for equality.

    Inherits
    --------
    OrderedDict
    '''

    def allclose(self, mu):
        if set(self.keys()) != set(mu.keys()):
            return False
        for k, v in self.iteritems():
            if not float_cmp_all(v, mu[k]):
                return False
        return True

    def copy(self):
        c = Parameter()
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


def parse_parameter(mu, parameter_type={}):
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

    if not isinstance(mu, dict):
        if isinstance(mu, (tuple, list)):
            if len(parameter_type) == 1 and len(mu) != 1:
                mu = (mu,)
        else:
            mu = (mu,)
        if len(mu) != len(parameter_type):
            raise ValueError('Parameter length does not match.')
        mu = Parameter(zip(parameter_type.keys(), mu))
    elif set(mu.keys()) != set(parameter_type.keys()):
        raise ValueError('Components do not match')
    else:
        mu = Parameter((k, mu[k]) for k in parameter_type)
    for k, v in mu.iteritems():
        if not isinstance(v, np.ndarray):
            mu[k] = np.array(v)
    if not all(mu[k].shape == parameter_type[k] for k in mu.keys()):
        raise ValueError('Component dimensions do not match')
    return mu


def parse_parameter_type(parameter_type):
    '''Takes a parameter type specification and makes it a valid parameter type.

    A parameter type is an ordered dict whose values are tuples of natural numbers
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
    if isinstance(parameter_type, ParameterSpaceInterface):
        return OrderedDict(parameter_type.parameter_type)
    parameter_type = OrderedDict(parameter_type)
    for k, v in parameter_type.iteritems():
        if not isinstance(v, tuple):
            if v == 0 or v == 1:
                parameter_type[k] = tuple()
            else:
                parameter_type[k] = tuple((v,))
    return parameter_type


class Parametric(object):
    '''Mixin class for objects whose evaluations depend on a parameter.

    Attributes
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

    parameter_type = OrderedDict()
    global_parameter_type = OrderedDict()
    local_parameter_type = OrderedDict()
    parameter_maps = {'self': {}}
    parameter_user_map = {}

    _parameter_space = None

    @property
    def parameter_space(self):
        return self._parameter_space

    @parameter_space.setter
    def parameter_space(self, ps):
        assert dict(self.parameter_type) == dict(ps.parameter_type)
        self._parameter_space = ps

    @property
    def parametric(self):
        return len(self.parameter_type) > 0

    def parse_parameter(self, mu):
        return parse_parameter(mu, self.parameter_type)

    def map_parameter(self, mu, target='self', component=None):
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

        Returns
        -------
        The mapped `Parameter`.
        '''
        mu = self.parse_parameter(mu)
        mu_global = Parameter()
        for k in self.global_parameter_type:
            mu_global[k] = mu[self.parameter_user_map[k]] if k in self.parameter_user_map else mu[k]
        if component is None:
            parameter_map = self.parameter_maps[target]
        else:
            parameter_map = self.parameter_maps[target][component]
        mu_mapped = Parameter()
        for k, (v, m) in parameter_map.iteritems():
            mu_mapped[k] = mu_global[v][component, ...] if m else mu_global[v]
        return mu_mapped

    def build_parameter_type(self, local_type=OrderedDict(), inherits=OrderedDict(), local_global=False):
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
        if local_global:
            global_type = local_type.copy()
            parameter_maps['self'] = OrderedDict((k, (k, False)) for k in local_type.keys())
        else:
            parameter_map = OrderedDict((k, ('.{}'.format(k), False)) for k in local_type.keys())
            global_type = OrderedDict(('.{}'.format(k), v) for k, v in local_type.iteritems())
            parameter_maps['self'] = parameter_map

        for n in inherits:
            if inherits[n] is None:
                parameter_maps[n] = OrderedDict()
                continue
            if isinstance(inherits[n], tuple) or isinstance(inherits[n], list):
                merged_param_map = OrderedDict()
                for k, v in inherits[n][0].parameter_type.iteritems():
                    if (k.startswith('.') and
                            all(v == inherits[n][i].parameter_type[k] for i in xrange(1, len(inherits[n])))):
                        global_name = '.{}{}'.format(n, k)
                        global_type[global_name] = v
                        merged_param_map[k] = (global_name, True)
                parameter_map_list = list()
                for i in xrange(len(inherits[n])):
                    pt = inherits[n][i].parameter_type
                    parameter_map = merged_param_map.copy()
                    for k, v in pt.iteritems():
                        if k in merged_param_map:
                            next
                        if k.startswith('.'):
                            global_name = '.{}_{}{}'.format(n, i, k)
                            global_type[global_name] = v
                            parameter_map[k] = (global_name, False)
                        else:
                            if k in global_type and global_type[k] != v:
                                raise ValueError('Component dimensions of global name {} do not match'.format(k))
                            global_type[k] = v
                            parameter_map[k] = (k, False)
                    parameter_map_list.append(parameter_map)
                parameter_maps[n] = parameter_map_list
            else:
                parameter_map = OrderedDict()
                for k, v in inherits[n].parameter_type.iteritems():
                    if k.startswith('.'):
                        global_name = '.{}{}'.format(n, k)
                        global_type[global_name] = v
                        parameter_map[k] = (global_name, False)
                    else:
                        if k in global_type and global_type[k] != v:
                            raise ValueError('Component dimensions of global name {} do not match'.format(k))
                        global_type[k] = v
                        parameter_map[k] = (k, False)
                parameter_maps[n] = parameter_map

        self.global_parameter_type = global_type
        self.parameter_type = global_type
        self.parameter_maps = parameter_maps

    def rename_parameter(self, name_map):
        '''Rename a parameter component of the object's parameter type.

        This method can be called by the object's owner to rename local parameter components
        (whose name begins with '.') to global parameter components.

        Parameters
        ----------
        name_map
            A dict of the form `{'.oldname1': 'newname1', ...}` defining the name mapping.
        '''
        for k, v in name_map.iteritems():
            if not k.startswith('.'):
                raise ValueError('Can only rename parameters with local name (starting with ".")')
            if not k in self.global_parameter_type:
                raise ValueError('There is no parameter named {}'.format(k))
            if k in self.parameter_user_map:
                raise ValueError('{} has already been renamed to {}'.format(k, self.parameter_user_map[k]))
            self.parameter_user_map[k] = v
        parameter_type = OrderedDict()
        for k, v in self.global_parameter_type.iteritems():
            if k in self.parameter_user_map:
                k = self.parameter_user_map[k]
            if k in parameter_type and parameter_type[k] != v:
                raise ValueError('Mismatching shapes for parameter {}: {} and {}'.format(k, parameter_type[k], v))
            parameter_type[k] = v
        self.parameter_type = parameter_type

    def parameter_info(self):
        '''Return an info string about the object's parameter type and how it is built.'''

        msg = 'The parameter_type is: {}\n\n'.format(self.parameter_type)
        msg += 'We have the following parameter-maps:\n\n'
        for n, mp in self.parameter_maps.iteritems():
            if not isinstance(mp, list):
                pad = ' ' * (len(n) + 2)
                msg += '{}: '.format(n)
                for k, (v, f) in mp.iteritems():
                    msg += '{} <- {}'.format(k, v)
                    if v in self.parameter_user_map:
                        msg += ' <- {}'.format(self.parameter_user_map[k])
                    msg += '\n' + pad
            else:
                for i, m in mp.enumerate():
                    tag = '{}[{}]: '.format(n, i)
                    pad = ' ' * len(tag)
                    msg += tag
                    for k, (v, f) in m.iteritems():
                        msg += '{} <- {}'.format(k, v)
                        if f:
                            msg += '[{}, ...]'.format(i)
                        if v in self.parameter_user_map:
                            msg += ' <- {}'.format(self.parameter_user_map[k])
                        msg += '\n' + pad
            msg += '\n'
        return msg
