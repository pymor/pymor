from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import numpy as np

import pymor.core as core
from pymor.la import float_cmp_all
from .interfaces import ParameterSpaceInterface


class Parameter(OrderedDict):
    def allclose(self, mu):
        if set(self.keys()) != set(mu.keys()):
            return False
        for k,v in self.iteritems():
            if not float_cmp_all(v, mu[k]):
                return False
        return True

    def copy(self):
        c = Parameter()
        for k,v in self.iteritems():
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
    if not isinstance(mu, dict):
        if isinstance(mu, (tuple, list)):
            if len(parameter_type) == 1 and len(mu) != 1:
                mu = (mu, )
        else:
            mu = (mu, )
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
    if isinstance(parameter_type, ParameterSpaceInterface):
        return OrderedDict(parameter_type.parameter_type)
    parameter_type = OrderedDict(parameter_type)
    for k,v in parameter_type.iteritems():
        if not isinstance(v, tuple):
            if v == 0 or v == 1:
                parameter_type[k] = tuple()
            else:
                parameter_type[k] = tuple((v,))
    return parameter_type


class Parametric(object):

    parameter_type = OrderedDict()
    global_parameter_type = OrderedDict()
    local_parameter_type = OrderedDict()
    parameter_maps = {'self':{}}
    parameter_user_map = {}

    _parameter_space = None

    @property
    def parameter_space(self):
        return self._parameter_space

    @parameter_space.setter
    def parameter_space(self, ps):
        assert dict(self.parameter_type) == dict(ps.parameter_type)
        self._parameter_space = ps

    def parse_parameter(self, mu):
        return parse_parameter(mu, self.parameter_type)

    def map_parameter(self, mu, target='self', component=None):
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
            mu_mapped[k] = mu_global[v][component,...] if m else mu_global[v]
        return mu_mapped

    def build_parameter_type(self, local_type=OrderedDict(), inherits=OrderedDict(), local_global=False):
        local_type = parse_parameter_type(local_type)
        self.local_parameter_type = local_type
        parameter_maps = {}
        if local_global:
            global_type = local_type.copy()
            parameter_maps['self'] = OrderedDict((k,(k, False)) for k in local_type.keys())
        else:
            parameter_map = OrderedDict((k,('.{}'.format(k), False)) for k in local_type.keys())
            global_type = OrderedDict(('.{}'.format(k), v) for k, v in local_type.iteritems())
            parameter_maps['self'] = parameter_map

        for n in inherits:
            if inherits[n] is None:
                parameter_maps[n] = OrderedDict()
                continue
            if isinstance(inherits[n], tuple) or isinstance(inherits[n], list):
                merged_param_map = OrderedDict()
                for k,v in inherits[n][0].parameter_type.iteritems():
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
        msg = 'The parameter_type is: {}\n\n'.format(self.parameter_type)
        msg += 'We have the following parameter-maps:\n\n'
        for n,mp in self.parameter_maps.iteritems():
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
