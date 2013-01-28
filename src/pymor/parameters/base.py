from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import numpy as np

import pymor.core as core


class Parameter(OrderedDict):
    def allclose(self, mu):
        if set(self.keys()) != set(mu.keys()):
            return False
        for k,v in self.iteritems():
            if not np.allclose(v, mu[k]):
                return False
        return True

    def copy(self):
        c = Parameter()
        for k,v in self.iteritems():
            c[k] = v.copy()
        return c


class Parametric(object):

    parameter_type = OrderedDict()
    global_parameter_type = OrderedDict()
    local_parameter_type = OrderedDict()
    parameter_maps = {'self':{}}
    parameter_user_map = {}

    def parse_parameter(self, mu):
        if not isinstance(mu, dict):
            if isinstance(mu, (tuple, list)):
                if len(mu) != 1 or not isinstance(mu[0], (tuple, list, np.ndarray)):
                    mu = (mu, )
            else:
                mu = (mu, )
            if len(mu) != len(self.parameter_type):
                raise ValueError('Parameter length does not match.')
            mu = Parameter(zip(self.parameter_type.keys(), mu))
        elif set(mu.keys()) != set(self.parameter_type.keys()):
            raise ValueError('Components do not match')
        else:
            mu = Parameter((k, mu[k]) for k in self.parameter_type)
        for k, v in mu.iteritems():
            if not isinstance(v, np.ndarray):
                mu[k] = np.array(v, ndmin=1)
        if not all(mu[k].shape == self.parameter_type[k] for k in mu.keys()):
            raise ValueError('Component dimensions do not match')
        return mu

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

    def set_parameter_type(self, local_type=OrderedDict(), inherits=OrderedDict(), local_global=False):
        local_type = OrderedDict(local_type)
        for k,v in local_type.iteritems():
            if not isinstance(v, tuple):
                local_type[k] = tuple((v,))
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
                next
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
