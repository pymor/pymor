# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from collections import defaultdict
import functools
import inspect
import pkgutil


class DefaultContainer(object):

    def __init__(self):
        self._data = defaultdict(dict)

    def _add_defaults_for_function(self, defaultsdict, func):
        path = func.__module__ + '.' + func.__name__ + '.'
        for k, v in defaultsdict.iteritems():
            self._data[path + k]['code'] = v

    def _add_wrapper_function(self, func):
        path = func.__module__ + '.' + func.__name__ + '.'
        for k, v in self._data.iteritems():
            if k.startswith(path):
                v['func'] = func

    def update(self, defaults, type='user'):
        assert type in ('user', 'file')
        import pymor.core.interfaces
        if pymor.core.interfaces.ImmutableMeta.sids_created:
            raise ValueError('''Defaults have been locked after calculation of the first state id.
This is for you own good! (Changing defaults after state ids have been calculated will
break caching.) Please set defaults before importing further parts of pyMOR!''')
        for k, v in defaults.iteritems():
            self._data[k][type] = v
            func = self._data[k].get('func', None)
            if func:
                argname = k.split('.')[-1]
                func._defaultsdict[argname] = v
                argspec = inspect.getargspec(func)
                argind = argspec.args.index(argname) - len(argspec.args)
                defaults = list(argspec.defaults)
                defaults[argind] = v
                func.__defaults__ = tuple(defaults)

    def get(self, key, code=True, file=True, user=True):
        values = self._data[key]
        if user and 'user' in values:
            return values['user'], 'user'
        elif file and 'file' in values:
            return values['file'], 'file'
        elif code and 'code' in values:
            return values['code'], 'code'
        else:
            raise ValueError('No default value matching the specified criteria')

    def __getitem__(self, key):
        if isinstance(key, str):
            self.get(key)[0]
        else:
            path = key.__module__ + '.' + key.__name__ + '.'
            result = {}
            for k in self._data:
                if k.startswith(path):
                    result[k.split('.')[-1]] = self.get(k)[0]
            return result

    def keys(self):
        return self._data.keys()

    def import_all(self):
        packages = set(k.split('.')[0] for k in self._data.keys())
        for package in packages:
            _import_all(package)

    def check_consistency(self):
        self.import_all()
        for k, v in self._data.iteritems():
            if 'user' in v and not 'code' in v:
                raise ValueError('undefined default provided by user: ' + k)
            elif 'file' in v and not 'code' in v:
                raise ValueError('undefined default provided in config file: ' + k)


_default_container = DefaultContainer()


def defaults(**kwargs):
    #FIXME this will have to be adapted for Python 3

    def the_decorator(func):

        global _default_container
        _default_container._add_defaults_for_function(kwargs, func)
        defaultsdict = _default_container[func]

        defaults = list(func.__defaults__)
        argspec = inspect.getargspec(func)
        argnames = argspec.args

        defaultsdict_copy = defaultsdict.copy()
        def get_default(name, signature_default):
            return defaultsdict_copy.pop(name, signature_default)

        new_defaults = tuple(get_default(n, v) for n, v in zip(argnames[-len(defaults):], defaults))

        if defaultsdict_copy:
            raise ValueError('Wrapped function misses the following arguments, for which defaults are provided: \n    '
                             + str(defaultsdict_copy.keys()))

        argstring = ', '.join(argnames[:-len(new_defaults)])
        if new_defaults:
            argstring += ', ' + ', '.join('{}={}'.format(k, v) for k, v in zip(argnames[-len(new_defaults):],
                                                                               new_defaults))
        if argspec.varargs:
            argstring += ', *' + argspec.varargs
        if argspec.keywords:
            argstring += ', **' + argspec.keywords

        wrapper_code = '''
def {0}({1}):
    loc = locals()
    argdict = {{arg: loc[arg] if loc[arg] is not None else defaultsdict[arg] for arg in argnames}}
    return func(**argdict)

global __wrapper_func
__wrapper_func = {0}
        '''.format(func.__name__, argstring)

        exec wrapper_code in locals()
        wrapper = functools.wraps(func)(__wrapper_func)

        # On Python 2 we have to add the __wrapped__ attribute to the wrapper
        # manually to help IPython find the right source code location
        wrapper.__wrapped__ = func

        # add defaultsdict to the function object, so that we can change it later
        # on if we wish
        wrapper._defaultsdict = defaultsdict

        _default_container._add_wrapper_function(wrapper)

        return wrapper

    return the_decorator


def _import_all(package_name = 'pymor'):

    package = __import__(package_name)

    def onerror(name):
        raise ImportError('Failed to import package ' + name)

    for p in pkgutil.walk_packages(package.__path__, package_name + '.', onerror=onerror):
        __import__(p[1])


def print_defaults(import_all=True, code=True, file=True, user=True,
                   max_key_components=2):

    if import_all:
        _default_container.import_all()

    keys = []
    values = []
    comments = []
    shortened = False
    for k in sorted(_default_container.keys()):
        try:
            v, c = _default_container.get(k, code=code, file=file, user=user)
        except ValueError:
            continue
        ks = k.split('.')
        if len(ks) > max_key_components:
            shortened = True
        keys.append('.'.join(ks[-min(len(ks), max_key_components):]))
        values.append(repr(v))
        comments.append(c)
    key_width = max(map(len, keys))
    value_width = max(map(len, values))

    key_string = 'key (shortened)' if shortened else 'key'
    header = '''
{:{key_width}}   {:{value_width}}   source'''[1:].format(key_string, 'value',
                                                         key_width=key_width, value_width=value_width)

    print(header)
    print('=' * len(header))

    lks = keys[0].split('.')[:-1]
    for k, v, c in zip(keys, values, comments):

        ks = k.split('.')[:-1]
        if lks != ks:
            print('')
        lks = ks

        print('{:{key_width}}   {:{value_width}}   {}'.format(k, v, c,
                                                              key_width=key_width,
                                                              value_width=value_width))


def write_defaults_to_file(path='./pymor_defaults.py', packages=('pymor'), code=True, file=True, user=True):

    for package in packages:
        _import_all(package)

    comments_map = {'code': 'from source code',
                    'file': 'from config file',
                    'user': 'defined by user'}

    keys = []
    values = []
    comments = []
    for k in sorted(_default_container.keys()):
        keys.append("'" + k + "'")
        try:
            v, c = _default_container.get(k, code=code, file=file, user=user)
        except ValueError:
            continue
        values.append(repr(v))
        comments.append(comments_map[c])
    key_width = max(map(len, keys))
    value_width = max(map(len, values))

    with open(path, 'w') as f:
        print('''
# pyMOR defaults config file
# This file has be automatically created by pymor.core.defaults.write_defaults_to_file'.

d = {}
'''[1:], file=f)

        lks = keys[0].split('.')[:-1]
        for k, v, c in zip(keys, values, comments):

            ks = k.split('.')[:-1]
            if lks != ks:
                print('', file=f)
            lks = ks

            print('d[{:{key_width}}] = {:{value_width}}  # {}'.format(k, v, c,
                                                                      key_width=key_width,
                                                                      value_width=value_width),
                  file=f)


def load_defaults_from_file(path='./pymor_defaults.py'):
    env = {}
    exec open(path).read() in env
    _default_container.update(env['d'], type='file')


def set_defaults(defaults, check=True):
    _default_container.update(defaults, type='user')
    if check:
        _default_container.check_consistency()
