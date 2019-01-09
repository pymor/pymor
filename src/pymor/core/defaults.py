# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module contains pyMOR's facilities for handling default values.

A default value in pyMOR is always the default value of some
function argument. To mark the value of an optional function argument
as a user-modifiable default value use the :func:`defaults` decorator.
As an additional feature, if `None` is passed for such an argument,
its default value is used instead of `None`. This is useful
for writing code of the following form::

    @default('option')
    def algorithm(U, option=42):
        ...

    def method_called_by_user(V, option_for_algorithm=None):
        ...
        algorithm(U, option=option_for_algorithm)
        ...

If the user does not provide `option_for_algorithm` to
`method_called_by_user`, the default `42` is automatically chosen
without the implementor of `method_called_by_user` having to care
about this.

The user interface for handling default values in pyMOR is provided
by :func:`set_defaults`, :func:`load_defaults_from_file`,
:func:`write_defaults_to_file` and :func:`print_defaults`.

If pyMOR is imported, it will automatically search for a configuration
file named `pymor_defaults.py` in the current working directory.
If found, the file is loaded via :func:`load_defaults_from_file`.
However, as a security precaution, the file will only be loaded if it is
owned by the user running the Python interpreter
(:func:`load_defaults_from_file` uses `exec` to load the configuration).
As an alternative, the environment variable `PYMOR_DEFAULTS` can be
used to specify the path of a configuration file. If empty or set to
`NONE`, no configuration file will be loaded whatsoever.

.. _defaults_warning:
.. warning::
   The state of pyMOR's global defaults enters the calculation of each
   |state id|. Thus, if you first instantiate an immutable object and
   then change the defaults, the resulting object will have a different
   |state id| than if you first change the defaults. (This is necessary
   as the object can save internal state upon initialization, which
   depends on the state of the global defaults.) As a consequence, the
   key generated for :mod:`caching <pymor.core.cache>` will depend on the
   time the defaults have been changed. While no wrong results will be
   produced, changing defaults at different times will cause unnecessary
   cache misses and will pollute the cache with duplicate entries.

   An exemption from this rule are defaults which are listed in the
   `sid_ignore` argument of the :func:`defaults` decorator. Such
   defaults will not enter the |state id| calculation. This allows the
   user to change defaults related to input/output, e.g.
   :mod:`logging <pymor.core.logger>`, without breaking caching.
   Before marking defaults as ignored in your own code, however, make
   sure to double check that these defaults will not affect the result
   of any mathematical algorithm.
"""

from collections import defaultdict, OrderedDict
import functools
import importlib
import inspect
import pkgutil
import textwrap

from pymor.tools.table import format_table


_default_container = None


class DefaultContainer(object):
    """Internal singleton class holding all default values defined in pyMOR.

    Not to be used directly.
    """

    def __new__(cls):
        global _default_container
        if _default_container is not None:
            raise ValueError('DefaultContainer is a singleton! Use pymor.core.defaults._default_container.')
        else:
            return object.__new__(cls)

    def __init__(self):
        self._data = defaultdict(dict)
        self.registered_functions = set()

    def _add_defaults_for_function(self, func, args, sid_ignore):

        if func.__doc__ is not None:
            new_docstring = inspect.cleandoc(func.__doc__)
            new_docstring += '''

Defaults
--------
'''
            new_docstring += '\n'.join(textwrap.wrap(', '.join(args), 80)) + '\n(see :mod:`pymor.core.defaults`)'
            func.__doc__ = new_docstring

        params = OrderedDict(inspect.signature(func).parameters)
        argnames = tuple(params.keys())
        defaultsdict = {}
        for n in args:
            p = params.get(n, None)
            if p is None:
                raise ValueError("Decorated function has no argument '{}'".format(n))
            if p.default is p.empty:
                raise ValueError("Decorated function has no default for argument '{}'".format(n))
            defaultsdict[n] = p.default

        path = func.__module__ + '.' + getattr(func, '__qualname__', func.__name__)
        if path in self.registered_functions:
            raise ValueError('Function with name {} already registered for default values!'.format(path))
        self.registered_functions.add(path)
        for k, v in defaultsdict.items():
            self._data[path + '.' + k]['func'] = func
            self._data[path + '.' + k]['code'] = v
            self._data[path + '.' + k]['sid_ignore'] = k in sid_ignore

        defaultsdict = {}
        for k in self._data:
            if k.startswith(path + '.'):
                defaultsdict[k.split('.')[-1]] = self.get(k)[0]

        func.argnames = argnames
        func.defaultsdict = defaultsdict
        self._update_function_signature(func)

    def _update_function_signature(self, func):
        sig = inspect.signature(func)
        params = OrderedDict(sig.parameters)
        for n, v in func.defaultsdict.items():
            params[n] = params[n].replace(default=v)
        func.__signature__ = sig.replace(parameters=params.values())

    def update(self, defaults, type='user'):
        if hasattr(self, '_sid'):
            del self._sid
        assert type in ('user', 'file')

        functions_to_update = set()

        for k, v in defaults.items():
            k_parts = k.split('.')

            func = self._data[k].get('func', None)
            if not func:
                head = k_parts[:-2]
                while head:
                    try:
                        importlib.import_module('.'.join(head))
                        break
                    except ImportError:
                        head = head[:-1]
            func = self._data[k].get('func', None)
            if not func:
                del self._data[k]
                raise KeyError(k)

            self._data[k][type] = v
            argname = k_parts[-1]
            func.defaultsdict[argname] = v
            functions_to_update.add(func)

        for func in functions_to_update:
            self._update_function_signature(func)

    def get(self, key):
        values = self._data[key]
        if 'user' in values:
            return values['user'], 'user', values['sid_ignore']
        elif 'file' in values:
            return values['file'], 'file', values['sid_ignore']
        elif 'code' in values:
            return values['code'], 'code', values['sid_ignore']
        else:
            raise ValueError('No default value matching the specified criteria')

    def __getitem__(self, key):
        assert isinstance(key, str)
        self.get(key)[0]

    def keys(self):
        return self._data.keys()

    def import_all(self):
        packages = {k.split('.')[0] for k in self._data.keys()}.union({'pymor'})
        for package in packages:
            _import_all(package)

    @property
    def sid(self):
        sid = getattr(self, '_sid', None)
        if not sid:
            from pymor.core.interfaces import generate_sid
            user_dict = {k: v['user'] if 'user' in v else v['file']
                         for k, v in self._data.items() if 'user' in v or 'file' in v and not v['sid_ignore']}
            self._sid = sid = generate_sid(user_dict)
        return sid


_default_container = DefaultContainer()


def defaults(*args, sid_ignore=()):
    """Function decorator for marking function arguments as user-configurable defaults.

    If a function decorated with :func:`defaults` is called, the values of the marked
    default parameters are set to the values defined via :func:`load_defaults_from_file`
    or :func:`set_defaults` in case no value has been provided by the caller of the function.
    Moreover, if `None` is passed as a value for a default argument, the argument
    is set to its default value, as well. If no value has been specified using
    :func:`set_defaults` or :func:`load_defaults_from_file`, the default value provided in
    the function signature is used.

    If the argument `arg` of function `f` in sub-module `m` of package `p` is
    marked as a default value, its value will be changeable by the aforementioned
    methods under the path `p.m.f.arg`.

    Note that the `defaults` decorator can also be used in user code.

    Parameters
    ----------
    args
        List of strings containing the names of the arguments of the decorated
        function to mark as pyMOR defaults. Each of these arguments has to be
        a keyword argument (with a default value).
    sid_ignore
        List of strings naming the defaults in `args` which should not enter
        |state id| calculation (because they do not affect the outcome of any
        computation). Such defaults will typically be IO related. Use with
        extreme caution!
    """
    assert all(isinstance(arg, str) for arg in args)

    def the_decorator(func):

        if not args:
            return func

        global _default_container
        _default_container._add_defaults_for_function(func, args=args, sid_ignore=sid_ignore)

        @functools.wraps(func, updated=())  # ensure that __signature__ is not copied
        def wrapper(*wrapper_args, **wrapper_kwargs):
            for k, v in zip(func.argnames, wrapper_args):
                if k in wrapper_kwargs:
                    raise TypeError("{} got multiple values for argument '{}'"
                                    .format(func.__name__, k))
                wrapper_kwargs[k] = v
            wrapper_kwargs = {k: v if v is not None else func.defaultsdict.get(k, None)
                              for k, v in wrapper_kwargs.items()}
            wrapper_kwargs = dict(func.defaultsdict, **wrapper_kwargs)
            return func(**wrapper_kwargs)

        return wrapper

    return the_decorator


def _import_all(package_name='pymor'):

    package = importlib.import_module(package_name)

    if hasattr(package, '__path__'):
        def onerror(name):
            from pymor.core.logger import getLogger
            logger = getLogger('pymor.core.defaults._import_all')
            logger.warning('Failed to import ' + name)

        for p in pkgutil.walk_packages(package.__path__, package_name + '.', onerror=onerror):
            try:
                importlib.import_module(p[1])
            except ImportError:
                from pymor.core.logger import getLogger
                logger = getLogger('pymor.core.defaults._import_all')
                logger.warning('Failed to import ' + p[1])


def print_defaults(import_all=True, shorten_paths=2):
    """Print all |default| values set in pyMOR.

    Parameters
    ----------
    import_all
        While :func:`print_defaults` will always print all defaults defined in
        loaded configuration files or set via :func:`set_defaults`, default
        values set in the function signature can only be printed after the
        modules containing these functions have been imported. If `import_all`
        is set to `True`, :func:`print_defaults` will therefore first import all
        of pyMOR's modules, to provide a complete lists of defaults.
    shorten_paths
        Shorten the paths of all default values by `shorten_paths` components.
        The last two path components will always be printed.
    """

    if import_all:
        _default_container.import_all()

    keys = ([], [])
    values = ([], [])
    comments = ([], [])

    for k in sorted(_default_container.keys()):
        v, c, i = _default_container.get(k)
        k_parts = k.split('.')
        if len(k_parts) >= shorten_paths + 2:
            keys[int(i)].append('.'.join(k_parts[shorten_paths:]))
        else:
            keys[int(i)].append('.'.join(k_parts))
        values[int(i)].append(repr(v))
        comments[int(i)].append(c)
    key_string = 'path (shortened)' if shorten_paths else 'path'

    for i, (ks, vls, cs) in enumerate(zip(keys, values, comments)):
        description = 'defaults not affecting state id calculation' if i else 'defaults affecting state id calcuation'
        rows = [[key_string, 'value', 'source']] + list(zip(ks, vls, cs))
        print(format_table(rows, title=description))
        if not i:
            print()
            print()
        print()


def write_defaults_to_file(filename='./pymor_defaults.py', packages=('pymor',)):
    """Write the currently set |default| values to a configuration file.

    The resulting file is an ordinary Python script and can be modified
    by the user at will. It can be loaded in a later session using
    :func:`load_defaults_from_file`.

    Parameters
    ----------
    filename
        Name of the file to write to.
    packages
        List of package names.
        To discover all default values that have been defined using the
        :func:`defaults` decorator, `write_defaults_to_file` will
        recursively import all sub-modules of the named packages before
        creating the configuration file.
    """

    for package in packages:
        _import_all(package)

    keys = ([], [])
    values = ([], [])
    as_comment = ([], [])

    for k in sorted(_default_container.keys()):
        v, c, i = _default_container.get(k)
        keys[int(i)].append("'" + k + "'")
        values[int(i)].append(repr(v))
        as_comment[int(i)].append(c == 'code')
    key_width = max(max([0] + list(map(len, ks))) for ks in keys)

    with open(filename, 'wt') as f:
        print('''
# pyMOR defaults config file
# This file has been automatically created by pymor.core.defaults.write_defaults_to_file'.

d = {}
'''[1:], file=f)

        for i, (ks, vls, cs) in enumerate(zip(keys, values, as_comment)):

            if i:
                print('''
########################################################################
#                                                                      #
# SETTING THE FOLLOWING DEFAULTS WILL NOT AFFECT STATE ID CALCULATION. #
#                                                                      #
########################################################################
'''[1:], file=f)
            else:
                print('''
########################################################################
#                                                                      #
# SETTING THE FOLLOWING DEFAULTS WILL AFFECT STATE ID CALCULATION.     #
#                                                                      #
########################################################################
'''[1:], file=f)

            lks = ks[0].split('.')[:-1] if ks else ''
            for c, k, v in zip(cs, ks, vls):
                ks = k.split('.')[:-1]
                if lks != ks:
                    print('', file=f)
                lks = ks

                print('{}d[{:{key_width}}] = {}'.format('# ' if c else '', k, v,
                                                        key_width=key_width),
                      file=f)

            print(file=f)
            print(file=f)

    print('Written defaults to file ' + filename)


def load_defaults_from_file(filename='./pymor_defaults.py'):
    """Loads |default| values defined in configuration file.

    Suitable configuration files can be created via :func:`write_defaults_to_file`.
    The file is loaded via Python's :func:`exec` function, so be very careful
    with configuration files you have not created your own. You have been
    warned!

    Note that defaults should generally only be changed/loaded before
    |state ids| have been calculated. See this :ref:`warning <defaults_warning>`
    for details.

    Parameters
    ----------
    filename
        Path of the configuration file.
    """
    env = {}
    exec(open(filename, 'rt').read(), env)
    try:
        _default_container.update(env['d'], type='file')
    except KeyError as e:
        raise KeyError('Error loading defaults from file. Key {} does not correspond to a default'.format(e))


def set_defaults(defaults):
    """Set |default| values.

    This method sets the default value of function arguments marked via the
    :func:`defaults` decorator, overriding default values specified in the
    function signature or set earlier via :func:`load_defaults_from_file` or
    previous :func:`set_defaults` calls.

    Note that defaults should generally only be changed/loaded before state ids
    have been calculated. See this :ref:`warning <defaults_warning>` for details.

    Parameters
    ----------
    defaults
        Dictionary of default values. Keys are the full paths of the default
        values (see :func:`defaults`).
    """
    try:
        _default_container.update(defaults, type='user')
    except KeyError as e:
        raise KeyError('Error setting defaults. Key {} does not correspond to a default'.format(e))


def defaults_sid():
    """Return a |state id| for pyMOR's global |defaults|.

    This method is used for the calculation of |state ids| of |immutable|
    objects and for :mod:`~pymor.core.cache` key generation.
    """
    return _default_container.sid
