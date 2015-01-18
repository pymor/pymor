# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module contains pyMOR's facilities for handling default values.

A default value in pyMOR is always the default value of some
function argument. To mark the value of an optional function argument
as an user-modifiable default value, use the :func:defaults: decorator.
As an additional feature, if `None` is passed as value for such
an argument, its default value is used instead of `None`. This is useful
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

If pyMOR is imported, it will automatically search for configuration
files named `pymor_defaults.py` in the current working directory.
The first file found will be loaded via :func:`load_defaults_from_file`.
However, for your security, this file will only be loaded, if it is
owned by the user running the Python iterpreter.
(:func:`load_defaults_from_file` uses `exec` to load the configuration.)
As an alternative, the environment variable `PYMOR_DEFAULTS` can be
used to specify the path of a configuration file. If empty or set to
`NONE`, no configuration file will be loaded whatsoever.

.. _defaults_warning:
.. warning::
   The state of pyMOR's global defaults enters the calculation of each
   state id (see :mod:`pymor.core.interfaces`). Thus, if you first
   instantiate an immutable object and then change the defaults, the
   resulting object will have a different state id than if you first
   change the defaults.  (This is necessary as the object can save
   internal state upon initialization, which depends on the state of
   the global defaults.) As a consequence, the key generated for
   :mod:`caching <pymor.core.cache>` will depend on the time the
   defaults have been changed. While no wrong results will be produced,
   changing defaults at different times will cause unnecessary cache
   misses and will pollute the cache with duplicate entries.

   As a rule of thumb, defaults should be set once and for all at the
   start of a pyMOR application. Anything else should be considered
   a dirty hack. (pyMOR will warn you if it gets the impression you
   are trying to hack it.)
"""

from __future__ import absolute_import, division, print_function

from collections import defaultdict
import functools
import inspect
import pkgutil
import hashlib
import textwrap

_default_container = None
_default_container_sid = None


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
        self.registered_functions = {}

    def _add_defaults_for_function(self, defaultsdict, func, qualname=None):
        path = qualname or getattr(func, '__qualname__', func.__module__ + '.' + func.__name__)
        if path in self.registered_functions:
            raise ValueError('''Function with name {} already registered for default values!
For Python 2 compatibility, please supply the '_qualname' parameter when decorating
methods of classes!'''.format(path))
        for k, v in defaultsdict.iteritems():
            self._data[path + '.' + k]['code'] = v

        result = {}
        for k in self._data:
            if k.startswith(path + '.'):
                result[k.split('.')[-1]] = self.get(k)[0]
        return result

    def _add_wrapper_function(self, func, qualname=None):
        path = qualname or getattr(func, '__qualname__', func.__module__ + '.' + func.__name__)
        self.registered_functions[path] = func
        split_path = path.split('.')
        for k, v in self._data.iteritems():
            if k.split('.')[:-1] == split_path:
                v['func'] = func

    def update(self, defaults, type='user'):
        assert type in ('user', 'file')
        import pymor.core.interfaces
        if pymor.core.interfaces.ImmutableMeta.sids_created:
            from pymor.core.logger import getLogger
            getLogger('pymor.core.defaults').warn(
                'Changing defaults after calculation of the first state id. '
                + '(see pymor.core.defaults for more information.)')
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
        self._calc_sid()

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
        assert isinstance(key, str)
        self.get(key)[0]

    def keys(self):
        return self._data.keys()

    def import_all(self):
        packages = set(k.split('.')[0] for k in self._data.keys()).union({'pymor'})
        for package in packages:
            _import_all(package)

    def check_consistency(self, delete=False):
        self.import_all()
        from pymor.core.logger import getLogger
        logger = getLogger('pymor.core.defaults')
        keys_to_delete = []

        for k, v in self._data.iteritems():
            if ('user' in v or 'file' in v) and 'code' not in v:
                keys_to_delete.append(k)

        if delete:
            for k in keys_to_delete:
                del self._data[k]

        for k in keys_to_delete:
            logger.warn('Undefined default value: ' + k + (' (deleted)' if delete else ''))

        return len(keys_to_delete) > 0

    def _calc_sid(self):
        global _default_container_sid
        user_dict = {k: v['user'] if 'user' in v else v['file']
                     for k, v in self._data.items() if 'user' in v or 'file' in v}
        _default_container_sid = hashlib.sha256(repr(sorted(user_dict.items()))).digest()


_default_container = DefaultContainer()


def defaults(*args, **kwargs):
    """Function decorator for marking function arguments as user-configurable defaults.

    If a function decorated with `defaults` is called, the values of the marked
    default parameters are set to the values defined via :func:`load_defaults_from_file`
    or :func:`set_defaults` if no value is provided by the caller of the function.
    If no value is specified using these methods, the default value provided by in the
    function signature is used.

    Moreover, if `None` is passed as a value for a default argument, the argument
    is set to its default value, as well.

    If the argument `arg` of function `f` in sub-module `m` of package `p` is
    marked as a default value, its value will be changeable by the aforementioned
    methods under the path `p.m.f.arg`.

    The `defaults` decorator can also be used for user code.

    Parameters
    ----------
    args
        List of strings containing the names of the arguments of the decorated
        function to mark as pyMOR defaults. Each of these arguments has to be
        a keyword argument (with a default value).
    qualname
        If a method of a class is decorated, the fully qualified name of the
        method should be provided, as this name cannot be derived at decoration
        time in Python 2.
    """
    # FIXME this will have to be adapted for Python 3

    assert all(isinstance(arg, str) for arg in args)
    assert set(kwargs.keys()) <= {'qualname'}
    qualname = kwargs.get('qualname', None)

    def the_decorator(func):

        if not args:
            return func

        if func.__doc__ is not None:
            new_docstring = inspect.cleandoc(func.__doc__)
            new_docstring += '''

Defaults
--------
'''
            new_docstring += '\n'.join(textwrap.wrap(', '.join(args), 80)) + '\n(see :mod:`pymor.core.defaults`)'
            func.__doc__ = new_docstring

        defaults = func.__defaults__
        if not defaults:
            raise ValueError('Wrapped function has no optional arguments at all!')
        defaults = list(defaults)
        argspec = inspect.getargspec(func)
        argnames = argspec.args

        if not set(args) <= set(argnames):
            raise ValueError('Decorated function has no arguments named: '
                             + ', '.join(set(args) - set(argnames)))

        if not set(args) <= set(argnames[-len(defaults):]):
            raise ValueError('Decorated function has no defaults for arguments named: '
                             + ', '.join(set(args) - set(argnames[-len(defaults):])))

        defaultsdict = {}
        for n, v in zip(argnames[-len(defaults):], defaults):
            if n in args:
                defaultsdict[n] = v

        global _default_container
        defaultsdict = _default_container._add_defaults_for_function(defaultsdict, func, qualname=qualname)

        new_defaults = tuple(defaultsdict.get(n, v) for n, v in zip(argnames[-len(defaults):], defaults))

        argstring_parts = []
        argstring_parts.extend(argnames[:-len(new_defaults)])
        argstring_parts.extend('{}={}'.format(k, repr(v)) for k, v in zip(argnames[-len(new_defaults):], new_defaults))
        if argspec.varargs:
            argstring_parts.append('*' + argspec.varargs)
        if argspec.keywords:
            argstring_parts.append('**' + argspec.keywords)
        argstring = ', '.join(argstring_parts)

        wrapper_code = '''
def {0}({1}):
    loc = locals()
    argdict = {{arg: loc[arg] if loc[arg] is not None else defaultsdict.get(arg, None) for arg in argnames}}
    return wrapped_func(**argdict)
        '''.format(func.__name__, argstring)

        if func.__name__ in ('wrapped_func', 'argname', 'defaultsdict'):
            raise ValueError('Functions decorated with @default may not have the name ' + func.__name__)
        wrapper_globals = {'wrapped_func': func, 'argnames': argnames, 'defaultsdict': defaultsdict}
        exec wrapper_code in wrapper_globals
        wrapper = functools.wraps(func)(wrapper_globals[func.__name__])

        # On Python 2 we have to add the __wrapped__ attribute to the wrapper
        # manually to help IPython find the right source code location
        wrapper.__wrapped__ = func

        # add defaultsdict to the function object, so that we can change it later
        # on if we wish
        wrapper._defaultsdict = defaultsdict

        _default_container._add_wrapper_function(wrapper, qualname=qualname)

        return wrapper

    return the_decorator


def _import_all(package_name='pymor'):

    package = __import__(package_name)

    if hasattr(package, '__path__'):
        def onerror(name):
            from pymor.core.logger import getLogger
            logger = getLogger('pymor.core.defaults._import_all')
            logger.warn('Failed to import ' + name)

        for p in pkgutil.walk_packages(package.__path__, package_name + '.', onerror=onerror):
            try:
                __import__(p[1])
            except ImportError:
                from pymor.core.logger import getLogger
                logger = getLogger('pymor.core.defaults._import_all')
                logger.warn('Failed to import ' + p[1])


def print_defaults(import_all=True, shorten_paths=2):
    """Print all default values set in pyMOR.

    Parameters
    ----------
    import_all
        While `print_defaults` will always print all defaults defined in
        loaded configuration files or set via :func:`set_defaults`, default
        values set in the function signature can only be printed, if the
        modules containing these functions have been imported. If `import_all`
        is set to `True`, `print_defaults` will therefore first import all
        of pyMOR's modules, to provide a complete lists of defaults.
    shorten_paths
        Shorten the paths of all default values by `shorten_paths` components.
        The last two path components will always be printed.
    """

    if import_all:
        _default_container.import_all()

    keys = []
    values = []
    comments = []
    for k in sorted(_default_container.keys()):
        try:
            v, c = _default_container.get(k, code=True, file=True, user=True)
        except ValueError:
            continue
        ks = k.split('.')
        if len(ks) >= shorten_paths + 2:
            keys.append('.'.join(ks[shorten_paths:]))
        else:
            keys.append('.'.join(ks))
        values.append(repr(v))
        comments.append(c)
    key_width = max(map(len, keys))
    value_width = max(map(len, values))

    key_string = 'path (shortened)' if shorten_paths else 'path'
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


def write_defaults_to_file(filename='./pymor_defaults.py', packages=('pymor',), code=True, file=True, user=True):
    """Write the currently set default values in pyMOR to a configuration file.

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
    code
        If `False`, ignore default values, defined in function signatures.
    file
        If `False`, ignore default values loaded from a configuration file.
    user
        If `False`, ignore default values provided via :func:`set_defaults`.
    """

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

    with open(filename, 'w') as f:
        print('''
# pyMOR defaults config file
# This file has been automatically created by pymor.core.defaults.write_defaults_to_file'.

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


def load_defaults_from_file(filename='./pymor_defaults.py'):
    """Loads default values define in a configuration file.

    Such configuration files can be created via :func:`write_defaults_to_file`.
    The file is loaded via Python's `exec` function, so be very careful
    with configuration files you have not created your own. You have been
    warned!

    (Note that defaults should only be changed/loaded before state ids have been
    calculated. See this :ref:`warning <defaults_warning>` for details.)

    Parameters
    ----------
    filename
        Path of the configuration file.
    """
    env = {}
    exec open(filename).read() in env
    _default_container.update(env['d'], type='file')


def set_defaults(defaults, check=True):
    """Set default values.

    This method sets the default value of function arguments marked via the
    :func:`defaults` decorator, overriding default values specified in the
    function signature or set earlier via :func:`load_defaults_from_file` or
    previous `set_defaults` calls.

    (Note that defaults should only be changed/loaded before state ids have been
    calculated. See this :ref:`warning <defaults_warning>` for details.)

    Parameters
    ----------
    defaults
        Dictionary of default values. Keys are the full paths of the default
        values. (See :func:`defaults`.)
    check
        If `True`, recursively import all pacakges associated to the paths
        of the set default values. Then check if defaults with the provided
        paths have actually been defined using the :func:`defaults` decorator.
    """
    _default_container.update(defaults, type='user')
    if check:
        _default_container.check_consistency(delete=True)


def defaults_sid():
    """Return a state id for pyMOR's global defaults.

    This method is used for the calculation of state ids of immutable objects
    (see :mod:`pymor.core.interfaces`.) and for :mod:`~pymor.core.cache` key
    generation.

    For other uses see the implementation of
    :meth:`pymor.operators.numpy.NumpyMatrixBasedOperator.assemble`.
    """
    return _default_container_sid
