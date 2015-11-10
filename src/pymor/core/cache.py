# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Andreas Buhr <andreas@andreasbuhr.de>

"""This module provides the caching facilities of pyMOR.

Any class that wishes to provide cached method calls should derive from
:class:`CacheableInterface`. Methods which are to be cached can then
be marked using the :class:`cached` decorator.

To ensure consistency, :class:`CacheableInterface` derives from
:class:`~pymor.core.interfaces.ImmutableInterface`: The return value of a
cached method call should only depend on its arguments as well as
the immutable state of the class instance.

Making this assumption, the keys for cache lookup are created from
the following data:

    1. the instance's |state id| if available, else the instance's
       unique id (see :class:`~pymor.core.interfaces.BasicInterface`),
    2. the method's `__name__`,
    3. the state id of the arguments,
    4. the state id of pyMOR's global :mod:`~pymor.core.defaults`.

Note, however, that instances of :class:`~pymor.core.interfaces.ImmutableInterface`
are allowed to have mutable private attributes. It is the implementors
responsibility not to break things.
(See this :ref:`warning <ImmutableInterfaceWarning>`.)

Backends for storage of cached return values derive from :class:`CacheRegion`.
Currently two backends are provided for memory-based and disk-based caching
(:class:`MemoryRegion` and :class:`SQLiteRegion`). The available regions
are stored in the module level `cache_regions` dict. The user can add
additional regions (e.g. multiple disk cache regions) as required.
:class:`CacheableInterface` has a `region` attribute through which a key of
the `cache_regions` dict can provided to select a cache region which should
be used by the instance. (Setting `region` to `None` or `'none'` disables caching.)

By default, a 'memory' and a 'disk' cache region are automatically configured. The
path and maximum size of the disk region as well as the maximum number of keys of
the memory cache region can be configured via the
`pymor.core.cache.default_regions.disk_path`,
`pymor.core.cache.default_regions.disk_max_size` and
`pymor.core.cache.default_regions.memory_max_keys` |defaults|.

There two ways to disable and enable caching in pyMOR:

    1. Calling :func:`disable_caching` (:func:`enable_caching`), to disable
       (enable) caching globally.
    2. Calling :meth:`CacheableInterface.disable_caching`
       (:meth:`CacheableInterface.enable_caching`) to disable (enable) caching
       for a given instance.

Caching of a method is only active if caching has been enabled both globally
(enabled by default) and on instance level. For debugging purposes, it is moreover
possible to set the environment variable `PYMOR_CACHE_DISABLE=1` which overrides
any call to :func:`enable_caching`.

A cache region can be emptied using :meth:`CacheRegion.clear`. The function
:func:`clear_caches` clears each cache region registered in `cache_regions`.
"""


from __future__ import absolute_import, division, print_function
# cannot use unicode_literals here, or else dbm backend fails

import atexit
from collections import OrderedDict
import datetime
from functools import partial
import getpass
import inspect
import os
import sqlite3
import tempfile
from types import MethodType

from pymor.core.defaults import defaults, defaults_sid
from pymor.core.interfaces import ImmutableInterface, generate_sid
from pymor.core.pickle import dump, load


@atexit.register
def cleanup_non_persisten_regions():
    for region in cache_regions.values():
        if not region.persistent:
            region.clear()


class CacheRegion(object):
    """Base class for all pyMOR cache regions.

    Attributes
    ----------
    persistent
        If `True`, cache entries are kept between multiple
        program runs.
    """

    persistent = False

    def get(self, key):
        raise NotImplementedError

    def set(self, key, value):
        raise NotImplementedError

    def clear(self):
        """Clear the entire cache region."""
        raise NotImplementedError


class MemoryRegion(CacheRegion):

    NO_VALUE = {}

    def __init__(self, max_keys):
        self.max_keys = max_keys
        self._cache = OrderedDict()

    def get(self, key):
        value = self._cache.get(key, self.NO_VALUE)
        if value is self.NO_VALUE:
            return False, None
        else:
            return True, value

    def set(self, key, value):
        if len(self._cache) == self.max_keys:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def clear(self):
        self._cache = OrderedDict()


class SQLiteRegion(CacheRegion):

    def __init__(self, path, max_size, persistent):
        self.path = path
        self.max_size = max_size
        self.persistent = persistent
        self.bytes_written = 0
        if not os.path.exists(path):
            os.mkdir(path)
            self.conn = conn = sqlite3.connect(os.path.join(path, 'pymor_cache.db'))
            c = conn.cursor()
            c.execute('''CREATE TABLE entries
                         (id INTEGER PRIMARY KEY, key TEXT UNIQUE, filename TEXT, size INT)''')
            conn.commit()
        else:
            self.conn = sqlite3.connect(os.path.join(path, 'pymor_cache.db'))
            if persistent:
                self.housekeeping()
            else:
                self.clear()

    def get(self, key):
        c = self.conn.cursor()
        t = (key,)
        c.execute('SELECT filename FROM entries WHERE key=?', t)
        result = c.fetchall()
        if len(result) == 0:
            return False, None
        elif len(result) == 1:
            file_path = os.path.join(self.path, result[0][0])
            with open(file_path) as f:
                value = load(f)
            return True, value
        else:
            raise RuntimeError('Cache is corrupt!')

    def set(self, key, value):
        fd, file_path = tempfile.mkstemp('.dat', datetime.datetime.now().isoformat()[:-7] + '-', self.path)
        filename = os.path.basename(file_path)
        try:
            f = os.fdopen(fd, 'w')
            dump(value, f)
            file_size = f.tell()
        finally:
            f.close()
        conn = self.conn
        c = conn.cursor()
        try:
            c.execute("INSERT INTO entries(key, filename, size) VALUES ('{}', '{}', {})"
                      .format(key, filename, file_size))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.commit()
            from pymor.core.logger import getLogger
            getLogger('pymor.core.cache.SQLiteRegion').warn('Key already present in cache region, ignoring.')
            os.unlink(file_path)
        self.bytes_written += file_size
        if self.bytes_written >= 0.1 * self.max_size:
            self.housekeeping()

    def clear(self):
        # Try to safely delete all cache entries, even if another process
        # accesses the same region.
        self.bytes_written = 0
        conn = self.conn
        c = conn.cursor()
        c.execute('SELECT id, filename FROM entries ORDER BY id ASC')
        entries = c.fetchall()
        if entries:
            ids_to_delete, files_to_delete = zip(*entries)
            c.execute('DELETE FROM entries WHERE id in ({})'.format(','.join(map(str, ids_to_delete))))
            conn.commit()
            path = self.path
            for filename in files_to_delete:
                try:
                    os.unlink(os.path.join(path, filename))
                except OSError:
                    from pymor.core.logger import getLogger
                    getLogger('pymor.core.cache.SQLiteRegion').warn('Cannot delete cache entry ' + filename)

    def housekeeping(self):
        self.bytes_written = 0
        conn = self.conn
        c = conn.cursor()
        c.execute('SELECT SUM(size) FROM entries')
        size = c.fetchone()
        size = size[0] if size is not None else 0
        if size > self.max_size:
            bytes_to_delete = size - self.max_size + 0.75 * self.max_size
            deleted = 0
            ids_to_delete = []
            files_to_delete = []
            c.execute('SELECT id, filename, size FROM entries ORDER BY id ASC')
            while deleted < bytes_to_delete:
                id_, filename, file_size = c.fetchone()
                ids_to_delete.append(id_)
                files_to_delete.append(filename)
                deleted += file_size
            c.execute('DELETE FROM entries WHERE id in ({})'.format(','.join(map(str, ids_to_delete))))
            conn.commit()
            path = self.path
            for filename in files_to_delete:
                try:
                    os.unlink(os.path.join(path, filename))
                except OSError:
                    from pymor.core.logger import getLogger
                    getLogger('pymor.core.cache.SQLiteRegion').warn('Cannot delete cache entry ' + filename)

            from pymor.core.logger import getLogger
            getLogger('pymor.core.cache.SQLiteRegion').info('Removed {} old cache entries'.format(len(ids_to_delete)))


@defaults('disk_path', 'disk_max_size', 'persistent_path', 'persistent_max_size', 'memory_max_keys',
          sid_ignore=('disk_path', 'disk_max_size', 'persistent_path', 'persistent_max_size', 'memory_max_keys'))
def default_regions(disk_path=os.path.join(tempfile.gettempdir(), 'pymor.cache.' + getpass.getuser()),
                    disk_max_size=1024 ** 3,
                    persistent_path=os.path.join(tempfile.gettempdir(), 'pymor.persistent.cache.' + getpass.getuser()),
                    persistent_max_size=1024 ** 3,
                    memory_max_keys=1000):

    parse_size_string = lambda size: \
        int(size[:-1]) * 1024 if size[-1] == 'K' else \
        int(size[:-1]) * 1024 ** 2 if size[-1] == 'M' else \
        int(size[:-1]) * 1024 ** 3 if size[-1] == 'G' else \
        int(size)

    if isinstance(disk_max_size, str):
        disk_max_size = parse_size_string(disk_max_size)

    cache_regions['disk'] = SQLiteRegion(path=disk_path, max_size=disk_max_size, persistent=False)
    cache_regions['persistent'] = SQLiteRegion(path=persistent_path, max_size=persistent_max_size, persistent=True)
    cache_regions['memory'] = MemoryRegion(memory_max_keys)

cache_regions = {}

_caching_disabled = int(os.environ.get('PYMOR_CACHE_DISABLE', 0)) == 1
if _caching_disabled:
    from pymor.core.logger import getLogger
    getLogger('pymor.core.cache').warn('caching globally disabled by environment')


def enable_caching():
    """Globally enable caching."""
    global _caching_disabled
    _caching_disabled = int(os.environ.get('PYMOR_CACHE_DISABLE', 0)) == 1


def disable_caching():
    """Globally disable caching."""
    global _caching_disabled
    _caching_disabled = True


def clear_caches():
    """Clear all cache regions."""
    for r in cache_regions.itervalues():
        r.clear()


class cached(object):
    """Decorator to make a method of `CacheableInterface` actually cached."""

    def __init__(self, function):
        self.decorated_function = function
        argspec = inspect.getargspec(function)
        self.argnames = argnames = argspec.args[1:]  # first argument is self
        defaults = function.__defaults__
        if defaults:
            self.defaults = {k: v for k, v in zip(argnames[-len(defaults):], defaults)}
        else:
            self.defaults = None

    def __call__(self, im_self, *args, **kwargs):
        """Via the magic that is partial functions returned from __get__, im_self is the instance object of the class
        we're decorating a method of and [kw]args are the actual parameters to the decorated method"""
        if not cache_regions:
            default_regions()
        try:
            region = cache_regions[im_self.cache_region]
        except KeyError:
            raise KeyError('No cache region "{}" found'.format(im_self.cache_region))

        # compute id for self
        if region.persistent:
            self_id = getattr(im_self, 'sid')
            if not self_id:     # this can happen when cache_region is already set by the class to
                                # a persistent region
                self_id = im_self.generate_sid()
        else:
            self_id = im_self.uid

        # ensure that passing a value as positional or keyword argument does not matter
        kwargs.update(zip(self.argnames, args))

        # ensure the values of optional parameters enter the cache key
        defaults = self.defaults
        if defaults:
            kwargs = dict(defaults, **kwargs)

        key = generate_sid((self.decorated_function.__name__, self_id, kwargs, defaults_sid()))
        found, value = region.get(key)
        if found:
            return value
        else:
            im_self.logger.debug('creating new cache entry for {}.{}'
                                 .format(im_self.__class__.__name__, self.decorated_function.__name__))
            value = self.decorated_function(im_self, **kwargs)
            region.set(key, value)
            return value

    def __get__(self, instance, instancetype):
        """Implement the descriptor protocol to make decorating instance method possible.
        Return a partial function where the first argument is the instance of the decorated instance object.
        """
        if instance is None:
            return MethodType(self.decorated_function, None, instancetype)
        elif _caching_disabled or instance.cache_region is None:
            return partial(self.decorated_function, instance)
        else:
            return partial(self.__call__, instance)


class CacheableInterface(ImmutableInterface):
    """Base class for anything that wants to use our built-in caching.

    Attributes
    ----------
    cache_region
        Name of the `CacheRegion` to use. Must correspond to a key in
        :attr:`pymor.core.cache.cache_regions`. If `None` or `'none'`, caching
        is disabled.
    """

    sid_ignore = ImmutableInterface.sid_ignore | {'cache_region'}

    cache_region = None

    def disable_caching(self):
        """Disable caching for this instance."""
        self.__cache_region = None

    def enable_caching(self, region):
        """Enable caching for this instance.

        Parameters
        ----------
        region
            Name of the `CacheRegion` to use. Must correspond to a key in
            `pymor.core.cache.cache_regions`. If `None` or `'none'`, caching
            is disabled.
        """
        if region in (None, 'none'):
            self.__dict__['cache_region'] = None
        else:
            self.__dict__['cache_region'] = region
            r = cache_regions.get(region, None)
            if r and r.persistent:
                self.generate_sid()
