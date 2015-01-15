# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module provides the caching facilities of pyMOR.

Any class that wishes to provide cached method calls should derive from
:class:`CacheableInterface`. Methods which are to be cached can then
be marked using the :class:`cached` decorator.

To ensure consistency, :class:`CacheableInterface` derives from
:class:`~pymor.core.interfaces.ImmutableInterface`: The return value of a
cached method should only depend on its arguments as well as
the immutable state of the class instance.

Making this assumption, the keys for cache lookup are created from
the following data:

    1. the instance's state id (see :class:`~pymor.core.interfaces.ImmutableInterface`)
       if available, else the instance's unique id
       (see :class:`~pymor.core.interfaces.BasicInterface`),
    2. the method's `__name__`,
    3. the state id of each argument if available, else its pickled
       state.
    4. the state of pyMOR's global :mod:`~pymor.core.defaults`.

Note, however, that instances of :class:`~pymor.core.interfaces.ImmutableInterface`
are allowed to have mutable private attributes. It is the implementors
responsibility not to break things.
(See this :ref:`warning <ImmutableInterfaceWarning>`.)

Backends for storage of cached return values derive from :class:`CacheRegion`.
Currently two backends are provided for memory-based and disk-based caching
(:class:`MemoryRegion` and :class:`SQLiteRegion`). The available regions
are stored in the module level `cache_regions` dict. The user can add
additional regions (e.g. multiple disk cache regions) as required.
:class:`CacheableInterface` takes a `region` argument through which a key of
the `cache_regions` dict can provided to select a cache region which should
be used by the instance. (Setting `region` to `None` or `'none'` disables caching.)

By default, a 'memory' and a 'disk' cache region are automatically configured. The
path and maximum size of the disk region as well as the maximum number of keys of
the memory cache region can be configured via the
`pymor.core.cache._setup_default_regions.disk_path`,
`pymor.core.cache._setup_default_regions.disk_max_size` and
`pymor.core.cache._setup_default_regions.memory_max_keys` |defaults|.
(Note that changing these defaults will result in changed |state ids|, so moving
a disk cache and changing the default path accordingly will result in cache
misses.) As an alternative, these defaults can be overridden by the
`PYMOR_CACHE_PATH`, `PYMOR_CACHE_MAX_SIZE` and `PYMOR_CACHE_MEMORY_MAX_KEYS`
environment variables. (These variables do not enter |state id| calculation
and are therefore the preferred way to configure caching.)


There are multiple ways to disable and enable caching in pyMOR:

    1. Calling :func:`disable_caching` (:func:`enable_caching`).
    2. Setting `cache_regions[region].enabled` to `False` or `True`.
    3. Calling :meth:`CacheableInterface.disable_caching`
       (:meth:`CacheableInterface.enable_caching`).

Caching of a method is only active, if caching is enabled on global,
region and instance level. For debugging purposes, it is moreover possible
to set the environment variable `PYMOR_CACHE_DISABLE=1` which overrides
any call to :func:`enable_caching`.

A cache region can be emptied using :meth:`CacheRegion.clear`. The function
:func:`clear_caches` clears each cache region registered in `cache_regions`.
"""


from __future__ import absolute_import, division, print_function
# cannot use unicode_literals here, or else dbm backend fails

import base64
from collections import OrderedDict
import datetime
from functools import partial
import getpass
import os
import sqlite3
import tempfile
from types import MethodType

from pymor.core.defaults import defaults, defaults_sid
from pymor.core.interfaces import ImmutableInterface
from pymor.core.pickle import dump, dumps, load


class CacheRegion(object):
    """Base class for all pyMOR cache regions.

    Attributes
    ----------
    enabled
        If `False` caching is disabled for this region.
    """

    enabled = True

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

    enabled = True

    def __init__(self, path, max_size):
        self.path = path
        self.max_size = max_size
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
            self.housekeeping()

    def get(self, key):
        c = self.conn.cursor()
        t = (base64.b64encode(key),)
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
        key = base64.b64encode(key)
        now = datetime.datetime.now()
        filename = now.isoformat() + '.dat'
        file_path = os.path.join(self.path, filename)
        while os.path.exists(file_path):
            now = now + datetime.timedelta(microseconds=1)
            filename = now().isoformat()
            file_path = os.path.join(self.path, filename)
        fd = os.open(file_path, os.O_WRONLY | os.O_EXCL | os.O_CREAT)
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


@defaults('disk_path', 'disk_max_size', 'memory_max_keys')
def _setup_default_regions(disk_path=os.path.join(tempfile.gettempdir(), 'pymor.cache.' + getpass.getuser()),
                           disk_max_size=1024 ** 3,
                           memory_max_keys=1000):
    cache_regions['disk'] = SQLiteRegion(path=disk_path, max_size=disk_max_size)
    cache_regions['memory'] = MemoryRegion(memory_max_keys)

cache_regions = {}
_setup_default_regions(disk_path=os.environ.get('PYMOR_CACHE_PATH', None),
                       disk_max_size=((lambda size:
                                       None if not size else
                                       int(size[:-1]) * 1024 if size[-1] == 'K' else
                                       int(size[:-1]) * 1024 ** 2 if size[-1] == 'M' else
                                       int(size[:-1]) * 1024 ** 3 if size[-1] == 'G' else
                                       int(size))
                                      (os.environ.get('PYMOR_CACHE_MAX_SIZE', '').strip().upper())),
                       memory_max_keys=((lambda num: int(num) if num else None)
                                        (os.environ.get('PYMOR_CACHE_MEMORY_MAX_KEYS', None))))

_caching_disabled = int(os.environ.get('PYMOR_CACHE_DISABLE', 0)) == 1
if _caching_disabled:
    from pymor.core.getLogger import getLogger
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

    def __call__(self, im_self, *args, **kwargs):
        """Via the magic that is partial functions returned from __get__, im_self is the instance object of the class
        we're decorating a method of and [kw]args are the actual parameters to the decorated method"""
        region = cache_regions[im_self.cache_region]
        if not region.enabled:
            return self.decorated_function(im_self, *args, **kwargs)

        key = (self.decorated_function.__name__, getattr(im_self, 'sid', im_self.uid),
               tuple(getattr(x, 'sid', x) for x in args),
               tuple((k, getattr(v, 'sid', v)) for k, v in sorted(kwargs.iteritems())),
               defaults_sid())
        key = dumps(key)
        found, value = region.get(key)
        if found:
            return value
        else:
            im_self.logger.debug('creating new cache entry for {}.{}'
                                 .format(im_self.__class__.__name__, self.decorated_function.__name__))
            value = self.decorated_function(im_self, *args, **kwargs)
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

    @property
    def cache_region(self):
        try:
            return self.__cache_region
        except AttributeError:
            self.__cache_region = 'memory' if 'memory' in cache_regions else None
            return self.__cache_region

    @cache_region.setter
    def cache_region(self, region):
        if region in (None, 'none'):
            self.__cache_region = None
        elif region not in cache_regions:
            raise ValueError('Unkown cache region.')
        else:
            self.__cache_region = region

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
        self.cache_region = region
