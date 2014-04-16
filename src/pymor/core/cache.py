# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''This module provides the caching facilities of pyMOR.

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

Note, however, that instances of :class:`~pymor.core.interfaces.ImmutableInterface`
are allowed to have mutable private attributes. It is the implementors
responsibility not to break things.

Backends for storage of cached return values derive from :class:`CacheRegion`.
Currently two backends are provided for memory-based and disk-based caching
(:class:`DogpileMemoryCacheRegion` and :class:`DogpileDiskCacheRegion`). The
available regions are stored in the module level `cache_regions` dict. The
user can add additional regions (e.g. multiple disk cache regions) as
required. :class:`CacheableInterface` takes a `region` argument
through which a key of the `cache_regions` dict can provided to select
a cache region which should be used by the instance. (Setting `region` to
`None` or `'none'` disables caching.)

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
'''


from __future__ import absolute_import, division, print_function
#cannot use unicode_literals here, or else dbm backend fails

from functools import partial
import os
from types import MethodType

import numpy as np

from pymor import defaults
from pymor.core import dumps, ImmutableInterface
import pymor.core.dogpile_backends


class CacheRegion(object):
    '''Base class for all pyMOR cache regions.

    Attributes
    ----------
    enabled
        If `False` caching is disabled for this region.
    '''

    enabled = True

    def get(self, key):
        raise NotImplementedError

    def set(self, key, value):
        raise NotImplementedError

    def clear(self):
        '''Clear the entire cache region.'''
        raise NotImplementedError


class DogpileCacheRegion(CacheRegion):

    def get(self, key):
        value = self._cache_region.get(key)
        if value is pymor.core.dogpile_backends.NO_VALUE:
            return False, None
        else:
            return True, value

    def set(self, key, value):
        self._cache_region.set(key, value)


class DogpileMemoryCacheRegion(DogpileCacheRegion):

    def __init__(self):
        self._new_region()

    def _new_region(self):
        from dogpile import cache as dc
        self._cache_region = dc.make_region()
        self._cache_region.configure_from_config(pymor.core.dogpile_backends.DEFAULT_MEMORY_CONFIG, '')

    def clear(self):
        self._new_region()

    def set(self, key, value):
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
        self._cache_region.set(key, value)


class DogpileDiskCacheRegion(DogpileCacheRegion):

    def __init__(self, filename=None, max_size=1024 ** 3):
        self.filename = filename
        self.max_size = max_size
        self._new_region()

    def _new_region(self):
        from dogpile import cache as dc
        self._cache_region = dc.make_region()
        config = dict(pymor.core.dogpile_backends.DEFAULT_DISK_CONFIG)
        if self.filename:
            config['arguments.filename'] = os.path.expanduser(self.filename)
        if self.max_size:
            config['arguments.max_size'] = self.max_size
        self._cache_region.configure_from_config(config, '')

    def clear(self):
        import glob
        filename = self._cache_region.backend.filename
        del self._cache_region
        files = glob.glob(filename + '*')
        map(os.unlink, files)
        self._new_region()


cache_regions = {'memory': DogpileMemoryCacheRegion(),
                 'disk': DogpileDiskCacheRegion()}
_caching_disabled = int(os.environ.get('PYMOR_CACHE_DISABLE', 0)) == 1
if _caching_disabled:
    from pymor.core import getLogger
    getLogger('pymor.core.cache').warn('caching globally disabled by environment')


def enable_caching():
    '''Globally enable caching.'''
    global _caching_disabled
    _caching_disabled = int(os.environ.get('PYMOR_CACHE_DISABLE', 0)) == 1


def disable_caching():
    '''Globally disable caching.'''
    global _caching_disabled
    _caching_disabled = True


def clear_caches():
    '''Clear all cache regions.'''
    for r in cache_regions.itervalues():
        r.clear()


class cached(object):
    '''Decorator to make a method of `CacheableInterface` actually cached.'''

    def __init__(self, function):
        self.decorated_function = function

    def __call__(self, im_self, *args, **kwargs):
        '''Via the magic that is partial functions returned from __get__, im_self is the instance object of the class
        we're decorating a method of and [kw]args are the actual parameters to the decorated method'''
        region = cache_regions[im_self.cache_region]
        if not region.enabled:
            return self.decorated_function(im_self, *args, **kwargs)

        key = (self.decorated_function.__name__, getattr(im_self, 'sid', im_self.uid),
               tuple(getattr(x, 'sid', x) for x in args),
               tuple((k, getattr(v, 'sid', v)) for k, v in sorted(kwargs.iteritems())),
               defaults.sid)
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
        '''Implement the descriptor protocol to make decorating instance method possible.
        Return a partial function where the first argument is the instance of the decorated instance object.
        '''
        if instance is None:
            return MethodType(self.decorated_function, None, instancetype)
        elif _caching_disabled or instance.cache_region is None:
            return partial(self.decorated_function, instance)
        else:
            return partial(self.__call__, instance)


class CacheableInterface(ImmutableInterface):
    '''Base class for anything that wants to use our built-in caching.

    Attributes
    ----------
    cache_region
        Name of the `CacheRegion` to use. Must correspond to a key in
        :attr:`pymor.core.cache.cache_regions`. If `None` or `'none'`, caching
        is disabled.
    '''

    @property
    def cache_region(self):
        try:
            return self.__cache_region
        except AttributeError:
            self.__cache_region = 'memory' if 'memory' in cache_regions else None

    @cache_region.setter
    def cache_region(self, region):
        if region in (None, 'none'):
            self.__cache_region = None
        elif region not in cache_regions:
            raise ValueError('Unkown cache region.')
        else:
            self.__cache_region = region

    def disable_caching(self):
        '''Disable caching for this instance.'''
        self.__cache_region = None

    def enable_caching(self, region):
        '''Enable caching for this instance.

        Parameters
        ----------
        region
            Name of the `CacheRegion` to use. Must correspond to a key in
            `pymor.core.cache.cache_regions`. If `None` or `'none'`, caching
            is disabled.
        '''
        self.cache_region = region
