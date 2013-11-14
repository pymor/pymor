# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
#cannot use unicode_literals here, or else dbm backend fails

from functools import partial
import os

import numpy as np

from pymor import defaults
from pymor.core import dumps, ImmutableInterface
import pymor.core.dogpile_backends


class CacheRegion(object):

    enabled = True

    def get(self, key):
        raise NotImplementedError

    def set(self, key, value):
        raise NotImplementedError

    def clear(self):
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

    def __init__(self, filename=None):
        self.filename = filename
        self._new_region()

    def _new_region(self):
        from dogpile import cache as dc
        self._cache_region = dc.make_region()
        config = dict(pymor.core.dogpile_backends.DEFAULT_DISK_CONFIG)
        if self.filename:
            config['arguments.filename'] = os.path.expanduser(self.filename)
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
    global _caching_disabled
    _caching_disabled = int(os.environ.get('PYMOR_CACHE_DISABLE', 0)) == 1


def disable_caching():
    global _caching_disabled
    _caching_disabled = True


def clear_caches():
    for r in cache_regions.itervalues():
        r.clear()


class cached(object):

    def __init__(self, function):
        self.decorated_function = function

    def __call__(self, im_self, *args, **kwargs):
        '''Via the magic that is partial functions returned from __get__, im_self is the instance object of the class
        we're decorating a method of and [kw]args are the actual parameters to the decorated method'''
        region = cache_regions[im_self._cache_region]
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
        if _caching_disabled or instance._cache_region is None:
            return partial(self.decorated_function, instance)
        else:
            return partial(self.__call__, instance)


class CacheableInterface(ImmutableInterface):
    '''Base class for anything that wants to use our built-in caching.
    '''

    def __init__(self, region='memory'):
        self.enable_caching(region)

    def disable_caching(self):
        self._cache_region = None

    def enable_caching(self, region):
        if region in (None, 'none'):
            self._cache_region = None
        else:
            assert region in cache_regions
            self._cache_region = region
