# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
#cannot use unicode_literals here, or else dbm backend fails

from functools import partial
import os


from pymor import defaults
from pymor.core import dumps, ImmutableInterface
import pymor.core.dogpile_backends


class CacheRegion(object):

    def get(self, key):
        raise NotImplementedError

    def set(self, key, value):
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
        from dogpile import cache as dc
        self._cache_region = dc.make_region()
        self._cache_region.configure_from_config(pymor.core.dogpile_backends.DEFAULT_MEMORY_CONFIG, '')


class DogpileDiskCacheRegion(DogpileCacheRegion):

    def __init__(self):
        from dogpile import cache as dc
        self._cache_region = dc.make_region()
        self._cache_region.configure_from_config(pymor.core.dogpile_backends.DEFAULT_DISK_CONFIG, '')


cache_regions = {'memory': DogpileMemoryCacheRegion(),
                 'disk': DogpileDiskCacheRegion()}


class cached(object):

    def __init__(self, function):
        self.decorated_function = function

    def __call__(self, im_self, *args, **kwargs):
        '''Via the magic that is partial functions returned from __get__, im_self is the instance object of the class
        we're decorating a method of and [kw]args are the actual parameters to the decorated method'''
        region = cache_regions[im_self._cache_region]
        key = (self.decorated_function.__name__, getattr(im_self, 'sid', im_self.uid),
                tuple(x.sid if hasattr(x, 'sid') else x for x in args),
                tuple((k, v.sid if hasattr(v, 'sid') else v) for k, v in kwargs.iteritems()),
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
        if instance._cache_region is None:
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
        self._cache_disabled = int(os.environ.get('PYMOR_CACHE_DISABLE', 0)) == 1
        if self._cache_disabled:
            self.logger.warn('caching globally disabled')
            self._cache_region = None
        elif region in (None, 'none'):
            self._cache_region = None
        else:
            assert region in cache_regions
            self._cache_region = region
