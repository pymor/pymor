# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
#cannot use unicode_literals here, or else dbm backend fails
from functools import partial
import os

from pymor import defaults
import pymor.core
from pymor.core.interfaces import BasicInterface
from pymor.tools import memory




class cached(BasicInterface):

    def __init__(self, function):
        super(cached, self).__init__()
        self.decorated_function = function
        self._cache_disabled = int(os.environ.get('PYMOR_CACHE_DISABLE', 0)) == 1
        if self._cache_disabled:
            self.logger.warn('caching globally disabled')

    def __call__(self, im_self, *args, **kwargs):
        '''Via the magic that is partial functions returned from __get__, im_self is the instance object of the class
        we're decorating a method of and [kw]args are the actual parameters to the decorated method'''
        cache = im_self._cache_region
        keygen = im_self.keygen_generator(im_self._namespace, self.decorated_function)
        key = keygen(*args, **kwargs)

        def creator_function():
            self.logger.debug('creating new cache entry for {}.{}'
                              .format(im_self.__class__.__name__, self.decorated_function.__name__))
            return self.decorated_function(im_self, *args, **kwargs)
        return cache.get_or_create(key, creator_function, im_self._expiration_time)

    def __get__(self, instance, instancetype):
        '''Implement the descriptor protocol to make decorating instance method possible.
        Return a partial function where the first argument is the instance of the decorated instance object.
        '''
        if self._cache_disabled or instance._cache_disabled_for_instance:
            return partial(self.decorated_function, instance)
        return partial(self.__call__, instance)


class Cachable(object):
    '''Base class for anything that wants to use our built-in caching.
    provides custom __{g,s}etstate__ functions to allow using derived
    classes with the pickle module
    '''

    def __init__(self, config=DEFAULT_MEMORY_CONFIG, disable=False):
        if disable:
            self._cache_disabled_for_instance = True
        else:
            self._cache_disabled_for_instance = False
            self._cache_config = config
            self._init_cache()
            self._namespace = self.__class__.__name__
            self._expiration_time = None

    def _init_cache(self):
        self._cache_region = dc.make_region(function_key_generator=self.keygen_generator)
        self._cache_region.configure_from_config(self._cache_config, '')

    def keygen_generator(self, namespace, function):
        '''I am the default generator function for (potentially) function specific keygens.
        I construct a key from the function name and given namespace
        plus string representations of all positional and keyword args.
        '''
        fname = function.__name__
        namespace = str(namespace)

        def keygen(*arg, **kwargs):
            return (namespace + "_" + fname + '_' + str(getattr(self, 'sid', id(self))) + '_' +
                    "_".join(s.sid if hasattr(s, 'sid') else str(s) for s in arg)
                    + '__'.join(x.sid if hasattr(x, 'sid') else str(x) for x in kwargs.iteritems())
                    + '_' + defaults.sid)
        return keygen

    def __getstate__(self):
        '''cache regions contain lock objects that cannot be pickled.
        Therefore we don't include them in the state that the pickle protocol gets to see.
        '''
        return {name: getattr(self, name) for name in self.__dict__.keys() if name != '_cache_region'}

    def __setstate__(self, d):
        '''Since we cannot pickle the cache region, we have to re-init
        the region from the pickled config when the pickle module
        calls this function.
        '''
        self.__dict__.update(d)
        self._init_cache()
