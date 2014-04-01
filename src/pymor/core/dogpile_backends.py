# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''
This module contains backend implementations of pyMOR cache regions using
the `dogpile <https://pypi.python.org/pypi/dogpile.cache>`_ package.

Not to be used directly.
'''

from collections import OrderedDict
from collections import deque
import getpass
from os.path import join
from pprint import pformat
from tempfile import gettempdir
import os
import sys

from dogpile import cache as dc
from dogpile.cache.backends.file import DBMBackend

import pymor.core
from pymor.core import BasicInterface
from pymor.tools import memory


# patch dogpile.cache.compat module to use
# latest pickle protocol
import dogpile.cache.compat
import types
patched_pickle = types.ModuleType('pickle')
patched_pickle.dumps = pymor.core.dumps
patched_pickle.loads = pymor.core.loads
dogpile.cache.compat.pickle = patched_pickle

NO_CACHE_CONFIG = {"backend": 'Dummy'}
DEFAULT_MEMORY_CONFIG = {"backend": 'LimitedMemory', 'arguments.max_kbytes': 20000}
SMALL_MEMORY_CONFIG = {"backend": 'LimitedMemory', 'arguments.max_keys': 20,
                       'arguments.max_kbytes': 20}
DEFAULT_DISK_CONFIG = {"backend": 'LimitedFile',
                       "arguments.filename": join(gettempdir(), 'pymor.cache.{}.dbm'.format(getpass.getuser())),
                       'arguments.max_keys': 2000,
                       'arguments.max_size': 1024 ** 3}
SMALL_DISK_CONFIG = {"backend": 'LimitedFile',
                     "arguments.filename": join(gettempdir(), 'pymor.small_cache.{}.dbm'.format(getpass.getuser())),
                     'arguments.max_keys': 20}

NO_VALUE = dc.api.NO_VALUE


class DummyBackend(BasicInterface, dc.api.CacheBackend):

    def __init__(self, argument_dict):
        self.logger.debug('DummyBackend args {}'.format(pformat(argument_dict)))

    def get(self, key):
        return dc.api.NO_VALUE

    def set(self, key, value):
        pass

    def delete(self, key):
        pass


class LimitedMemoryBackend(BasicInterface, dc.api.CacheBackend):

    def __init__(self, argument_dict):
        '''If argument_dict contains a value for max_kbytes this the total memory limit in kByte that is enforced on the
        internal cache dictionary, otherwise it's set to sys.maxint.
        If argument_dict contains a value for max_keys this maximum amount of cache values kept in the
        internal cache dictionary, otherwise it's set to sys.maxlen.
        If necessary values are deleted from the cache in FIFO order.
        '''
        self.logger.debug('LimitedMemoryBackend args {}'.format(pformat(argument_dict)))
        self._max_keys = argument_dict.get('max_keys', sys.maxsize)
        self._max_bytes = argument_dict.get('max_kbytes', sys.maxint / 1024) * 1024
        self._cache = OrderedDict()

    def get(self, key):
        return self._cache.get(key, dc.api.NO_VALUE)

    def print_limit(self, additional_size=0):
        self.logger.info('LimitedMemoryBackend at {}({}) keys -- {}({}) Byte'
                         .format(len(self._cache), self._max_keys,
                                 memory.getsizeof(self._cache) / 8, self._max_bytes))

    def _enforce_limits(self, new_value):
        additional_size = memory.getsizeof(new_value) / 8
        while len(self._cache) > 0 and not (len(self._cache) <= self._max_keys and
                                            (memory.getsizeof(self._cache) + additional_size) / 8 <= self._max_bytes):
            self.logger.debug('shrinking limited memory cache')
            self._cache.popitem(last=False)

    def set(self, key, value):
        self._enforce_limits(value)
        self._cache[key] = value

    def delete(self, key):
        self._cache.pop(key)


class LimitedFileBackend(DBMBackend, BasicInterface):

    def __init__(self, argument_dict):
        '''If argument_dict contains a value for max_keys this maximum amount of cache values kept in the
        internal cache file, otherwise its set to sys.maxlen.
        If necessary values are deleted from the cache in FIFO order.
        '''
        argument_dict['filename'] = argument_dict.get('filename', os.path.join(gettempdir(), 'pymor'))
        super(LimitedFileBackend, self).__init__(argument_dict)
        self.logger.debug('LimitedFileBackend args {}'.format(pformat(argument_dict)))
        self._max_keys = argument_dict.get('max_keys', sys.maxsize)
        self._keylist_fn = self.filename + '.keys'
        self._max_size = argument_dict.get('max_size', None)
        try:
            self._keylist, self._size = pymor.core.load(open(self._keylist_fn, 'rb'))
        except:
            self._keylist = deque()
            self._size = 0
        self._enforce_limits(None)
        self.print_limit()

    def _dump_keylist(self):
        pymor.core.dump((self._keylist, self._size), open(self._keylist_fn, 'wb'))

    def _new_key(self, key, size):
        self._keylist.append((key, size))
        self._size += size
        self._dump_keylist()

    def get(self, key):
        return super(LimitedFileBackend, self).get(key)

    def print_limit(self, additional_size=0):
        self.logger.info('LimitedFileBackend at {}({}) keys, total size: {}({})'
                         .format(len(self._keylist), self._max_keys, self._size, self._max_size))

    def _enforce_limits(self, new_value):
        while len(self._keylist) > 0 and (not (len(self._keylist) <= self._max_keys)
                                          or self._max_size and self._size > self._max_size):
            self.logger.debug('shrinking limited file cache')
            key, size = self._keylist.popleft()
            self.delete(key)
            self._size -= size

    def set(self, key, value):
        self._enforce_limits(value)
        value = pymor.core.dumps(value)
        if not key in self._keylist:
            self._new_key(key, len(value))
        with self._dbm_file(True) as dbm:
            dbm[key] = value

    def delete(self, key):
        super(LimitedFileBackend, self).delete(key)
        try:
            #api says this method is supposed to be idempotent
            self._keylist.remove(key)
        except ValueError:
            pass
        self._dump_keylist()

dc.register_backend("LimitedMemory", "pymor.core.dogpile_backends", "LimitedMemoryBackend")
dc.register_backend("LimitedFile", "pymor.core.dogpile_backends", "LimitedFileBackend")
dc.register_backend("Dummy", "pymor.core.dogpile_backends", "DummyBackend")
